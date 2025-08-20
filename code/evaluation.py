import argparse
import logging
import os
import torch
import pandas as pd
from collections import defaultdict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    set_seed
)
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from colorama import Fore, init as colorama_init
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize colorama
colorama_init(autoreset=True)

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
LABEL_LIST = ["Yes", "To some extent", "No"]
MAX_RETRIES = 3

logger = logging.getLogger(__name__)

class EvaluationDatasetFormatter:
    """Handles formatting datasets for model training, ready for LLaMA / LoRA fine-tuning"""

    SYSTEM_PROMPT = (
        "You are an expert evaluator of AI tutors. "
        "For the given ### Task, ### Task Definition, ### Label Definition, ### Conversation History and ### Tutor Response, assess the pedagogical appropriateness of the Tutor Response. "
        "Output exactly one label without additional text: Yes, No, or To some extent"
    )

    # Task definitions
    TASK_DEFINITIONS = {
        "mistake_identification": "Has the tutor identified/recognized a mistake in a student’s response?",
        "mistake_location": "Does the tutor’s response accurately point to a genuine mistake and its location?",
        "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?",
        "actionability": "Is it clear from the tutor’s feedback what the student should do next?"
    }

    # Optional label definitions per task
    LABEL_DEFINITIONS = {
        "mistake_identification": {
            "Yes": "The tutor correctly identified the mistake in the student’s response.",
            "To some extent": "The tutor partially recognized the mistake but did not fully capture it.",
            "No": "The tutor failed to identify any mistake."
        },
        "mistake_location": {
            "Yes": "The tutor accurately points to the exact mistake and its location.",
            "To some extent": "The tutor points to a mistake but imprecisely or partially.",
            "No": "The tutor fails to indicate the mistake or its location."
        },
        "providing_guidance": {
            "Yes": "The tutor provides correct and relevant guidance, hints, examples, or explanation.",
            "To some extent": "The guidance is partially correct or not fully helpful.",
            "No": "The tutor fails to provide relevant guidance."
        },
        "actionability": {
            "Yes": "It is clear what the student should do next.",
            "To some extent": "The next steps are somewhat unclear or incomplete.",
            "No": "The feedback does not indicate any actionable steps."
        }
    }

    @classmethod
    def create_dataset(cls, batch: dict, tokenizer: AutoTokenizer, max_length: int = 2048, include_label_definitions: bool = False) -> dict:
        logger.debug(f"Formatting {len(batch['conversation'])} samples.")
        samples = []
        gold_labels = []
        conversation_id = []
        conversation_history = []
        current_response = []
        task_type = []

        for cid, conv, resp, label, task in zip(
            batch["id"],
            batch["conversation"],
            batch["response"],
            batch["annotation"],
            batch["task"]
        ):
            task_def = cls.TASK_DEFINITIONS.get(task.lower(), "No definition available for this task.")

            # Prepare label definitions string if needed
            label_def_str = ""
            # if include_label_definitions:
            label_defs = cls.LABEL_DEFINITIONS.get(task.lower(), {})
            if label_defs:
                label_lines = [f"- {k}: {v}" for k, v in label_defs.items()]
                label_def_str = "\n".join(label_lines) + "\n\n"

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{cls.SYSTEM_PROMPT}\n\n"
                        f"### Task: {task}\n"
                        f"### Task Definition: {task_def}\n\n"
                        # f"### Label Definition: \n{label_def_str}"
                        f"### Conversation History:\n{conv.strip()}\n\n"
                        f"### Tutor Response:{resp.strip()}\n\n"
                        f"Now provide the classification label."
                    ),
                },
                # {"role": "assistant", "content": label},
            ]

            text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            samples.append(text)
            gold_labels.append(label)
            conversation_id.append(cid)
            conversation_history.append(conv)
            current_response.append(resp)
            task_type.append(task)

        return Dataset.from_dict({"text": samples, "gold_labels": gold_labels, "conversation_id": conversation_id, "conversation_history": conversation_history, "current_response": current_response, "task_type": task_type})

def load_model_and_tokenizer(base_model_name, adapter_path=None, enable_lora=False):
    """Load base model and optionally merge LoRA adapters"""
    logger.info(f"Loading base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    if enable_lora and adapter_path:
        logger.info(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        try:
            model = model.merge_and_unload()
            logger.info("Successfully merged LoRA adapters")
        except Exception as e:
            logger.error(f"Adapter merging failed: {str(e)}")
            logger.info("Proceeding with unmerged adapters")
    else:
        logger.info("No LoRA adapters to load")

    return model.to(device), tokenizer

def evaluate(args):
    logger.info(Fore.CYAN + "\n========== 🚀 Starting Evaluation ==========\n")
    set_seed(42)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_path, args.enable_lora == "True")

    logger.info(f"Loading evaluation data from {args.eval_csv}")
    if args.num_examples <= 0:
        eval_df = pd.read_csv(args.eval_csv)
    else:
        eval_df = pd.read_csv(args.eval_csv, nrows=args.num_examples)

    eval_dataset = EvaluationDatasetFormatter.create_dataset(eval_df, tokenizer)

    results = []
    logger.info("Running inference...")
    for example in tqdm(eval_dataset, desc="Evaluating"):
        with torch.no_grad():
            torch.cuda.empty_cache()

            inputs = tokenizer(example["text"], return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                # temperature=0.0,
                top_k=50,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = decoded.strip().split('model')[-1].strip().split('\n')[0]
            # print(f"Generated part: {generated_part}")

            results.append({
                "id": example["conversation_id"],
                "history": example["conversation_history"],
                "response": example["current_response"],
                "task": example["task_type"],
                "prediction": generated_part,
                "gold_label": example["gold_labels"]
            })

    # ✅ Save results as JSON in args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, f"{os.path.basename(args.output_dir)}_predictions.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(Fore.YELLOW + f"Saved predictions to {output_json}")

    # === Prepare output file ===
    output_file = os.path.join(args.output_dir, f"{os.path.basename(args.output_dir)}_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:

        # === Overall Metrics ===
        preds = [r["prediction"] for r in results]
        labels = [r["gold_label"] for r in results]

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, labels=LABEL_LIST, average="macro", zero_division=0)
        report = classification_report(labels, preds, labels=LABEL_LIST, zero_division=0, digits=4)
        cm = confusion_matrix(labels, preds, labels=LABEL_LIST)

        header = "\n" + "="*30 + " Overall Evaluation " + "="*30
        overall_str = (
            f"{header}\n"
            f"Accuracy: {acc:.4f}\n"
            f"Macro F1: {macro_f1:.4f}\n\n"
            f"Classification Report:\n{report}\n\n"
            f"Labels: {LABEL_LIST}\n\n"
            f"Confusion Matrix (rows=true, cols=pred):\n{cm}\n"
        )

        logger.info(Fore.GREEN + overall_str)
        f.write(overall_str + "\n")

        # === Per-Task Metrics ===
        task_groups = defaultdict(list)
        for r in results:
            task_groups[r["task"]].append(r)

        for task, task_results in task_groups.items():
            task_preds = [r["prediction"] for r in task_results]
            task_labels = [r["gold_label"] for r in task_results]

            task_acc = accuracy_score(task_labels, task_preds)
            task_macro_f1 = f1_score(task_preds, task_labels, labels=LABEL_LIST, average="macro", zero_division=0)
            task_report = classification_report(task_labels, task_preds, labels=LABEL_LIST, zero_division=0, digits=4)
            task_cm = confusion_matrix(task_labels, task_preds, labels=LABEL_LIST)

            task_header = "\n" + "-"*25 + f" Task: {task} " + "-"*25
            task_str = (
                f"{task_header}\n"
                f"Accuracy: {task_acc:.4f}\n"
                f"Macro F1: {task_macro_f1:.4f}\n\n"
                f"Classification Report:\n{task_report}\n\n"
                f"Labels: {LABEL_LIST}\n\n"
                f"Confusion Matrix (rows=true, cols=pred):\n{task_cm}\n"
            )

            logger.info(Fore.CYAN + task_str)
            f.write(task_str + "\n")

    logger.info(Fore.YELLOW + f"\nSaved evaluation results to {output_file}")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Model Evaluation")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--eval_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--enable_lora", type=str, default="True")
    args = parser.parse_args()
    evaluate(args)