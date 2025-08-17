import argparse
import os
import logging
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import classification_report
from colorama import Fore, init as colorama_init

colorama_init(autoreset=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.DEBUG,  # DEBUG to get more info
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

LABEL_LIST = ["Yes", "No", "To some extent"]


# ---------- Data Loading ----------
def load_dataframe(path, name):
    logger.debug(f"Attempting to load {name} CSV from: {path}")
    if not os.path.exists(path):
        logger.error(Fore.RED + f"{name} CSV file not found: {path}")
        raise FileNotFoundError(f"{name} CSV file not found: {path}")
    df = pd.read_csv(path)
    logger.debug(f"First 2 rows of {name}:\n{df.head(2)}")
    for col in ["conversation", "response", "annotation"]:
        if col not in df.columns:
            raise ValueError(f"'{col}' column missing in {name} CSV")
    logger.info(Fore.GREEN + f"{name} CSV loaded successfully with {len(df)} rows.")
    return df


# ---------- Chat Template Formatting ----------

# def format_prompt(conversation: str, response: str, label: str) -> list[dict]:
#     """
#     Formats the input into a chat-style prompt for Gemma-3B-it inference.

#     Args:
#         conversation (str): The previous conversation history between tutor and student.
#         response (str): The tutor's latest response to be evaluated.
#         label (str): The expected classification label ("Yes", "No", or "To some extent").

#     Returns:
#         list[dict]: A list of message dictionaries formatted for the model's chat template.
#     """
#     system_prompt = (
#         "You are an expert evaluator of AI tutors. "
#         "Given the previous conversation history between a tutor and a student, "
#         "classify the tutor's response based on its pedagogical appropriateness. "
#         "Possible labels: Yes, No, To some extent. "
#         "Output only one label exactly as written, without explanations, punctuation, or additional text."
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {
#             "role": "user",
#             "content": (
#                 f"### Context:\n{conversation.strip()}\n\n"
#                 f"### Tutor Response:\n{response.strip()}\n\n"
#                 f"### Expected Label:\n{label.strip()}"
#             )
#         }
#     ]

#     return messages


# def create_dataset(batch, tokenizer):
#     """
#     Formats a batch of samples into chat-style prompts with gold labels.

#     Args:
#         batch (dict): A batch dict with keys 'conversation', 'response', 'annotation' (all lists).
#         tokenizer: Hugging Face tokenizer with apply_chat_template method.

#     Returns:
#         dict: dict with keys 'text' (list of formatted prompt strings) and 'gold_labels' (list of labels).
#     """
#     samples = []
#     gold_labels = []
    
#     for conv, resp, label in zip(batch["conversation"], batch["response"], batch["annotation"]):
#         messages = format_prompt(conv, resp, label)
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         samples.append(text)
#         gold_labels.append(label)
    
#     # Optional debugging output for first few samples
#     num_debug_samples = 3
#     for i in range(min(num_debug_samples, len(samples))):
#         logger.debug(f"Sample {i} prompt:\n{samples[i]}")
#         logger.debug(f"Sample {i} label: {gold_labels[i]}")

#     return {
#         "text": samples,
#         "gold_labels": gold_labels,
#     }


def create_dataset(batch, tokenizer):
    logger.debug(f"Formatting {len(batch['conversation'])} samples with chat template.")
    system_prompt = (
        "You are an expert evaluator of AI tutors. "
        "Given the previous conversation history between a tutor and a student, "
        "classify the tutor's response based on its pedagogical appropriateness. "
        "Possible labels: Yes, No, To some extent. "
        "Output only one label exactly as written, without explanations, punctuation, or additional text."
    )

    samples = []
    for i, (conv, resp, label) in enumerate(zip(batch["conversation"], batch["response"], batch["annotation"])):
        if i < 1:
            logger.debug(f"Original sample #{i}:\nConversation: {conv}\nResponse: {resp}\nLabel: {label}")

        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"### Context:\n{conv.strip()}\n\n### Tutor Response:\n{resp.strip()}\n\n### ### Expected Label:\n{label.strip()}"},
            # {"role": "assistant", "content": label}
        ]
        # tokenizer.chat_template = (
        #     "{% set loop_messages = messages %}"
        #     "{% for message in loop_messages %}"
        #     "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
        #     "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        #     "{{ content }}"
        #     "{% endfor %}"
        #     "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        # )
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        if i < 1:
            logger.debug(f"Formatted sample #{i}:\n{text[:300]}...")  # Only show first 300 chars
        samples.append(text)

    return {"text": samples}

# ---------- Training ----------
def train(args):
    logger.info(Fore.CYAN + "\n========== 🚀 Starting LoRA Fine-Tuning ==========\n")

    # Load data
    train_df = load_dataframe(args.train_csv, "Training")
    eval_df = load_dataframe(args.eval_csv, "Evaluation")

    # Convert to datasets
    logger.debug("Converting DataFrames to HuggingFace Datasets...")
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Load tokenizer
    logger.debug(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.debug(f"Tokenizer vocab size: {len(tokenizer)}")

    # Apply chat template
    logger.debug("Applying chat template to training set...")
    train_dataset = train_dataset.map(lambda x: create_dataset(x, tokenizer), batched=True)
    logger.debug("Applying chat template to evaluation set...")
    eval_dataset = eval_dataset.map(lambda x: create_dataset(x, tokenizer), batched=True)

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='eager',
    )
    logger.debug("Model loaded successfully.")

    # print("Has peft_config?", hasattr(model, "peft_config"))

    if hasattr(model, "peft_config"):
        logger.info("Unloading existing PEFT adapters before applying new ones.")
        model = model.unload()

    # print("Has peft_config?", hasattr(model, "peft_config"))

    # LoRA target modules map
    TARGET_MODULES_MAP = {
        "q_proj": "Query projection in attention layers",
        "k_proj": "Key projection in attention layers",
        "v_proj": "Value projection in attention layers",
    }
    logger.debug(f"Using LoRA target modules: {TARGET_MODULES_MAP}")

    # LoRA config
    peft_config = LoraConfig(
        base_model_name_or_path=args.model_name,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=list(TARGET_MODULES_MAP.keys()),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    # # Freeze all base model parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    #number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable params: {trainable_params:,}, Trainable %: {100 * trainable_params / total_params:.4f}%")


    # SFT config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        bf16=True
    )

    logger.debug("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        # peft_config=peft_config,
    )

    logger.info(Fore.YELLOW + "Starting training...")
    trainer.train()
    logger.info(Fore.GREEN + "Training completed.")

    logger.info("Saving model and tokenizer...")
    trainer.save_model(os.path.join(args.output_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "lora_model"))
    logger.info(Fore.GREEN + f"✅ LoRA model saved at: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Instruction Tuning with Chat Template & TRL")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--eval_csv", type=str, default="validation.csv")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()


    if args.train:
        train(args)