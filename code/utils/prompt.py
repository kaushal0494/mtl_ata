import logging
from transformers import AutoTokenizer
from colorama import Fore

logger = logging.getLogger(__name__)

class DatasetFormatter:
    """Handles formatting datasets for model training, ready for LLaMA / LoRA fine-tuning"""

    SYSTEM_PROMPT = (
        "You are an expert evaluator of AI tutors. "
        "For the given ### Task, ### Task Definition, ### Label Definition, ### Conversation History and ### Tutor Response, assess the pedagogical appropriateness of the Tutor Response. "
        "Output exactly one label without additional text: Yes, No, or To some extent"
    )

    TASK_DEFINITIONS = {
        "mistake_identification": "Has the tutor identified/recognized a mistake in a student’s response?",
        "mistake_location": "Does the tutor’s response accurately point to a genuine mistake and its location?",
        "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?",
        "actionability": "Is it clear from the tutor’s feedback what the student should do next?"
    }

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
    def create_dataset(cls, batch: dict, tokenizer: AutoTokenizer, max_length: int = 512, include_label_definitions: bool = False) -> dict:
        logger.debug(f"Formatting {len(batch['conversation'])} samples.")
        texts = []
        
        for conv, resp, label, task in zip(
            batch["conversation"],
            batch["response"],
            batch["annotation"],
            batch["task"]
        ):
            task_def = cls.TASK_DEFINITIONS.get(task.lower(), "No definition available for this task.")

            label_def_str = ""
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
                        f"### Label Definition: \n{label_def_str}"
                        f"### Conversation History:\n{conv.strip()}\n\n"
                        f"### Tutor Response:{resp.strip()}\n\n"
                        f"Now provide the classification label."
                    ),
                },
                {"role": "assistant", "content": label},
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized