import os
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from colorama import Fore, init as colorama_init
from sklearn.utils.class_weight import compute_class_weight

set_seed(42)

LABEL_LIST = ["Yes", "No", "To some extent"]

# Initialize colorama for colored console output
colorama_init(autoreset=True)

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# Reduce verbosity for some libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
LABEL_LIST = ["Yes", "To some extent", "No"]

class DataLoader:
    """Handles loading and validation of training data"""
    
    @staticmethod
    def load_dataframe(path: str, name: str) -> pd.DataFrame:
        """Load and validate a CSV file containing training data"""
        logger.debug(f"Attempting to load {name} CSV from: {path}")
        if not os.path.exists(path):
            logger.error(Fore.RED + f"{name} CSV file not found: {path}")
            raise FileNotFoundError(f"{name} CSV file not found: {path}")
            
        df = pd.read_csv(path)
        logger.debug(f"First 2 rows of {name}:\n{df.head(2)}")
        
        # Validate required columns
        required_columns = ["conversation", "response", "annotation"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"'{col}' column missing in {name} CSV")
                
        logger.info(Fore.GREEN + f"{name} CSV loaded successfully with {len(df)} rows.")
        return df

# Single task dataformattor 
# class DatasetFormatter:
#     """Handles formatting datasets for model training, ready for LLaMA / LoRA fine-tuning"""

#     SYSTEM_PROMPT = (
#         "As an expert evaluator of AI tutors, assess the pedagogical appropriateness of the tutor's response. "
#         "You will receive:\n"
#         "1. ### Conversation History showing student mistakes/confusion\n"
#         "2. ### Tutor Response attempting to address these issues\n\n"
#         # "Classification Criteria:\n"
#         # "- 'Yes': Fully addresses confusion with effective teaching strategies\n"
#         # "- 'No': Fails to address or exacerbates misunderstandings\n"
#         # "- 'To some extent': Partially effective but with notable deficiencies\n\n"
#         "Output exactly one label without additional text: Yes, No, or To some extent"
#     )

    # @classmethod
    # def create_dataset(cls, batch: dict, tokenizer: AutoTokenizer, max_length: int = 512) -> dict:
    #     """
    #     Formats a batch of examples into chat-style prompts and tokenizes them for LLaMA / causal LM training.
        
    #     Args:
    #         batch: Dictionary containing 'conversation', 'response', 'label' lists
    #         tokenizer: Hugging Face tokenizer
    #         max_length: Maximum sequence length for padding/truncation
        
    #     Returns:
    #         Dictionary with 'input_ids', 'attention_mask', 'labels' ready for training
    #     """
    #     logger.debug(f"Formatting {len(batch['conversation'])} samples.")
    #     texts = []
        
    #     for conv, resp, label in zip(batch["conversation"], batch["response"], batch["annotation"]):
    #         # Combine into chat prompt
    #         # messages = [
    #         #     {"role": "system", "content": cls.SYSTEM_PROMPT},
    #         #     {"role": "user", "content": f"### Conversation History:\n{conv.strip()}\n\n### Tutor Response:\n{resp.strip()}\n\n"},
    #         #     {"role": "assistant", "content": f"### Expected Label is:\n{label}"}
    #         # ]

    #         messages = [
    #             {
    #                 "role": "user", 
    #                 "content": (
    #                     f"{cls.SYSTEM_PROMPT}\n\n"
    #                     f"### Conversation History:\n{conv.strip()}\n\n"
    #                     f"### Tutor Response:\n{resp.strip()}\n\n"
    #                     f"Now provide the classification label."
    #                 )
    #             },
    #             {
    #                 "role": "assistant", 
    #                 "content": label
    #             },
    #         ]
            
    #         # Flatten messages into single string
    #         text = tokenizer.apply_chat_template(messages, tokenize=False)
    #         texts.append(text)
        
    #     # Tokenize the batch
    #     tokenized = tokenizer(
    #         texts,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors="pt"
    #     )
        
    #     # # Add labels for classification / evaluation
    #     tokenized["labels"] = tokenized["input_ids"]
    #     return tokenized

# Multi task data formatter
class DatasetFormatter:
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

# class for Focal loss
# class FocalLoss(torch.nn.Module):
#     """Focal Loss for multi-class classification"""
#     def __init__(self, gamma=2.0, weight=None, reduction="mean"):
#         super().__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction

#     def forward(self, logits, labels):
#         # logits: [batch, seq_len, vocab_size]
#         # labels: [batch, seq_len]
#         vocab_size = logits.size(-1)

#         # Flatten for CE
#         logits = logits.view(-1, vocab_size)
#         labels = labels.view(-1)

#         ce_loss = F.cross_entropy(
#             logits, labels,
#             weight=self.weight,
#             reduction="none"
#         )

#         pt = torch.exp(-ce_loss)  # probability of correct class
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss

#         if self.reduction == "mean":
#             return focal_loss.mean()
#         elif self.reduction == "sum":
#             return focal_loss.sum()
#         else:
#             return focal_loss

# class FocalLossCausalLM(AutoModelForCausalLM):
#     def __init__(self, config, gamma=2.0, weight=None, reduction="mean"):
#         super().__init__(config)
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction

#     def compute_focal_loss(self, logits, labels):
#         vocab_size = logits.size(-1)
#         logits = logits.view(-1, vocab_size)
#         labels = labels.view(-1)
#         ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
#         if self.reduction == "mean":
#             return focal_loss.mean()
#         elif self.reduction == "sum":
#             return focal_loss.sum()
#         else:
#             return focal_loss

#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
#         if labels is not None:
#             logits = outputs.logits
#             loss = self.compute_focal_loss(logits, labels)
#             outputs.loss = loss
#         return outputs

class ModelTrainer:
    """Handles the complete model training pipeline"""
    
    TARGET_MODULES_MAP = {
        "q_proj": "Query projection in attention layers",
        "k_proj": "Key projection in attention layers",
        "v_proj": "Value projection in attention layers",
    }

    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _setup_logging(self):
        """Log training configuration"""
        logger.info(Fore.CYAN + "\n========== 🚀 Starting LoRA Fine-Tuning ==========\n")
        logger.info(f"Model: {self.args.model_name}")
        logger.info(f"Training data: {self.args.train_csv}")
        logger.info(f"Evaluation data: {self.args.eval_csv}")
        logger.info(f"Output directory: {self.args.output_dir}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Epochs: {self.args.epochs}")
        logger.info(f"Learning rate: {self.args.learning_rate}")

    def _load_data(self):
        """Load and prepare training and evaluation data"""
        logger.info("Loading and preparing datasets...")
        train_df = DataLoader.load_dataframe(self.args.train_csv, "Training")
        eval_df = DataLoader.load_dataframe(self.args.eval_csv, "Evaluation")
        
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        return train_dataset, eval_dataset

    def _initialize_tokenizer(self):
        """Initialize and configure the tokenizer"""
        logger.info(f"Loading tokenizer: {self.args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        # Set up pad token for proper training
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        logger.debug(f"Tokenizer vocab size: {len(tokenizer)}")
        return tokenizer

    def _initialize_model(self):
        """Initialize and configure the base model"""
        logger.info(f"Loading model: {self.args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            attn_implementation="eager",
        )
        return model

    def _apply_lora(self, model):
        """Apply LoRA configuration to the model"""
        logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=list(self.TARGET_MODULES_MAP.keys()),
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)

        # Freeze base model parameters
        for name, param in model.named_parameters():
            if "lora" not in name.lower():  # Only LoRA params remain trainable
                param.requires_grad = False
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
        
        return model

    def train(self):
        """Execute the complete training pipeline"""
        self._setup_logging()
        
        # Load and prepare data
        train_dataset, eval_dataset = self._load_data()
        self.tokenizer = self._initialize_tokenizer()

        # Format datasets
        logger.info("Formatting training dataset...")
        train_dataset = train_dataset.map(
            lambda x: DatasetFormatter.create_dataset(x, self.tokenizer, self.args.max_length),
            batched=True,
        )
        
        logger.info("Formatting evaluation dataset...")
        eval_dataset = eval_dataset.map(
            lambda x: DatasetFormatter.create_dataset(x, self.tokenizer, self.args.max_length),
            batched=True,
        )
        # Initialize model
        self.model = self._initialize_model()
        self.model = self._apply_lora(self.model)


        # # Compute weights from training annotations for Focual loss
        # class_weights = compute_class_weight(
        #     class_weight="balanced",
        #     classes=np.array(LABEL_LIST),
        #     y=list(train_dataset["annotation"])
        # )
        # weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        # logger.info(f"Classes: {LABEL_LIST}")
        # logger.info(f"Weight Tensor: {weight_tensor}")

        # # Pass into FocalLoss
        # focal_loss = FocalLoss(gamma=2.0, weight=weight_tensor.to("cuda"))

        # Configure training
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            warmup_ratio=0.03,   # 3% of total training steps
            weight_decay=self.args.weight_decay,
            logging_steps=100,
            save_steps=500,
            eval_strategy="steps",
            eval_steps=500,
            save_total_limit=1,
            load_best_model_at_end=True,
            bf16=True,           # enable BF16 mixed precision
            fp16=False,          # ensure FP16 is off
            report_to="none",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            resume_from_checkpoint=False,
            overwrite_output_dir=True,
            max_grad_norm=1.0,  # clip gradients
        )
        

        # Initialize trainer
        logger.info("Initializing SFTTrainer...")
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            # loss_func=loss_func #focual loss
        )

        # Start training
        logger.info(Fore.YELLOW + "Starting training...")
        self.trainer.train()
        logger.info(Fore.GREEN + "Training completed.")

        # Save results
        self._save_model()

    def _save_model(self):
        """Save the trained model and tokenizer"""
        output_dir = os.path.join(self.args.output_dir, "lora_model")
        logger.info(f"Saving model and tokenizer to: {output_dir}")
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(Fore.GREEN + f"✅ LoRA model saved successfully at: {output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning for Pedagogical Response Classification"
    )
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--model_name", type=str,
                       help="Base model to fine-tune")
    parser.add_argument("--train_csv", type=str, default="train.csv",
                       help="Path to training data CSV")
    parser.add_argument("--eval_csv", type=str, default="validation.csv",
                       help="Path to evaluation data CSV")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save trained model")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    if args.train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        trainer = ModelTrainer(args)
        trainer.train()
    else:
        logger.warning("No action specified. Use --train to start training.")


if __name__ == "__main__":
    main()