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

from utils.data_loader import load_datasets
from utils.prompt import DatasetFormatter
from utils.constants import TARGET_MODULES_MAP, SEED
from utils.argparse import parse_args

set_seed(SEED)

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

class ModelTrainer:
    """Handles the complete model training pipeline"""

    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _setup_logging(self):
        """Log training configuration"""
        logger.info(Fore.CYAN + "\n========== 🚀 Starting LoRA Fine-Tuning ==========\n")
        logger.info(f"Model: {self.args.model_name}")
        logger.info(f"Input data directory: {self.args.data_dir}")
        logger.info(f"Selected evaluation dimensions: {self.args.dimensions}")
        logger.info(f"Output directory: {self.args.output_dir}")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Epochs: {self.args.epochs}")
        logger.info(f"Learning rate: {self.args.learning_rate}")

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

    def _apply_lora(self, model, r=8, lora_alpha=32, lora_dropout=0.05):
        """Apply LoRA configuration to the model using parameters from args"""
        logger.info(f"Applying LoRA configuration: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(TARGET_MODULES_MAP.keys()),
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

        # Freeze base model parameters
        for name, param in model.named_parameters():
            if "lora" not in name.lower():  # Only LoRA params remain trainable
                param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}, Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        return model

    def train(self):
        """Execute the complete training pipeline"""
        self._setup_logging()

        # Load and prepare data
        train_dataset, eval_dataset = load_datasets(self.args.data_dir, self.args.dimensions)
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
        self.model = self._apply_lora(
            self.model,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout
        )

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
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_strategy="steps",
            eval_steps=self.args.eval_steps,
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