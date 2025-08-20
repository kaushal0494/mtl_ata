# args_parser.py
"""
Argument parser for LoRA Fine-Tuning on Pedagogical Response Classification.

Provides all configurable hyperparameters and training options.
"""

import argparse

def parse_args():
    """Parse command-line arguments for training and model configuration."""

    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning for Pedagogical Response Classification"
    )

    # ----------------------
    # General options
    # ----------------------
    parser.add_argument(
        "--train", action="store_true",
        help="Run training pipeline"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Base model to fine-tune (HuggingFace model name)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing train/dev/test CSV files"
    )
    parser.add_argument(
        "--dimensions", nargs="+", default=None,
        help="List of dimensions to include (e.g., MI ML PG AC)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save trained model and checkpoints"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=16,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay coefficient"
    )

    # ----------------------
    # LoRA configuration
    # ----------------------
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA rank r"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout probability"
    )

    # ----------------------
    # Logging and checkpointing
    # ----------------------
    parser.add_argument(
        "--logging_steps", type=int, default=100,
        help="Steps interval between logging training metrics"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Steps interval between saving checkpoints"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Steps interval between evaluation"
    )

    return parser.parse_args()
