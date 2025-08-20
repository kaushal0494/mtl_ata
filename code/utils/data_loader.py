import os
import logging
import pandas as pd
from datasets import Dataset
from utils.constants import SEED

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )


def load_datasets(data_dir, dimensions, shuffle=True):
    """
    Load, concatenate, and optionally shuffle training and evaluation datasets
    for the specified dimensions.

    Args:
        data_dir (str): Directory containing train/dev CSV files.
        dimensions (list[str]): List of dimensions to load, e.g. ["MI", "ML", "PG", "AC"].
        shuffle (bool, optional): Whether to shuffle the datasets. Defaults to True.

    Returns:
        train_dataset, eval_dataset (datasets.Dataset): HuggingFace Dataset objects.
    """

    logger.info("Selected Dimensions: %s", dimensions)

    train_dfs = []
    eval_dfs = []

    logger.info("Loading and preparing datasets...")

    for dim in dimensions:
        train_path = os.path.join(data_dir, f"{dim}_train.csv")
        eval_path = os.path.join(data_dir, f"{dim}_dev.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

        logger.info(f"📂 Loading dimension: {dim}")
        logger.info(f"   → Training file: {train_path}")
        logger.info(f"   → Evaluation file: {eval_path}")

        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(eval_path)

        train_df["dimension"] = dim
        eval_df["dimension"] = dim

        logger.info(f"   ✓ Training samples: {len(train_df)} | Evaluation samples: {len(eval_df)}")

        train_dfs.append(train_df)
        eval_dfs.append(eval_df)

    train_all = pd.concat(train_dfs, ignore_index=True)
    eval_all = pd.concat(eval_dfs, ignore_index=True)

    if shuffle:
        logger.info("🔀 Shuffling datasets...")
        train_all = train_all.sample(frac=1, random_state=SEED).reset_index(drop=True)
        eval_all = eval_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_all)
    eval_dataset = Dataset.from_pandas(eval_all)

    logger.info(f"✅ Final training dataset size: {len(train_dataset)}")
    logger.info(f"✅ Final evaluation dataset size: {len(eval_dataset)}")
    logger.info(f"✅ Included dimensions: {', '.join(dimensions)}")

    return train_dataset, eval_dataset