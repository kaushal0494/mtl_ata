import os
import logging
import pandas as pd
from datasets import Dataset
from utils.constants import SEED

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

def oversample_dataset(df, method):
    """
    Apply oversampling (Random, SMOTE, ADASYN) to balance dataset.

    Args:
        df (pd.DataFrame): Input dataframe with 'label' column.
        method (str or None): One of ["random", "smote", "adasyn", None].

    Returns:
        pd.DataFrame: Oversampled dataframe.
    """
    if method is None:
        return df  # no oversampling

    X = df.drop(columns=["label"])
    y = df["label"]

    if method == "random":
        sampler = RandomOverSampler(random_state=SEED)
    elif method == "smote":
        sampler = SMOTE(random_state=SEED)
    elif method == "adasyn":
        sampler = ADASYN(random_state=SEED)
    else:
        raise ValueError(f"Unknown oversampling method: {method}")

    X_res, y_res = sampler.fit_resample(X, y)
    df_resampled = pd.concat([X_res, y_res], axis=1)
    return df_resampled

def load_datasets(data_dir, dimensions, shuffle=True, oversample_method=None):
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
    if oversample_method:
        logger.info("⚖️ Oversampling method: %s", oversample_method.upper())
    else:
        logger.info("⚖️ Oversampling method: None (original data)")

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
        logger.info(f"   🏷️ Training label distribution: {train_df['annotation'].value_counts().to_dict()} | Evaluation label distribution: {eval_df['annotation'].value_counts().to_dict()}")

        # Apply oversampling if requested
        if oversample_method:
            try:
                train_df = oversample_dataset(train_df, method=oversample_method)
                logger.info(f"   ✓ After {oversample_method.upper()} oversampling: {len(train_df)}")
                logger.info(f"   🏷️ Training label distribution after {oversample_method.upper()}: {train_df['annotation'].value_counts().to_dict()}")
            except ValueError as e:
                logger.warning(f"Skipping oversampling for {dim}: {e}")

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