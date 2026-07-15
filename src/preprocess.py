"""
Preprocessing pipeline for the MIND news and behaviors datasets.

Responsibilities:
    - Load raw TSV files into DataFrames with proper column names.
    - Clean missing values.
    - Build the unified `text` field (Category + SubCategory + Title + Abstract)
      used as input to the embedding model.
    - Clean the behaviors log (drop rows with no impressions).
    - Persist processed CSVs to disk.
"""

from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from src import config
from src.utils import logger


def load_news(path: Union[str, Path] = config.TRAIN_NEWS) -> pd.DataFrame:
    """Load a MIND news.tsv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"News file not found: {path}")

    df = pd.read_csv(path, sep="\t", header=None, names=config.NEWS_COLUMNS)
    logger.info(f"Loaded news file: {path} -> {df.shape}")
    return df


def load_behaviors(path: Union[str, Path] = config.TRAIN_BEHAVIORS) -> pd.DataFrame:
    """Load a MIND behaviors.tsv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Behaviors file not found: {path}")

    df = pd.read_csv(path, sep="\t", header=None, names=config.BEHAVIORS_COLUMNS)
    logger.info(f"Loaded behaviors file: {path} -> {df.shape}")
    return df


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean news data and build the unified text field used for embedding.

    Combining Category + SubCategory with Title + Abstract gives the
    embedding model topical context in addition to the raw text.
    """
    df = df.copy()

    for col in ("Title", "Abstract", "Category", "SubCategory"):
        df[col] = df[col].fillna("")

    df["text"] = (
        df["Category"] + " " + df["SubCategory"] + ". "
        + df["Title"] + ". " + df["Abstract"]
    ).str.strip()

    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    if df["NewsID"].duplicated().any():
        n_dupes = df["NewsID"].duplicated().sum()
        logger.warning(f"Found {n_dupes} duplicate NewsIDs — dropping duplicates.")
        df = df.drop_duplicates(subset="NewsID").reset_index(drop=True)

    return df


def clean_behaviors(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the behaviors log: fill missing history, drop rows with no impressions."""
    df = df.copy()
    df["History"] = df["History"].fillna("")
    df = df[df["Impressions"].notna()].reset_index(drop=True)
    return df


def run_preprocessing(
    news_path: Union[str, Path] = config.TRAIN_NEWS,
    behaviors_path: Union[str, Path] = config.TRAIN_BEHAVIORS,
    save: bool = True,
    processed_news_path: Union[str, Path] = config.PROCESSED_NEWS,
    processed_behaviors_path: Union[str, Path] = config.PROCESSED_BEHAVIORS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: load -> clean -> (optionally) save.

    Returns (news_df, behaviors_df).
    """
    news = clean_news(load_news(news_path))
    behaviors = clean_behaviors(load_behaviors(behaviors_path))

    if news.empty:
        raise ValueError("Preprocessing produced an empty news DataFrame.")
    if behaviors.empty:
        raise ValueError("Preprocessing produced an empty behaviors DataFrame.")

    if save:
        Path(processed_news_path).parent.mkdir(parents=True, exist_ok=True)
        news.to_csv(processed_news_path, index=False)
        behaviors.to_csv(processed_behaviors_path, index=False)
        logger.info(f"Saved processed news to {processed_news_path}")
        logger.info(f"Saved processed behaviors to {processed_behaviors_path}")

    return news, behaviors