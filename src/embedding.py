"""
Semantic embedding generation using Sentence Transformers.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src import config
from src.utils import logger

_model_cache = {}


def load_model(model_name: str = config.EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load (and cache) the Sentence Transformer model."""
    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name, device=config.DEVICE)
    return _model_cache[model_name]


def encode_news(
    news_df: pd.DataFrame,
    model: SentenceTransformer = None,
    text_column: str = "text",
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Encode a news DataFrame's text column into dense embeddings.

    Does NOT save to disk — use `build_and_save_embeddings` for that.
    """
    if text_column not in news_df.columns:
        raise ValueError(f"Column '{text_column}' not found in news DataFrame.")

    model = model or load_model()

    embeddings = model.encode(
        news_df[text_column].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize at encode time for cosine/IP consistency
    )

    if embeddings.shape[0] != len(news_df):
        raise ValueError(
            f"Embedding count mismatch: got {embeddings.shape[0]}, expected {len(news_df)}"
        )
    if np.isnan(embeddings).any():
        raise ValueError("Generated embeddings contain NaN values.")

    return embeddings.astype("float32")


def build_and_save_embeddings(
    news_df: pd.DataFrame,
    model: SentenceTransformer = None,
    save_path: Union[str, Path] = config.NEWS_EMBEDDINGS,
) -> np.ndarray:
    """Encode news articles and persist the resulting embedding matrix to disk."""
    embeddings = encode_news(news_df, model=model)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embeddings)
    logger.info(f"Saved news embeddings {embeddings.shape} to {save_path}")

    return embeddings


def load_embeddings(path: Union[str, Path] = config.NEWS_EMBEDDINGS) -> np.ndarray:
    """Load a previously saved embedding matrix."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return np.load(path)