"""
User profile construction from click history.

A user's profile vector is the L2-normalized mean of the embeddings of the
news articles in their click history. Users with no history (cold-start)
are excluded here and handled separately by the ranking module's fallback.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src import config
from src.utils import logger


def build_news_id_index(
    news_df: pd.DataFrame,
    save_path: Union[str, Path] = config.NEWS_ID_TO_IDX_PATH,
) -> Dict[str, int]:
    """Map NewsID -> row index in the embeddings array, and persist it."""
    if "NewsID" not in news_df.columns:
        raise ValueError("news_df must contain a 'NewsID' column.")

    mapping = {nid: idx for idx, nid in enumerate(news_df["NewsID"].tolist())}

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(mapping, f)

    logger.info(f"Saved NewsID index ({len(mapping)} entries) to {save_path}")
    return mapping


def load_news_id_index(path: Union[str, Path] = config.NEWS_ID_TO_IDX_PATH) -> Dict[str, int]:
    """Load a previously saved NewsID -> index mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NewsID index file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_user_vector(
    history_str: str,
    news_id_to_idx: Dict[str, int],
    news_embeddings: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Build a user embedding by averaging the embeddings of clicked news
    articles, then L2-normalizing the result.

    Returns None if the user has no usable history (cold-start).
    """
    if not history_str:
        return None

    ids = history_str.split()
    idxs = [news_id_to_idx[i] for i in ids if i in news_id_to_idx]

    if not idxs:
        return None

    vec = news_embeddings[idxs].mean(axis=0)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype("float32") if norm > 0 else vec.astype("float32")


def build_user_profiles(
    behaviors_df: pd.DataFrame,
    news_id_to_idx: Dict[str, int],
    news_embeddings: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build a {UserID: profile_vector} dict for every user with usable history."""
    if "UserID" not in behaviors_df.columns or "History" not in behaviors_df.columns:
        raise ValueError("behaviors_df must contain 'UserID' and 'History' columns.")

    user_vectors: Dict[str, np.ndarray] = {}

    for _, row in behaviors_df.drop_duplicates(subset="UserID").iterrows():
        vec = get_user_vector(row["History"], news_id_to_idx, news_embeddings)
        if vec is not None:
            user_vectors[row["UserID"]] = vec

    logger.info(
        f"Built user profiles for {len(user_vectors)} / "
        f"{behaviors_df['UserID'].nunique()} unique users"
    )
    return user_vectors


def save_user_profiles(
    user_vectors: Dict[str, np.ndarray],
    matrix_path: Union[str, Path] = config.USER_EMBEDDINGS,
    ids_path: Union[str, Path] = config.USER_EMBEDDINGS_IDS,
) -> Tuple[np.ndarray, list]:
    """Persist user profile vectors and their corresponding UserIDs."""
    if not user_vectors:
        raise ValueError("user_vectors is empty — nothing to save.")

    user_ids = list(user_vectors.keys())
    matrix = np.stack([user_vectors[uid] for uid in user_ids]).astype("float32")

    matrix_path = Path(matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    pd.Series(user_ids, name="UserID").to_csv(ids_path, index=False)

    logger.info(f"Saved {matrix.shape[0]} user profiles to {matrix_path}")
    return matrix, user_ids