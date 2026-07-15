"""
Candidate generation: retrieve similar news articles for a user vector
using the FAISS index.
"""

import faiss
import numpy as np
import pandas as pd

from src import config
from src.faiss_index import search


def recommend_candidates(
    user_vector: np.ndarray,
    index: faiss.Index,
    news_df: pd.DataFrame,
    top_k: int = config.CANDIDATE_POOL_SIZE,
) -> pd.DataFrame:
    """
    Retrieve the top_k most similar news articles to a user vector.

    Returns a DataFrame of candidate news rows with an added
    'similarity_score' column, sorted by similarity descending.
    """
    if user_vector is None:
        raise ValueError("user_vector is None — use the cold-start path instead.")

    top_k = min(top_k, index.ntotal)
    scores, idxs = search(index, user_vector, top_k=top_k)

    valid_mask = idxs >= 0
    idxs = idxs[valid_mask]
    scores = scores[valid_mask]

    candidates = news_df.iloc[idxs].copy().reset_index(drop=True)
    candidates["similarity_score"] = scores

    return candidates