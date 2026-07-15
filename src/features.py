"""
Feature engineering for the learned (user, candidate article) click-through
ranker. The same feature-building logic is used at training time and at
inference time, so predictions are never computed on a distribution
different from what the model was trained on.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import config


def _category_affinity(history_categories: pd.Series, category: str) -> float:
    """Fraction of a user's history that belongs to `category`."""
    if history_categories.empty:
        return 0.0
    return float((history_categories == category).mean())


def build_feature_row(
    similarity_score: float,
    popularity_score: float,
    candidate_category: str,
    candidate_subcategory: str,
    history_categories: pd.Series,
    history_subcategories: pd.Series,
    history_length: int,
    max_similarity_to_history: float,
    title_length: int,
    abstract_length: int,
) -> Dict[str, float]:
    """Assemble a single feature dict for one (user, candidate) pair."""
    return {
        "similarity_score": float(similarity_score),
        "popularity_score": float(popularity_score),
        "category_match": float(candidate_category in set(history_categories)),
        "subcategory_match": float(candidate_subcategory in set(history_subcategories)),
        "category_affinity": _category_affinity(history_categories, candidate_category),
        "history_length": float(history_length),
        "max_similarity_to_history": float(max_similarity_to_history),
        "title_length": float(title_length),
        "abstract_length": float(abstract_length),
    }


def build_features_for_candidates(
    history_ids: List[str],
    candidates_df: pd.DataFrame,
    news_df: pd.DataFrame,
    news_id_to_idx: Dict[str, int],
    news_embeddings: np.ndarray,
    popularity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a feature table for a batch of candidate articles for one user.

    `candidates_df` must contain NewsID, Category, SubCategory, Title,
    Abstract, and similarity_score (from FAISS retrieval).
    """
    hist_idxs = [news_id_to_idx[i] for i in history_ids if i in news_id_to_idx]
    history_rows = news_df[news_df["NewsID"].isin(history_ids)]
    history_categories = history_rows["Category"]
    history_subcategories = history_rows["SubCategory"]
    history_length = len(hist_idxs)

    pop_map = dict(zip(popularity_df["NewsID"], popularity_df["popularity_score"]))

    rows = []
    for _, cand in candidates_df.iterrows():
        cand_idx = news_id_to_idx.get(cand["NewsID"])
        if cand_idx is not None and hist_idxs:
            max_sim = float(np.max(news_embeddings[hist_idxs] @ news_embeddings[cand_idx]))
        else:
            max_sim = 0.0

        row = build_feature_row(
            similarity_score=cand.get("similarity_score", 0.0),
            popularity_score=pop_map.get(cand["NewsID"], 0.0),
            candidate_category=cand.get("Category", ""),
            candidate_subcategory=cand.get("SubCategory", ""),
            history_categories=history_categories,
            history_subcategories=history_subcategories,
            history_length=history_length,
            max_similarity_to_history=max_sim,
            title_length=len(str(cand.get("Title", ""))),
            abstract_length=len(str(cand.get("Abstract", ""))),
        )
        row["NewsID"] = cand["NewsID"]
        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_matrix(
    feature_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract the ordered numeric feature matrix used by the ranker models."""
    feature_columns = feature_columns or config.RANKER_FEATURE_COLUMNS
    missing = [c for c in feature_columns if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return feature_df[feature_columns].to_numpy(dtype="float32")