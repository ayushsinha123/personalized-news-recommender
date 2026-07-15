"""
Lightweight explainability helpers: surface *why* a recommendation was made
by comparing a user's history categories to the recommended categories, and
by finding the specific history article closest to each recommendation.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def explain_by_category(
    history_ids: List[str],
    recommended_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Compare category distribution of a user's history against the
    category distribution of their recommendations.
    """
    history_categories = news_df[news_df["NewsID"].isin(history_ids)]["Category"].value_counts()
    recommended_categories = recommended_df["Category"].value_counts()

    return {
        "history_categories": history_categories,
        "recommended_categories": recommended_categories,
    }


def most_similar_history_article(
    recommended_news_id: str,
    history_ids: List[str],
    news_id_to_idx: Dict[str, int],
    news_embeddings: np.ndarray,
) -> str:
    """
    For a given recommended article, find which article in the user's
    history it is most similar to (a simple "because you read X" signal).
    """
    if recommended_news_id not in news_id_to_idx:
        return ""

    rec_idx = news_id_to_idx[recommended_news_id]
    rec_vec = news_embeddings[rec_idx]

    hist_idxs = [news_id_to_idx[i] for i in history_ids if i in news_id_to_idx]
    if not hist_idxs:
        return ""

    hist_vecs = news_embeddings[hist_idxs]
    sims = hist_vecs @ rec_vec  # embeddings are pre-normalized -> dot product = cosine sim
    best_idx = hist_idxs[int(np.argmax(sims))]

    idx_to_news_id = {v: k for k, v in news_id_to_idx.items()}
    return idx_to_news_id.get(best_idx, "")


def plot_feature_importance(importance_df, top_n: int = 15, ax=None):
    """
    Plot a horizontal bar chart of LightGBM feature importances
    (expects the DataFrame returned by src.ranker.get_feature_importance).
    """
    import matplotlib.pyplot as plt

    top = importance_df.head(top_n).sort_values("importance")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance (gain)")
    ax.set_title("LightGBM Feature Importance")
    return ax