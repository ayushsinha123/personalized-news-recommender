"""
Hybrid ranking: combine content similarity with (log-scaled) popularity,
plus a popularity-based fallback for cold-start users.
"""

from typing import Optional

import faiss
import numpy as np
import pandas as pd

from src import config
from src.utils import logger


def compute_popularity_scores(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count clicks per NewsID from the behaviors log, then log-scale and
    normalize to [0, 1] so a single viral article doesn't dominate.
    """
    if "Impressions" not in behaviors_df.columns:
        raise ValueError("behaviors_df must contain an 'Impressions' column.")

    click_counts = {}
    for impressions in behaviors_df["Impressions"].dropna():
        for entry in impressions.split():
            try:
                news_id, label = entry.rsplit("-", 1)
            except ValueError:
                continue
            if label == "1":
                click_counts[news_id] = click_counts.get(news_id, 0) + 1

    if not click_counts:
        raise ValueError("No clicks found while computing popularity scores.")

    pop_df = pd.DataFrame(list(click_counts.items()), columns=["NewsID", "click_count"])
    pop_df["log_clicks"] = np.log1p(pop_df["click_count"])
    max_log = pop_df["log_clicks"].max()
    pop_df["popularity_score"] = pop_df["log_clicks"] / max_log if max_log > 0 else 0.0

    return pop_df[["NewsID", "popularity_score"]]


def rank_candidates_heuristic(
    candidates_df: pd.DataFrame,
    popularity_df: pd.DataFrame,
    top_k: int = config.DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Baseline ranking: combine similarity and popularity with fixed,
    hand-picked weights. Kept as a comparison point against the trained
    ranker (Logistic Regression / LightGBM) in evaluation.
    """
    if "similarity_score" not in candidates_df.columns:
        raise ValueError("candidates_df must contain a 'similarity_score' column.")

    merged = candidates_df.merge(popularity_df, on="NewsID", how="left")
    merged["popularity_score"] = merged["popularity_score"].fillna(0.0)

    merged["final_score"] = (
        config.SIMILARITY_WEIGHT * merged["similarity_score"]
        + config.POPULARITY_WEIGHT * merged["popularity_score"]
    )

    return merged.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def rank_candidates_ml(
    candidates_df: pd.DataFrame,
    history_ids: list,
    news_df: pd.DataFrame,
    news_id_to_idx: dict,
    news_embeddings: np.ndarray,
    popularity_df: pd.DataFrame,
    model,
    is_lightgbm: bool = True,
    top_k: int = config.DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Score FAISS candidates with a trained click-through ranker
    (Logistic Regression or LightGBM) instead of fixed heuristic weights.
    """
    from src.features import build_features_for_candidates, get_feature_matrix

    feature_df = build_features_for_candidates(
        history_ids, candidates_df, news_df, news_id_to_idx, news_embeddings, popularity_df
    )
    X = get_feature_matrix(feature_df)

    if is_lightgbm:
        scores = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    else:
        scores = model.predict_proba(X)[:, 1]

    merged = candidates_df.merge(feature_df[["NewsID"]], on="NewsID", how="inner").reset_index(drop=True)
    merged["final_score"] = scores

    return merged.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)


def recommend_for_user(
    user_id: str,
    user_vector: Optional[np.ndarray],
    index: faiss.Index,
    news_df: pd.DataFrame,
    popularity_df: pd.DataFrame,
    top_k: int = config.DEFAULT_TOP_K,
    ranker: str = "heuristic",
    model=None,
    history_ids: Optional[list] = None,
    news_id_to_idx: Optional[dict] = None,
    news_embeddings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Cold-start-aware recommendation entry point.

    If user_vector is None (no click history), fall back to a
    popularity-ranked list. Otherwise, retrieve similar candidates via
    FAISS and re-rank them using either:
      - ranker="heuristic": fixed similarity+popularity weights (default, backward compatible)
      - ranker="ml": a trained model (`model`), requires history_ids, news_id_to_idx, news_embeddings
    """
    if user_vector is None:
        logger.debug(f"User {user_id} has no history — using popularity fallback.")
        return (
            news_df.merge(popularity_df, on="NewsID", how="left")
            .fillna({"popularity_score": 0.0})
            .sort_values("popularity_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    from src.recommender import recommend_candidates

    candidates = recommend_candidates(user_vector, index, news_df, top_k=config.CANDIDATE_POOL_SIZE)

    if ranker == "heuristic":
        return rank_candidates_heuristic(candidates, popularity_df, top_k=top_k)

    if ranker == "ml":
        if model is None or history_ids is None or news_id_to_idx is None or news_embeddings is None:
            raise ValueError(
                "ranker='ml' requires model, history_ids, news_id_to_idx, and news_embeddings."
            )
        is_lightgbm = hasattr(model, "best_iteration") or model.__class__.__name__ == "Booster"
        return rank_candidates_ml(
            candidates, history_ids, news_df, news_id_to_idx, news_embeddings,
            popularity_df, model, is_lightgbm=is_lightgbm, top_k=top_k,
        )

    raise ValueError(f"Unknown ranker '{ranker}'. Use 'heuristic' or 'ml'.")