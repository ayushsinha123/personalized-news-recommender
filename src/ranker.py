"""
Learned click-through ranker: builds a labeled (user, candidate) training
dataset from MIND impressions, trains a Logistic Regression baseline and a
LightGBM model, evaluates them with ROC-AUC / PR-AUC, and persists them.

Train/validation split is done BY USER, not by row, so that a user's
impressions never appear in both splits — avoiding leakage that would
inflate validation AUC.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import config
from src.features import build_feature_row, get_feature_matrix
from src.user_profile import get_user_vector
from src.utils import logger

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


def _parse_impressions(impressions_str: str):
    """Split 'N123-1 N456-0' into [(NewsID, label), ...]."""
    pairs = []
    for entry in impressions_str.split():
        try:
            news_id, label = entry.rsplit("-", 1)
            pairs.append((news_id, int(label)))
        except ValueError:
            continue
    return pairs


def build_training_dataset(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
    news_id_to_idx: Dict[str, int],
    news_embeddings: np.ndarray,
    popularity_df: pd.DataFrame,
    negative_ratio: int = config.NEGATIVE_SAMPLING_RATIO,
    random_state: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build a labeled feature table for ranker training.

    For every impression, all clicked (positive) candidates are kept, and
    up to `negative_ratio` non-clicked (negative) candidates per positive
    are randomly sampled — MIND impressions are already a form of implicit
    negative sampling, this just keeps class balance under control.
    """
    rng = np.random.default_rng(random_state)

    news_lookup = news_df.set_index("NewsID")
    pop_map = dict(zip(popularity_df["NewsID"], popularity_df["popularity_score"]))

    rows = []

    for _, row in behaviors_df.iterrows():
        history_str = row["History"] if isinstance(row["History"], str) else ""
        history_ids = history_str.split() if history_str else []
        hist_idxs = [news_id_to_idx[i] for i in history_ids if i in news_id_to_idx]

        if not hist_idxs:
            continue  # ranker trains only on warm-start impressions; cold-start uses the popularity fallback

        history_rows = news_df[news_df["NewsID"].isin(history_ids)]
        history_categories = history_rows["Category"]
        history_subcategories = history_rows["SubCategory"]
        history_length = len(hist_idxs)

        user_vector = get_user_vector(history_str, news_id_to_idx, news_embeddings)
        if user_vector is None:
            continue

        pairs = _parse_impressions(row["Impressions"])
        pos_ids = [nid for nid, label in pairs if label == 1 and nid in news_id_to_idx]
        neg_ids = [nid for nid, label in pairs if label == 0 and nid in news_id_to_idx]

        if not pos_ids:
            continue

        n_neg = min(len(neg_ids), len(pos_ids) * negative_ratio)
        sampled_neg_ids = list(rng.choice(neg_ids, size=n_neg, replace=False)) if n_neg > 0 else []

        for news_id, label in [(i, 1) for i in pos_ids] + [(i, 0) for i in sampled_neg_ids]:
            if news_id not in news_lookup.index:
                continue
            cand = news_lookup.loc[news_id]
            cand_idx = news_id_to_idx[news_id]

            similarity_score = float(user_vector @ news_embeddings[cand_idx])
            max_sim = float(np.max(news_embeddings[hist_idxs] @ news_embeddings[cand_idx]))

            feat = build_feature_row(
                similarity_score=similarity_score,
                popularity_score=pop_map.get(news_id, 0.0),
                candidate_category=cand.get("Category", ""),
                candidate_subcategory=cand.get("SubCategory", ""),
                history_categories=history_categories,
                history_subcategories=history_subcategories,
                history_length=history_length,
                max_similarity_to_history=max_sim,
                title_length=len(str(cand.get("Title", ""))),
                abstract_length=len(str(cand.get("Abstract", ""))),
            )
            feat["UserID"] = row["UserID"]
            feat["NewsID"] = news_id
            feat[config.RANKER_LABEL_COLUMN] = label
            rows.append(feat)

    dataset = pd.DataFrame(rows)
    logger.info(
        f"Built training dataset: {len(dataset)} rows "
        f"({dataset[config.RANKER_LABEL_COLUMN].mean():.1%} positive)"
    )
    return dataset


def user_wise_train_val_split(
    dataset: pd.DataFrame,
    val_size: float = 1 - config.TRAIN_VAL_SPLIT_RATIO,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by UserID (not by row) so that a single user's
    impressions never appear in both train and validation — avoids leakage
    that would artificially inflate validation AUC.
    """
    user_ids = dataset["UserID"].unique()
    train_users, val_users = train_test_split(
        user_ids, test_size=val_size, random_state=random_state
    )
    train_df = dataset[dataset["UserID"].isin(train_users)].reset_index(drop=True)
    val_df = dataset[dataset["UserID"].isin(val_users)].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} rows / {len(train_users)} users")
    logger.info(f"Val:   {len(val_df)} rows / {len(val_users)} users")
    return train_df, val_df


def train_logistic_regression(
    train_df: pd.DataFrame,
    feature_columns=None,
) -> LogisticRegression:
    """Train a Logistic Regression baseline click-through model."""
    X = get_feature_matrix(train_df, feature_columns)
    y = train_df[config.RANKER_LABEL_COLUMN].to_numpy()

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns=None,
    params: dict = None,
):
    """Train a LightGBM click-through ranker with early stopping on the val set."""
    if not _HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    feature_columns = feature_columns or config.RANKER_FEATURE_COLUMNS
    X_train = get_feature_matrix(train_df, feature_columns)
    y_train = train_df[config.RANKER_LABEL_COLUMN].to_numpy()
    X_val = get_feature_matrix(val_df, feature_columns)
    y_val = val_df[config.RANKER_LABEL_COLUMN].to_numpy()

    default_params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        is_unbalance=True,
        verbosity=-1,
        seed=config.RANDOM_SEED,
    )
    params = {**default_params, **(params or {})}

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_columns, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)],
    )
    return model


def evaluate_model(model, df: pd.DataFrame, feature_columns=None, is_lightgbm: bool = False) -> Dict[str, float]:
    """Compute ROC-AUC and PR-AUC for a trained model on a labeled dataset."""
    X = get_feature_matrix(df, feature_columns)
    y = df[config.RANKER_LABEL_COLUMN].to_numpy()

    if is_lightgbm:
        y_scores = model.predict(X, num_iteration=model.best_iteration)
    else:
        y_scores = model.predict_proba(X)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y, y_scores)),
        "pr_auc": float(average_precision_score(y, y_scores)),
    }


def get_feature_importance(model, feature_columns=None) -> pd.DataFrame:
    """Return a sorted feature-importance table for a trained LightGBM model."""
    feature_columns = feature_columns or config.RANKER_FEATURE_COLUMNS
    importances = model.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model(model, path: Union[str, Path], is_lightgbm: bool = False) -> None:
    """Persist a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_lightgbm:
        model.save_model(str(path))
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)

    logger.info(f"Saved model to {path}")


def load_model(path: Union[str, Path], is_lightgbm: bool = False):
    """Load a trained model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if is_lightgbm:
        return lgb.Booster(model_file=str(path))
    with open(path, "rb") as f:
        return pickle.load(f)