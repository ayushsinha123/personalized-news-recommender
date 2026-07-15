"""
Ranking evaluation metrics: Precision@K, Recall@K, NDCG@K, Hit Rate@K, MAP@K.

All functions take a list of recommended NewsIDs (ranked, best first) and a
set/list of relevant (clicked) NewsIDs.
"""

from typing import List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(recommended_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    if not recommended_ids:
        return 0.0
    recommended_k = recommended_ids[:k]
    hits = len(set(recommended_k) & set(relevant_ids))
    return hits / len(recommended_k)


def recall_at_k(recommended_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    recommended_k = recommended_ids[:k]
    hits = len(set(recommended_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def hit_rate(recommended_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    recommended_k = recommended_ids[:k]
    return 1.0 if set(recommended_k) & set(relevant_ids) else 0.0


def ndcg_at_k(recommended_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    recommended_k = recommended_ids[:k]
    relevant_set = set(relevant_ids)

    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(recommended_k) if item in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    recommended_k = recommended_ids[:k]
    hits, score = 0, 0.0

    for i, item in enumerate(recommended_k):
        if item in relevant_ids:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant_ids), k) if relevant_ids else 0.0


def mean_average_precision(
    list_of_recommended: List[Sequence[str]],
    list_of_relevant: List[Sequence[str]],
    k: int = 10,
) -> float:
    """MAP@K averaged across multiple users."""
    if len(list_of_recommended) != len(list_of_relevant):
        raise ValueError("list_of_recommended and list_of_relevant must be the same length.")

    scores = [
        average_precision_at_k(rec, rel, k=k)
        for rec, rel in zip(list_of_recommended, list_of_relevant)
    ]
    return float(np.mean(scores)) if scores else 0.0


def compute_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """
    ROC-AUC for the click-prediction task itself (label vs. predicted
    click probability for individual (user, candidate) pairs).
    Returns NaN if y_true contains only one class (ROC-AUC undefined).
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_scores))


def compute_pr_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """
    Precision-Recall AUC (average precision) for the click-prediction task.
    More informative than ROC-AUC under heavy class imbalance, which is the
    case for MIND click labels.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_scores))