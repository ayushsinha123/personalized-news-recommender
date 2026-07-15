"""
Central configuration for the Personalized News Recommender project.

Every path, hyperparameter, and constant used across the project should be
imported from here rather than hardcoded in individual modules or notebooks.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATASET_DIR = BASE_DIR / "dataset"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

for _dir in (DATA_DIR, OUTPUTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw dataset paths (MIND dataset — gitignored)
# ---------------------------------------------------------------------------
TRAIN_DIR = DATASET_DIR / "train"
DEV_DIR = DATASET_DIR / "dev"

TRAIN_NEWS = TRAIN_DIR / "news.tsv"
TRAIN_BEHAVIORS = TRAIN_DIR / "behaviors.tsv"

DEV_NEWS = DEV_DIR / "news.tsv"
DEV_BEHAVIORS = DEV_DIR / "behaviors.tsv"

NEWS_COLUMNS = [
    "NewsID", "Category", "SubCategory", "Title",
    "Abstract", "URL", "TitleEntities", "AbstractEntities",
]
BEHAVIORS_COLUMNS = ["ImpressionID", "UserID", "Time", "History", "Impressions"]

# ---------------------------------------------------------------------------
# Processed data artifacts
# ---------------------------------------------------------------------------
PROCESSED_NEWS = DATA_DIR / "processed_news.csv"
PROCESSED_BEHAVIORS = DATA_DIR / "processed_behaviors.csv"

DEV_PROCESSED_NEWS = DATA_DIR / "dev_processed_news.csv"
DEV_PROCESSED_BEHAVIORS = DATA_DIR / "dev_processed_behaviors.csv"

# ---------------------------------------------------------------------------
# Embedding artifacts
# ---------------------------------------------------------------------------
NEWS_EMBEDDINGS = DATA_DIR / "news_embeddings.npy"
DEV_NEWS_EMBEDDINGS = DATA_DIR / "dev_news_embeddings.npy"

USER_EMBEDDINGS = DATA_DIR / "user_embeddings.npy"
USER_EMBEDDINGS_IDS = DATA_DIR / "user_embeddings_ids.csv"

NEWS_ID_TO_IDX_PATH = DATA_DIR / "news_id_to_idx.pkl"
DEV_NEWS_ID_TO_IDX_PATH = DATA_DIR / "dev_news_id_to_idx.pkl"

# ---------------------------------------------------------------------------
# FAISS index artifacts
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
DEV_FAISS_INDEX_PATH = DATA_DIR / "dev_faiss.index"

# ---------------------------------------------------------------------------
# Popularity artifacts
# ---------------------------------------------------------------------------
POPULARITY_SCORES = DATA_DIR / "popularity_scores.csv"
DEV_POPULARITY_SCORES = DATA_DIR / "dev_popularity_scores.csv"

# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RECOMMENDATIONS_DIR = OUTPUTS_DIR / "recommendations"

for _dir in (METRICS_DIR, FIGURES_DIR, RECOMMENDATIONS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

METRICS_PATH = METRICS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 128
DEVICE = "cpu"  # set to "cuda" if a GPU is available

# ---------------------------------------------------------------------------
# FAISS configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_TYPE = "IndexFlatIP"  # inner product on normalized vectors = cosine similarity
DEFAULT_TOP_K = 10
CANDIDATE_POOL_SIZE = 100  # candidates retrieved from FAISS before re-ranking

# ---------------------------------------------------------------------------
# Ranking configuration (heuristic baseline — kept for comparison)
# ---------------------------------------------------------------------------
SIMILARITY_WEIGHT = 0.7
POPULARITY_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# Learned ranker configuration
# ---------------------------------------------------------------------------
RANKER_FEATURE_COLUMNS = [
    "similarity_score",
    "popularity_score",
    "category_match",
    "subcategory_match",
    "category_affinity",
    "history_length",
    "max_similarity_to_history",
    "title_length",
    "abstract_length",
]

RANKER_LABEL_COLUMN = "clicked"

# Negative-sampling ratio (negatives : positives) when building the
# training dataset from MIND impressions
NEGATIVE_SAMPLING_RATIO = 4

LOGREG_MODEL_PATH = DATA_DIR / "ranker_logreg.pkl"
LIGHTGBM_MODEL_PATH = DATA_DIR / "ranker_lightgbm.pkl"

TRAIN_VAL_SPLIT_RATIO = 0.85  # fraction of users used for training the ranker

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
EVAL_K = 10
EVAL_SAMPLE_SIZE = 2000

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")