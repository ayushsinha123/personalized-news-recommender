"""
FAISS index construction, persistence, and similarity search.

Embeddings are normalized (see embedding.py), so an IndexFlatIP (inner
product) index is equivalent to cosine-similarity search.
"""

from pathlib import Path
from typing import Tuple, Union

import faiss
import numpy as np

from src import config
from src.utils import logger


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product FAISS index from an embedding matrix."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")

    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(f"Built FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.Index, path: Union[str, Path] = config.FAISS_INDEX_PATH) -> None:
    """Persist a FAISS index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info(f"Saved FAISS index to {path}")


def load_index(path: Union[str, Path] = config.FAISS_INDEX_PATH) -> faiss.Index:
    """Load a FAISS index from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {path}")
    return faiss.read_index(str(path))


def search(
    index: faiss.Index,
    query_vector: np.ndarray,
    top_k: int = config.DEFAULT_TOP_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the index for the top_k nearest neighbors of a single query vector.

    Returns (scores, indices), each of shape (top_k,).
    """
    query_vector = np.ascontiguousarray(query_vector.astype("float32")).reshape(1, -1)
    scores, idxs = index.search(query_vector, top_k)
    return scores[0], idxs[0]