# global_faiss_index.py
import os
from typing import Optional, Tuple

import numpy as np

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


def _dir(area_id: int, period_days: int) -> str:
    return os.path.join(ARTIFACT_DIR, "global_index", f"area_{int(area_id)}", f"days_{int(period_days)}")


def index_path(area_id: int, period_days: int) -> str:
    return os.path.join(_dir(area_id, period_days), "faiss.index")


def ids_path(area_id: int, period_days: int) -> str:
    return os.path.join(_dir(area_id, period_days), "ids.npy")


def ensure_dir(area_id: int, period_days: int) -> None:
    os.makedirs(_dir(area_id, period_days), exist_ok=True)


def build_index(vectors: np.ndarray, ids: np.ndarray):
    """
    vectors: (N, d) float32, normalized
    ids:     (N,) int64
    """
    import faiss

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    ids = ids.astype(np.int64)

    d = int(vectors.shape[1])
    base = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors, ids)
    return index


def save_index(area_id: int, period_days: int, index, ids: np.ndarray) -> None:
    import faiss

    ensure_dir(area_id, period_days)
    faiss.write_index(index, index_path(area_id, period_days))
    np.save(ids_path(area_id, period_days), ids.astype(np.int64))


def load_index(area_id: int, period_days: int):
    import faiss

    p = index_path(area_id, period_days)
    if not os.path.exists(p):
        return None
    return faiss.read_index(p)


def load_ids(area_id: int, period_days: int) -> Optional[np.ndarray]:
    p = ids_path(area_id, period_days)
    if not os.path.exists(p):
        return None
    return np.load(p)


def search(index, query_vec: np.ndarray, top_k: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    query_vec: (1, d) float32 normalized
    returns: (scores, ids) both shape (top_k,)
    """
    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype(np.float32)

    scores, ids = index.search(query_vec, int(top_k))
    return scores[0], ids[0]
