
import os
import shutil
from typing import Dict, List, Tuple, Optional

import numpy as np

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
MODEL_DIM_DEFAULT = 384

def _search_dir(search_id: int) -> str:
    return os.path.join(ARTIFACT_DIR, "saved_searches", str(int(search_id)))

def index_path(search_id: int) -> str:
    return os.path.join(_search_dir(search_id), "faiss.index")

def ensure_dir(search_id: int) -> None:
    os.makedirs(_search_dir(search_id), exist_ok=True)

def delete_index_dir(search_id: int) -> None:
    d = _search_dir(search_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)

def build_index(vectors: np.ndarray, ids: np.ndarray):
    """
    vectors: (N,d) float32, normalized
    ids: (N,) int64
    """
    import faiss
    d = int(vectors.shape[1])
    base = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors.astype(np.float32), ids.astype(np.int64))
    return index

def save_index(search_id: int, index) -> None:
    import faiss
    ensure_dir(search_id)
    faiss.write_index(index, index_path(search_id))

def load_index(search_id: int):
    import faiss
    p = index_path(search_id)
    if not os.path.exists(p):
        return None
    return faiss.read_index(p)
