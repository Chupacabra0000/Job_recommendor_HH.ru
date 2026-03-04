# embedding_store.py
from __future__ import annotations

import numpy as np

from db import init_db, get_embedding_db as _get, put_embedding_db as _put


def init_store() -> None:
    # ensure tables exist
    init_db()


def get_embedding(vacancy_id: str, model_name: str) -> np.ndarray | None:
    blob = _get(str(vacancy_id), str(model_name))
    if blob is None:
        return None
    # stored as raw float32 bytes
    return np.frombuffer(blob, dtype=np.float32)


def put_embedding(vacancy_id: str, model_name: str, emb: np.ndarray) -> None:
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    _put(str(vacancy_id), str(model_name), int(emb.shape[0]), emb.tobytes())
