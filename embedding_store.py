import os
import sqlite3
import pickle
from typing import Optional

import numpy as np

DB_PATH = os.getenv("EMBEDDINGS_DB_PATH", os.path.join("artifacts", "embeddings.sqlite3"))

def init_store() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                vacancy_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                emb BLOB NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (vacancy_id, model_name)
            )"""
        )
        conn.commit()

def get_embedding(vacancy_id: str, model_name: str) -> Optional[np.ndarray]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT emb FROM embeddings WHERE vacancy_id=? AND model_name=?",
            (str(vacancy_id), str(model_name)),
        )
        row = cur.fetchone()
        if not row:
            return None
        arr = pickle.loads(row[0])
        return np.asarray(arr, dtype=np.float32)

def put_embedding(vacancy_id: str, model_name: str, emb: np.ndarray) -> None:
    emb = np.asarray(emb, dtype=np.float32)
    blob = pickle.dumps(emb, protocol=pickle.HIGHEST_PROTOCOL)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(vacancy_id, model_name, dim, emb, updated_at) VALUES (?,?,?,?,datetime('now'))",
            (str(vacancy_id), str(model_name), int(emb.shape[0]), blob),
        )
        conn.commit()
