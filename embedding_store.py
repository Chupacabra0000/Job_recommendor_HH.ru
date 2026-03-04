# embedding_store.py
from __future__ import annotations

from db import init_db, get_embedding as _get, put_embedding as _put


def init_store() -> None:
    init_db()


def get_embedding(vacancy_id: str, model_name: str):
    return _get(vacancy_id, model_name)


def put_embedding(vacancy_id: str, model_name: str, vec) -> None:
    _put(vacancy_id, model_name, vec)
