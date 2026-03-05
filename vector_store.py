import os
import json
import numpy as np
from typing import Optional, Tuple

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


def _model_dir(model_name: str) -> str:
    safe = model_name.replace("/", "_").replace(":", "_")
    return os.path.join(ARTIFACT_DIR, "vector_store", safe)


def _paths(model_name: str) -> Tuple[str, str, str]:
    d = _model_dir(model_name)
    return (
        os.path.join(d, "vecs.f32"),
        os.path.join(d, "ids.npy"),
        os.path.join(d, "meta.json"),
    )


def init_store(model_name: str, dim: int) -> None:
    vec_path, ids_path, meta_path = _paths(model_name)
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)

    if not os.path.exists(meta_path):
        meta = {"dim": int(dim), "count": 0}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # create empty ids file
        np.save(ids_path, np.zeros((0,), dtype=np.int64))

        # create empty vecs file (0 x dim)
        open(vec_path, "wb").close()


def load_meta(model_name: str) -> Optional[dict]:
    _, _, meta_path = _paths(model_name)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ids(model_name: str) -> Optional[np.ndarray]:
    _, ids_path, _ = _paths(model_name)
    if not os.path.exists(ids_path):
        return None
    return np.load(ids_path)


def load_memmap(model_name: str) -> Optional[np.memmap]:
    vec_path, _, meta_path = _paths(model_name)
    if not os.path.exists(meta_path) or not os.path.exists(vec_path):
        return None
    meta = load_meta(model_name)
    dim = int(meta["dim"])
    count = int(meta["count"])
    if count <= 0:
        # still return a valid empty memmap-like array
        return np.memmap(vec_path, dtype=np.float32, mode="r", shape=(0, dim))
    return np.memmap(vec_path, dtype=np.float32, mode="r", shape=(count, dim))


def append_vectors(model_name: str, ids_new: np.ndarray, vecs_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Appends vectors to vecs.f32 and ids.npy. Returns (all_ids, appended_vecs)
    where appended_vecs is the (ids_new, vecs_new) you passed in (normalized float32).
    """
    vec_path, ids_path, meta_path = _paths(model_name)
    meta = load_meta(model_name)
    if meta is None:
        raise RuntimeError("Vector store not initialized. Call init_store first.")

    dim = int(meta["dim"])
    count = int(meta["count"])

    ids_new = np.asarray(ids_new, dtype=np.int64)
    vecs_new = np.asarray(vecs_new, dtype=np.float32)
    if vecs_new.ndim != 2 or vecs_new.shape[1] != dim:
        raise ValueError(f"vecs_new must be (N,{dim}) float32, got {vecs_new.shape}")

    n_add = int(vecs_new.shape[0])
    if n_add == 0:
        all_ids = np.load(ids_path)
        return all_ids, vecs_new

    # append vectors as raw float32 bytes
    with open(vec_path, "ab") as f:
        f.write(vecs_new.tobytes(order="C"))

    # append ids
    all_ids = np.load(ids_path)
    all_ids = np.concatenate([all_ids.astype(np.int64), ids_new.astype(np.int64)])
    np.save(ids_path, all_ids)

    # update meta
    meta["count"] = count + n_add
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return all_ids, vecs_new
