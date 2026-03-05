# global_index_manager.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from hh_client import fetch_vacancies
from db import (
    init_db,
    upsert_global_vacancies,
    set_global_index_state,
    get_global_index_state,
    global_has_vacancy_ids,
)
from global_faiss_index import build_index, save_index, load_index_and_ids
from vector_store import (
    init_store as init_vec_store,
    load_ids as load_vec_ids,
    load_memmap as load_vec_memmap,
    append_vectors,
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Policy A: Always incremental + full rebuild every 24h
FULL_REBUILD_EVERY_HOURS = 24


@dataclass
class GlobalIndexConfig:
    area_id: int
    period_days: int
    max_items: int = 5000
    per_page: int = 50


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


def _vid_to_int64(vid: str) -> int:
    """
    HH vacancy ids are numeric strings, but we keep a safe fallback.
    Returns a stable signed int64.
    """
    s = str(vid).strip()
    try:
        return int(s)
    except Exception:
        h = np.frombuffer(s.encode("utf-8", errors="ignore"), dtype=np.uint8)
        x = np.uint64(1469598103934665603)  # FNV offset basis
        for b in h:
            x ^= np.uint64(int(b))
            x *= np.uint64(1099511628211)  # FNV prime
        return int(np.int64(x.view(np.int64)))


def _job_text_from_item(it: Dict) -> str:
    title = (it.get("name") or "").strip()

    employer = ""
    emp = it.get("employer") or {}
    if isinstance(emp, dict):
        employer = (emp.get("name") or "").strip()

    snippet = it.get("snippet") or {}
    req = ""
    resp = ""
    if isinstance(snippet, dict):
        req = (snippet.get("requirement") or "").strip()
        resp = (snippet.get("responsibility") or "").strip()

    schedule = it.get("schedule") or {}
    mode = schedule.get("name", "") if isinstance(schedule, dict) else ""

    return " ".join([p for p in [title, employer, mode, req, resp] if p]).strip()


def _build_item_map(items: List[Dict]) -> Dict[str, Dict]:
    return {str(x.get("id", "")).strip(): x for x in items if x.get("id")}


def _ensure_vectors_for_ids(
    model: SentenceTransformer,
    ids_needed: np.ndarray,
    id_to_text: Dict[int, str],
    *,
    dim: int,
) -> None:
    """
    Ensures every id in ids_needed exists in vector_store (memmap).
    Encodes only missing vectors and appends them.
    """
    init_vec_store(MODEL_NAME, dim)

    vec_ids = load_vec_ids(MODEL_NAME)
    vec_id_set = set(vec_ids.tolist()) if vec_ids is not None else set()

    missing = [int(x) for x in ids_needed.tolist() if int(x) not in vec_id_set]
    if not missing:
        return

    # Keep deterministic order
    missing_ids = np.asarray(missing, dtype=np.int64)
    missing_texts: List[str] = []
    for i64 in missing_ids.tolist():
        txt = id_to_text.get(int(i64), "")
        missing_texts.append(txt or "")

    new_vecs = model.encode(
        missing_texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    new_vecs = _normalize(np.asarray(new_vecs, dtype=np.float32))

    append_vectors(MODEL_NAME, missing_ids, new_vecs)


def _load_vectors_for_ids(ids_needed: np.ndarray) -> np.ndarray:
    """
    Loads vectors for the given int64 ids from vector_store memmap.
    Assumes vectors exist for all ids.
    """
    vec_ids = load_vec_ids(MODEL_NAME)
    mm = load_vec_memmap(MODEL_NAME)
    if vec_ids is None or mm is None:
        raise RuntimeError("Vector store is not initialized or empty.")

    # Build id->position mapping (fast enough for tens/hundreds of thousands)
    id_to_pos = {int(v): i for i, v in enumerate(vec_ids.tolist())}

    # Map requested ids to positions
    pos = np.empty((len(ids_needed),), dtype=np.int64)
    for i, v in enumerate(ids_needed.tolist()):
        p = id_to_pos.get(int(v))
        if p is None:
            raise RuntimeError(f"Vector missing in store for id={int(v)}")
        pos[i] = int(p)

    # Fancy index reads into a normal ndarray (fast, contiguous)
    vecs = np.asarray(mm[pos], dtype=np.float32)
    return vecs


def refresh_global_index(
    cfg: GlobalIndexConfig,
    *,
    force: bool = False,
    min_hours_between_refresh: int = 6,
) -> Tuple[bool, str]:
    """
    Policy A:
      - Incremental updates normally (DB+FAISS)
      - Full rebuild at least every 24h (or when index missing)
    Returns (did_refresh, message)
    """
    init_db()

    key_refresh = f"global_index:last_refresh:area={int(cfg.area_id)}:days={int(cfg.period_days)}"
    key_full = f"global_index:last_full_rebuild:area={int(cfg.area_id)}:days={int(cfg.period_days)}"

    last_refresh = get_global_index_state(key_refresh)
    last_full = get_global_index_state(key_full)

    # Throttle frequent refresh
    if not force and last_refresh:
        try:
            last_dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00"))
            if _utcnow() - last_dt < timedelta(hours=min_hours_between_refresh):
                return False, f"Global index is fresh (last refresh: {last_refresh})."
        except Exception:
            pass

    idx_existing, ids_existing = load_index_and_ids(int(cfg.area_id), int(cfg.period_days))
    need_full = False

    if idx_existing is None or ids_existing is None:
        need_full = True

    if not need_full and not force and last_full:
        try:
            last_full_dt = datetime.fromisoformat(last_full.replace("Z", "+00:00"))
            if _utcnow() - last_full_dt >= timedelta(hours=FULL_REBUILD_EVERY_HOURS):
                need_full = True
        except Exception:
            need_full = True
    elif not need_full and not force and not last_full:
        need_full = True

    if force:
        need_full = True

    # 1) Fetch global pool (not keyword-limited)
    items = fetch_vacancies(
        text=None,
        area=int(cfg.area_id),
        max_items=int(cfg.max_items),
        per_page=int(cfg.per_page),
        period_days=int(cfg.period_days),
        order_by="publication_time",
    )
    item_by_id = _build_item_map(items)

    # 2) Build DB rows (global_vacancies)
    rows: List[Dict] = []
    for it in items:
        vid = str(it.get("id", "")).strip()
        if not vid:
            continue

        salary_text = ""
        sal = it.get("salary")
        if isinstance(sal, dict):
            s_from = sal.get("from")
            s_to = sal.get("to")
            cur = sal.get("currency")
            if s_from is not None and s_to is not None:
                salary_text = f"{s_from}–{s_to} {cur}"
            elif s_from is not None:
                salary_text = f"от {s_from} {cur}"
            elif s_to is not None:
                salary_text = f"до {s_to} {cur}"

        snippet = it.get("snippet") or {}
        req = (snippet.get("requirement") or "").strip() if isinstance(snippet, dict) else ""
        resp = (snippet.get("responsibility") or "").strip() if isinstance(snippet, dict) else ""

        emp = it.get("employer") or {}
        employer = (emp.get("name") or "") if isinstance(emp, dict) else ""

        rows.append(
            dict(
                vacancy_id=vid,
                area_id=int(cfg.area_id),
                published_at=it.get("published_at", "") or "",
                title=(it.get("name") or "")[:400],
                employer=(employer or "")[:300],
                url=(it.get("alternate_url") or "")[:700],
                snippet_req=req[:800],
                snippet_resp=resp[:800],
                salary_text=(salary_text or "")[:120],
            )
        )

    # 2.5) Upsert DB rows
    if need_full:
        upsert_global_vacancies(rows)
    else:
        all_vids = [r["vacancy_id"] for r in rows]
        existing_vids = global_has_vacancy_ids(all_vids)
        new_rows = [r for r in rows if r["vacancy_id"] not in existing_vids]
        if new_rows:
            upsert_global_vacancies(new_rows)

    # 3) Ensure vectors exist in vector_store for this pool (encode only missing)
    model = _get_model()
    dim = int(model.get_sentence_embedding_dimension())
    init_vec_store(MODEL_NAME, dim)

    pool_ids = np.asarray([_vid_to_int64(r["vacancy_id"]) for r in rows], dtype=np.int64)

    # Build texts for pool ids (for missing encoding only)
    id_to_text: Dict[int, str] = {}
    for r in rows:
        vid_s = str(r["vacancy_id"])
        vid_i = _vid_to_int64(vid_s)
        it = item_by_id.get(vid_s)
        id_to_text[int(vid_i)] = _job_text_from_item(it) if it else f"{r['title']} {r['employer']} {r['snippet_req']} {r['snippet_resp']}"

    _ensure_vectors_for_ids(model, pool_ids, id_to_text, dim=dim)

    # 4) Load vectors from memmap for current pool
    pool_vecs = _load_vectors_for_ids(pool_ids)
    pool_vecs = _normalize(pool_vecs)

    # 5) FAISS update:
    #    - Full rebuild every 24h: rebuild EXACTLY for the current pool
    #    - Otherwise: incremental add only new ids (vectors pulled from memmap)
    if need_full or idx_existing is None or ids_existing is None:
        index = build_index(pool_vecs, pool_ids)
        save_index(int(cfg.area_id), int(cfg.period_days), index, pool_ids)
        mode = "FULL rebuild"
        did_update = True
    else:
        existing_set = set(ids_existing.tolist())
        mask = np.array([int(x) not in existing_set for x in pool_ids], dtype=bool)
        if np.any(mask):
            add_ids = pool_ids[mask].astype(np.int64)
            add_vecs = pool_vecs[mask].astype(np.float32)

            idx_existing.add_with_ids(add_vecs, add_ids)
            merged_ids = np.concatenate([ids_existing.astype(np.int64), add_ids.astype(np.int64)])
            save_index(int(cfg.area_id), int(cfg.period_days), idx_existing, merged_ids)

            did_update = True
        else:
            did_update = False
        mode = "INCREMENTAL"

    stamp = _utcnow().isoformat().replace("+00:00", "Z")
    set_global_index_state(key_refresh, stamp)
    if need_full:
        set_global_index_state(key_full, stamp)

    msg = (
        f"Global index refreshed ({mode}): fetched={len(rows)}, dim={dim}, "
        f"index_updated={'yes' if did_update else 'no'}, at {stamp}."
    )
    return True, msg
