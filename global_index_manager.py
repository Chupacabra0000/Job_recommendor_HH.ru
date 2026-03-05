# global_index_manager.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from hh_client import fetch_vacancies
from embedding_store import init_store, get_embedding, put_embedding
from db import (
    init_db,
    upsert_global_vacancies,
    set_global_index_state,
    get_global_index_state,
    global_has_vacancy_ids,
    set_global_index_state_if_newer,
)
from global_faiss_index import build_index, save_index, load_index_and_ids

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
        # simple stable hash -> 64-bit
        x = np.uint64(1469598103934665603)  # FNV offset basis
        for b in h:
            x ^= np.uint64(int(b))
            x *= np.uint64(1099511628211)  # FNV prime
        # map to signed int64 range
        return int(np.int64(x.view(np.int64)))


def refresh_global_index(
    cfg: GlobalIndexConfig,
    *,
    force: bool = False,
    min_hours_between_refresh: int = 6,
) -> Tuple[bool, str]:
    """
    Policy A:
      - Incremental updates on each refresh
      - Full rebuild at least every 24h (or when index missing)
    Returns (did_refresh, message)
    """
    init_db()
    init_store()

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

    # Decide whether full rebuild is needed (24h rule) or index missing
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
        # no full rebuild stamp yet -> do one
        need_full = True

    if force:
        need_full = True

    # 1) Fetch latest vacancies for the given area+period (global pool not keyword-limited)
    items = fetch_vacancies(
        text=None,
        area=int(cfg.area_id),
        max_items=int(cfg.max_items),
        per_page=int(cfg.per_page),
        period_days=int(cfg.period_days),
        order_by="publication_time",
    )

    # Build item map once (prevents O(N^2) next(...) scans)
    item_by_id = {str(x.get("id", "")).strip(): x for x in items if x.get("id")}

    # 2) Convert to DB rows
    rows: List[Dict[str, str]] = []
    for it in items:
        vid = str(it.get("id", "")).strip()
        if not vid:
            continue

        # salary text (light)
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

    # 2.5) DB upsert strategy:
    #   - Full rebuild day: upsert all (OK once/day)
    #   - Incremental: upsert only new vacancy_ids (reduces DB writes)
    if need_full:
        upsert_global_vacancies(rows)
    else:
        all_vids = [r["vacancy_id"] for r in rows]
        existing_vids = global_has_vacancy_ids(all_vids)
        new_rows = [r for r in rows if r["vacancy_id"] not in existing_vids]
        if new_rows:
            upsert_global_vacancies(new_rows)

    # 3) Build vectors, embedding only missing ids (cached)
    model = _get_model()
    dim = int(model.get_sentence_embedding_dimension())

    all_vecs = np.zeros((len(rows), dim), dtype=np.float32)
    all_ids = np.zeros((len(rows),), dtype=np.int64)

    missing_texts: List[str] = []
    missing_vids: List[str] = []
    missing_pos: List[int] = []

    for i, r in enumerate(rows):
        vid = str(r["vacancy_id"])
        all_ids[i] = np.int64(_vid_to_int64(vid))

        cached = get_embedding(vid, MODEL_NAME)
        if cached is not None:
            all_vecs[i] = np.asarray(cached, dtype=np.float32)
            continue

        it = item_by_id.get(vid)
        txt = _job_text_from_item(it) if it else f"{r['title']} {r['employer']} {r['snippet_req']} {r['snippet_resp']}"
        missing_vids.append(vid)
        missing_texts.append(txt)
        missing_pos.append(i)

    if missing_texts:
        new_vecs = model.encode(
            missing_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        new_vecs = np.asarray(new_vecs, dtype=np.float32)

        for vid, vec in zip(missing_vids, new_vecs):
            put_embedding(vid, MODEL_NAME, vec)

        for j, pos in enumerate(missing_pos):
            all_vecs[pos] = new_vecs[j]

    all_vecs = _normalize(all_vecs)

    # 4) FAISS strategy:
    #   - Full rebuild every 24h: rebuild from scratch
    #   - Otherwise: incremental add only new ids
    if need_full or idx_existing is None or ids_existing is None:
        index = build_index(all_vecs, all_ids)
        save_index(int(cfg.area_id), int(cfg.period_days), index, all_ids)
        did_update_index = True
        mode = "FULL rebuild"
    else:
        existing_set = set(ids_existing.tolist())
        mask = np.array([int(x) not in existing_set for x in all_ids], dtype=bool)
        if np.any(mask):
            add_vecs = all_vecs[mask].astype(np.float32)
            add_ids = all_ids[mask].astype(np.int64)

            # idx_existing is IndexIDMap2 -> supports add_with_ids
            idx_existing.add_with_ids(add_vecs, add_ids)

            merged_ids = np.concatenate([ids_existing.astype(np.int64), add_ids.astype(np.int64)])
            save_index(int(cfg.area_id), int(cfg.period_days), idx_existing, merged_ids)

            did_update_index = True
        else:
            did_update_index = False

        mode = "INCREMENTAL"

    stamp = _utcnow().isoformat().replace("+00:00", "Z")
    set_global_index_state(key_refresh, stamp)
    if need_full:
        set_global_index_state(key_full, stamp)

    msg = (
        f"Global index refreshed ({mode}): fetched={len(rows)} vacancies, "
        f"dim={dim}, added_to_index={'yes' if did_update_index else 'no'}, at {stamp}."
    )
    return True, msg
