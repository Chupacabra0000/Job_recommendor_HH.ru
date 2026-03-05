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


@dataclass
class GlobalIndexConfig:
    area_id: int
    period_days: int
    max_items: int = 5000
    per_page: int = 50


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


def refresh_global_index(
    cfg: GlobalIndexConfig,
    *,
    force: bool = False,
    min_hours_between_refresh: int = 6,
) -> Tuple[bool, str]:
    """
    Returns (did_refresh, message)
    """
    init_db()
    init_store()

    key = f"global_index:last_refresh:area={int(cfg.area_id)}:days={int(cfg.period_days)}"
    last = get_global_index_state(key)

    if not force and last:
        try:
            last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
            if _utcnow() - last_dt < timedelta(hours=min_hours_between_refresh):
                return False, f"Global index is fresh (last refresh: {last})."
        except Exception:
            pass

    # 1) Fetch latest vacancies for the given area+period
    items = fetch_vacancies(
        text=None,  # IMPORTANT: global pool should not be keyword-limited
        area=int(cfg.area_id),
        max_items=int(cfg.max_items),
        per_page=int(cfg.per_page),
        period_days=int(cfg.period_days),
        order_by="publication_time",
    )

    # 2) Store metadata into DB (for quick filtering + timeline display)
    rows = []
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

    upsert_global_vacancies(rows)

    # 3) Embed only missing vacancy_ids (cached in SQLite)
    model = SentenceTransformer(MODEL_NAME)
    dim = int(model.get_sentence_embedding_dimension())

    missing_texts: List[str] = []
    missing_vids: List[str] = []

    all_vecs = np.zeros((len(rows), dim), dtype=np.float32)
    all_ids = np.zeros((len(rows),), dtype=np.int64)

    for i, r in enumerate(rows):
        vid = str(r["vacancy_id"])
        all_ids[i] = int(vid)

        cached = get_embedding(vid, MODEL_NAME)
        if cached is not None:
            all_vecs[i] = cached
            continue

        # build text from original items by matching id (fast enough for 5k)
        # fallback to metadata-only if missing
        it = next((x for x in items if str(x.get("id", "")).strip() == vid), None)
        txt = _job_text_from_item(it) if it else f"{r['title']} {r['employer']} {r['snippet_req']} {r['snippet_resp']}"
        missing_vids.append(vid)
        missing_texts.append(txt)

    if missing_texts:
        new_vecs = model.encode(
            missing_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        new_vecs = np.asarray(new_vecs, dtype=np.float32)

        # write to cache
        for vid, vec in zip(missing_vids, new_vecs):
            put_embedding(vid, MODEL_NAME, vec)

        # fill into all_vecs
        miss_map = {int(v): new_vecs[j] for j, v in enumerate(missing_vids)}
        for i in range(len(all_ids)):
            if np.linalg.norm(all_vecs[i]) < 1e-8:
                all_vecs[i] = miss_map.get(int(all_ids[i]), all_vecs[i])

    # normalize (safety)
    all_vecs = _normalize(all_vecs)

    # 4) Rebuild global FAISS index (robust + fast for up to ~20k)
    index = build_index(all_vecs, all_ids)
    save_index(int(cfg.area_id), int(cfg.period_days), index, all_ids)

    stamp = _utcnow().isoformat().replace("+00:00", "Z")
    set_global_index_state(key, stamp)
    return True, f"Global index refreshed: {len(rows)} vacancies, dim={dim}, at {stamp}."
