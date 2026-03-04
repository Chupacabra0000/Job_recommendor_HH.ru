import math
import os
import pickle
import re
import html as _html
import hashlib
from typing import Dict, List
from datetime import datetime, timezone, timedelta

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from db import (
    init_db, create_user, authenticate,
    list_resumes, list_favorites, add_favorite, remove_favorite,
    create_session, get_user_by_token, delete_session,
    list_saved_searches, get_latest_saved_search, create_or_get_saved_search,
    upsert_saved_search_results, touch_ranked, touch_refreshed,
    enforce_saved_search_limit, delete_saved_search, list_default_timeline
)

from hh_client import fetch_vacancies, vacancy_details
from sentence_transformers import SentenceTransformer

from tfidf_terms import extract_terms
from embedding_store import init_store, get_embedding, put_embedding
from hh_areas import fetch_areas_tree, list_regions_and_cities
from search_cleanup import enforce_limit_and_cleanup


# ---------- constants ----------
COL_JOB_ID = "Job Id"
COL_WORKPLACE = "workplace"
COL_MODE = "working_mode"
COL_SALARY = "salary"
COL_POSITION = "position"
COL_DUTIES = "job_role_and_duties"
COL_SKILLS = "requisite_skill"
COL_DESC = "offer_details"

DEFAULT_STARTUP_LIMIT = 500
PER_TERM = 50
TERMS_MIN = 6
TERMS_MAX = 10

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_DESC_CHARS = 2500

# ---------- page ----------
st.set_page_config(page_title="HH Job Recommender", page_icon="", layout="wide")
init_db()
init_store()

# ---------- state ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "details_cache" not in st.session_state:
    st.session_state.details_cache = {}
if "terms_text" not in st.session_state:
    st.session_state.terms_text = ""
if "resume_hash_for_terms" not in st.session_state:
    st.session_state.resume_hash_for_terms = ""
if "refresh_nonce" not in st.session_state:
    st.session_state.refresh_nonce = 0
if "last_results_df" not in st.session_state:
    st.session_state.last_results_df = None
if "last_results_meta" not in st.session_state:
    st.session_state.last_results_meta = None
if "last_fetch_at" not in st.session_state:
    st.session_state.last_fetch_at = None
if "page" not in st.session_state:
    st.session_state.page = 1


# ---------- persistent login (URL token) ----------
token = st.query_params.get("token", "")
if st.session_state.user is None and token:
    u = get_user_by_token(token)
    if u:
        st.session_state.user = u


# ---------- helpers ----------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc]).strip()


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _truncate(s: str, n: int = MAX_DESC_CHARS) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_details(vacancy_id: str) -> str:
    full = vacancy_details(vacancy_id)
    return _strip_html(full.get("description") or "")


def _job_text(row: pd.Series) -> str:
    parts = [
        str(row.get(COL_POSITION, "") or ""),
        str(row.get(COL_WORKPLACE, "") or ""),
        str(row.get(COL_MODE, "") or ""),
        str(row.get(COL_SALARY, "") or ""),
        str(row.get(COL_SKILLS, "") or ""),
        str(row.get(COL_DUTIES, "") or ""),
        _truncate(str(row.get(COL_DESC, "") or "")),
    ]
    s = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", s).strip()


def _salary_to_text(sal: dict | None) -> str:
    if not isinstance(sal, dict):
        return ""
    s_from = sal.get("from")
    s_to = sal.get("to")
    cur = sal.get("currency")
    if s_from is not None and s_to is not None:
        return f"{s_from}–{s_to} {cur}"
    if s_from is not None:
        return f"от {s_from} {cur}"
    if s_to is not None:
        return f"до {s_to} {cur}"
    return ""


def _items_to_df(items: List[dict]) -> pd.DataFrame:
    rows = []

    for it in items:
        vid = str(it.get("id", "")).strip()
        if not vid:
            continue
        title = (it.get("name") or "").strip()

        employer = ""
        if it.get("employer"):
            employer = (it["employer"] or {}).get("name", "")

        schedule = it.get("schedule") or {}
        working_mode = schedule.get("name", "") if isinstance(schedule, dict) else ""

        salary_text = _salary_to_text(it.get("salary"))

        snippet = it.get("snippet") or {}
        duties = ""
        skills = ""

        if isinstance(snippet, dict):
            duties = (snippet.get("responsibility") or "").strip()
            skills = (snippet.get("requirement") or "").strip()

        rows.append({
            COL_JOB_ID: vid,
            COL_WORKPLACE: employer,
            COL_MODE: working_mode,
            COL_SALARY: salary_text,
            COL_POSITION: title,
            COL_DUTIES: duties,
            COL_SKILLS: skills,
            COL_DESC: "",
            "alternate_url": it.get("alternate_url", "") or "",
            "published_at": it.get("published_at", "") or ""
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["job_text"] = df.apply(_job_text, axis=1)
    else:
        df["job_text"] = ""
    return df


def _dedupe_merge(list_of_items: List[List[dict]]) -> List[dict]:
    seen = set()
    out = []
    for items in list_of_items:
        for it in items:
            vid = str(it.get("id", "")).strip()
            if not vid or vid in seen:
                continue
            seen.add(vid)
            out.append(it)
    return out


# ---------- GLOBAL PRE-INDEX (area_id + period_days) ----------
# Goal: build a reusable global FAISS index of recent vacancies for the chosen area/timeframe.
# This avoids re-fetching + re-embedding + rebuilding FAISS on every resume search.

GLOBAL_INDEX_DIR = os.path.join("artifacts", "global_index")
GLOBAL_MAX_ITEMS = 5000  # global pool size cap (per area/timeframe)
GLOBAL_TOPK = 1200       # how many candidates to pull from FAISS before term filtering


def _global_dir(area_id: int, period_days: int) -> str:
    return os.path.join(GLOBAL_INDEX_DIR, f"area_{int(area_id)}", f"days_{int(period_days)}")


def _global_meta_path(area_id: int, period_days: int) -> str:
    return os.path.join(_global_dir(area_id, period_days), "meta.pkl")


def _global_faiss_path(area_id: int, period_days: int) -> str:
    return os.path.join(_global_dir(area_id, period_days), "faiss.index")


def _global_is_stale(area_id: int, period_days: int, update_hours: int) -> bool:
    meta_p = _global_meta_path(area_id, period_days)
    idx_p = _global_faiss_path(area_id, period_days)
    if not (os.path.exists(meta_p) and os.path.exists(idx_p)):
        return True
    try:
        mtime = min(os.path.getmtime(meta_p), os.path.getmtime(idx_p))
        age_s = (datetime.now(timezone.utc).timestamp() - mtime)
        return age_s >= (int(update_hours) * 3600)
    except Exception:
        return True


def _item_to_row(it: dict) -> dict:
    vid = str(it.get("id", "")).strip()
    title = (it.get("name") or "").strip()

    emp = it.get("employer") or {}
    employer = emp.get("name", "") if isinstance(emp, dict) else ""

    schedule = it.get("schedule") or {}
    working_mode = schedule.get("name", "") if isinstance(schedule, dict) else ""

    snippet = it.get("snippet") or {}
    duties = (snippet.get("responsibility") or "").strip() if isinstance(snippet, dict) else ""
    skills = (snippet.get("requirement") or "").strip() if isinstance(snippet, dict) else ""

    salary_text = _salary_to_text(it.get("salary"))

    row = {
        COL_JOB_ID: vid,
        COL_WORKPLACE: employer,
        COL_MODE: working_mode,
        COL_SALARY: salary_text,
        COL_POSITION: title,
        COL_DUTIES: duties,
        COL_SKILLS: skills,
        COL_DESC: "",
        "alternate_url": it.get("alternate_url", "") or "",
        "published_at": it.get("published_at", "") or "",
    }
    row["job_text"] = _job_text(pd.Series(row))
    return row


def _build_embeddings_for_df(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    dim = model.get_sentence_embedding_dimension()
    embs = np.zeros((len(df), dim), dtype=np.float32)

    missing_texts = []
    missing_idx = []
    for i, row in df.iterrows():
        vid = str(row[COL_JOB_ID])
        cached = get_embedding(vid, MODEL_NAME)
        if cached is not None:
            embs[i] = cached
        else:
            missing_idx.append(i)
            missing_texts.append(str(row["job_text"] or ""))

    if missing_texts:
        new_emb = model.encode(
            missing_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        new_emb = np.asarray(new_emb, dtype=np.float32)

        for j, i in enumerate(missing_idx):
            embs[i] = new_emb[j]
            put_embedding(str(df.loc[i, COL_JOB_ID]), MODEL_NAME, new_emb[j])

    denom = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / denom


def _build_global_index(area_id: int, period_days: int) -> pd.DataFrame:
    items = fetch_vacancies(
        text=None,
        area=int(area_id),
        max_items=int(GLOBAL_MAX_ITEMS),
        per_page=50,
        period_days=int(period_days),
        order_by="publication_time",
    )

    rows = []
    for it in items:
        vid = str(it.get("id", "")).strip()
        if not vid:
            continue
        rows.append(_item_to_row(it))

    meta_df = pd.DataFrame(rows)
    if meta_df.empty:
        return meta_df

    model = SentenceTransformer(MODEL_NAME)
    job_embs = _build_embeddings_for_df(meta_df, model)

    import faiss  # local import

    d = job_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(job_embs.astype(np.float32))

    out_dir = _global_dir(area_id, period_days)
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, _global_faiss_path(area_id, period_days))

    with open(_global_meta_path(area_id, period_days), "wb") as f:
        pickle.dump(meta_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    return meta_df


def _load_global_meta(area_id: int, period_days: int) -> pd.DataFrame | None:
    p = _global_meta_path(area_id, period_days)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            df = pickle.load(f)
        if isinstance(df, pd.DataFrame):
            return df
        return None
    except Exception:
        return None


def _load_global_faiss(area_id: int, period_days: int):
    p = _global_faiss_path(area_id, period_days)
    if not os.path.exists(p):
        return None
    import faiss
    try:
        return faiss.read_index(p)
    except Exception:
        return None


def _ensure_global_index(area_id: int, period_days: int, update_hours: int) -> pd.DataFrame | None:
    if _global_is_stale(area_id, period_days, update_hours):
        with st.spinner("Обновляем глобальный индекс вакансий (ускорение)..."):
            try:
                return _build_global_index(area_id, period_days)
            except Exception as e:
                st.warning(f"Не удалось обновить глобальный индекс: {e}")
                return _load_global_meta(area_id, period_days)
    return _load_global_meta(area_id, period_days)


def _rank_from_global_index(
    area_id: int,
    period_days: int,
    update_hours: int,
    resume_text: str,
    terms: List[str],
) -> pd.DataFrame | None:
    meta_df = _ensure_global_index(area_id, period_days, update_hours)
    index = _load_global_faiss(area_id, period_days)
    if meta_df is None or index is None or meta_df.empty:
        return None

    model = SentenceTransformer(MODEL_NAME)
    q = model.encode([resume_text], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)

    topk = min(int(GLOBAL_TOPK), len(meta_df))
    scores, idxs = index.search(q, topk)

    cand = meta_df.iloc[idxs[0]].copy().reset_index(drop=True)
    cand["similarity_score"] = scores[0].astype(float)

    # Post-filter by chosen TF-IDF terms (keeps UX semantics).
    tnorm = [t.lower() for t in terms if t.strip()]
    if tnorm:
        def _has_term(s: str) -> bool:
            s = (s or "").lower()
            return any(t in s for t in tnorm)

        mask = cand["job_text"].astype(str).apply(_has_term)
        filtered = cand[mask].copy()
        if len(filtered) >= 50:
            cand = filtered

    cand = cand.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return cand


def _rank_with_faiss(job_embs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    import faiss
    d = job_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(job_embs.astype(np.float32))
    scores, idx = index.search(query_vec.astype(np.float32), job_embs.shape[0])
    score_arr = np.zeros((job_embs.shape[0],), dtype=np.float32)
    score_arr[idx[0]] = scores[0]
    return score_arr


# ---------- auth ----------
def auth_screen():
    st.markdown("## Вход / Регистрация")
    t1, t2 = st.tabs(["Вход", "Регистрация"])
    with t1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти", use_container_width=True):
            user = authenticate(email, password)
            if not user:
                st.error("Неверный email или пароль.")
            else:
                tok = create_session(int(user["id"]), days_valid=30)
                st.query_params["token"] = tok
                st.session_state.user = user
                st.rerun()
    with t2:
        email_r = st.text_input("Email", key="reg_email")
        p1 = st.text_input("Пароль (мин. 6 символов)", type="password", key="reg_password")
        p2 = st.text_input("Повторите пароль", type="password", key="reg_password2")
        if st.button("Создать аккаунт", use_container_width=True):
            if p1 != p2:
                st.error("Пароли не совпадают.")
            else:
                ok, msg = create_user(email_r, p1)
                st.success(msg) if ok else st.error(msg)


if st.session_state.user is None:
    auth_screen()
    st.stop()


# ---------- main ----------
user_id = int(st.session_state.user["id"])

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _areas_cached():
    tree = fetch_areas_tree()
    regions, cities_by_region_id = list_regions_and_cities(tree, country_name="Россия")
    return regions, cities_by_region_id

regions, cities_by_region_id = _areas_cached()

# defaults: Novosibirsk region
default_region_name = "Новосибирская область"
region_names = [r["name"] for r in regions]
default_region_idx = region_names.index(default_region_name) if default_region_name in region_names else 0

st.sidebar.markdown("### Параметры")
region_name = st.sidebar.selectbox("Регион", region_names, index=default_region_idx)
region = next(r for r in regions if r["name"] == region_name)
region_id = int(region["id"])

cities = cities_by_region_id.get(region_id, [])
city_names = [c["name"] for c in cities] if cities else []
default_city = "Новосибирск"
city_idx = city_names.index(default_city) if default_city in city_names else 0
city_name = st.sidebar.selectbox("Город", city_names, index=city_idx) if city_names else None
area_id = int(next(c["id"] for c in cities if c["name"] == city_name)) if city_name else region_id

period_days = st.sidebar.selectbox("Период вакансий (дни)", [14, 30, 60, 90, 180], index=2)
update_hours = st.sidebar.selectbox("Авто-обновление (часы)", [6, 12, 24], index=2)

st.sidebar.markdown("---")

# logout
if st.sidebar.button("Выйти", use_container_width=True):
    tok = st.query_params.get("token", "")
    if tok:
        delete_session(tok)
    st.query_params.clear()
    st.session_state.user = None
    st.rerun()

# ---------- resume ----------
st.sidebar.markdown("### Резюме")
resumes = list_resumes(user_id)
resume_source = st.sidebar.radio("Источник резюме", ["No resume", "Upload PDF", "Created resume"], index=0)

resume_text = ""
has_resume = False
rid = None

if resume_source == "Upload PDF":
    up = st.sidebar.file_uploader("PDF резюме", type=["pdf"])
    if up is not None:
        resume_text = extract_text_from_pdf(up.read())
        has_resume = bool(resume_text.strip())
elif resume_source == "Created resume":
    if resumes:
        labels = [f'{r["id"]}: {r["name"]}' for r in resumes]
        sel = st.sidebar.selectbox("Выберите резюме", labels)
        rid = int(sel.split(":")[0])
        r = next(x for x in resumes if int(x["id"]) == rid)
        resume_text = r["text"]
        has_resume = bool(resume_text.strip())
        st.session_state["selected_resume_label"] = r["name"]
    else:
        st.sidebar.info("Нет сохраненных резюме.")

# TF-IDF terms auto-update when resume changes (no HH calls)
if has_resume:
    rh = hashlib.sha256(resume_text.encode("utf-8", errors="ignore")).hexdigest()
    if rh != st.session_state.resume_hash_for_terms:
        auto_terms = extract_terms(resume_text, top_k=TERMS_MAX)
        auto_terms = auto_terms[:TERMS_MAX]
        if len(auto_terms) < TERMS_MIN:
            auto_terms = list(dict.fromkeys(auto_terms + ["python", "sql"]))[:TERMS_MIN]
        st.session_state.terms_text = "\n".join(auto_terms)
        st.session_state.resume_hash_for_terms = rh

    st.sidebar.markdown("### Термины (можно редактировать)")
    st.session_state.terms_text = st.sidebar.text_area("TF-IDF термины", value=st.session_state.terms_text, height=140)

# manual query button
do_search = st.sidebar.button("Поиск", use_container_width=True)

# ---------- header ----------
st.title("💼 HH.ru Job Recommender")
st.caption("По умолчанию показываем 500 вакансий. Все последующие запросы к HH — только по кнопке **Поиск**.")

favorites = set(list_favorites(user_id))


# ---------- data fetch helpers ----------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_default_startup(area_id: int, period_days: int, nonce: int):
    return fetch_vacancies(
        text=None,
        area=int(area_id),
        max_items=int(DEFAULT_STARTUP_LIMIT),
        per_page=50,
        period_days=int(period_days),
        order_by="publication_time",
    )


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_term(area_id: int, term: str, max_items: int, period_days: int, nonce: int):
    return fetch_vacancies(
        text=term,
        area=int(area_id),
        max_items=int(max_items),
        per_page=50,
        period_days=int(period_days),
        order_by="relevance",
    )


def _render_jobs(df: pd.DataFrame, favorites_set: set, show_scores: bool):
    if df is None or df.empty:
        st.info("Нет вакансий для отображения.")
        return

    page_size = 25
    total = len(df)
    pages = max(1, math.ceil(total / page_size))

    cols = st.columns([1, 2, 1])
    with cols[0]:
        if st.button("⬅️", disabled=(st.session_state.page <= 1)):
            st.session_state.page = max(1, st.session_state.page - 1)
    with cols[1]:
        st.write(f"Страница {st.session_state.page} / {pages} — всего {total}")
    with cols[2]:
        if st.button("➡️", disabled=(st.session_state.page >= pages)):
            st.session_state.page = min(pages, st.session_state.page + 1)

    start = (st.session_state.page - 1) * page_size
    end = min(total, start + page_size)
    subset = df.iloc[start:end]

    for _, row in subset.iterrows():
        vid = str(row[COL_JOB_ID])
        title = str(row.get(COL_POSITION, ""))
        employer = str(row.get(COL_WORKPLACE, ""))
        url = str(row.get("alternate_url", ""))

        left, right = st.columns([6, 1])
        with left:
            st.markdown(f"**{title}** — {employer}")
            if url:
                st.markdown(url)
            if show_scores and "similarity_score" in row and pd.notna(row.get("similarity_score")):
                st.caption(f"Score: {float(row.get('similarity_score')):.4f}")
        with right:
            if vid in favorites_set:
                if st.button("★", key=f"fav_rm_{vid}"):
                    remove_favorite(user_id, vid)
                    st.rerun()
            else:
                if st.button("☆", key=f"fav_add_{vid}"):
                    add_favorite(user_id, vid)
                    st.rerun()

        with st.expander("Описание"):
            if vid not in st.session_state.details_cache:
                st.session_state.details_cache[vid] = _fetch_details(vid)
            st.write(_truncate(st.session_state.details_cache[vid], 2500))


# ---------- DEFAULT VIEW (startup / no click) ----------
if not do_search and st.session_state.last_results_df is None:
    with st.spinner("Загружаем дефолтные вакансии (без эмбеддингов)..."):
        items = _fetch_default_startup(int(area_id), int(period_days), int(st.session_state.refresh_nonce))
        df0 = _items_to_df(items)
    df0["similarity_score"] = pd.NA
    st.session_state.last_results_df = df0
    st.session_state.last_results_meta = {"mode": "default_latest", "area_id": int(area_id)}
    st.session_state.last_fetch_at = datetime.now(timezone.utc)


# ---------- ACTION: manual search ----------
if do_search:
    st.session_state.details_cache = {}
    st.session_state.refresh_nonce += 1

    if not has_resume:
        with st.spinner("Загружаем вакансии по параметрам (без эмбеддингов)..."):
            items = _fetch_default_startup(int(area_id), int(period_days), int(st.session_state.refresh_nonce))
            df = _items_to_df(items)
        df["similarity_score"] = pd.NA
        st.session_state.last_results_df = df
        st.session_state.last_results_meta = {"mode": "explicit_params", "area_id": int(area_id)}
        st.session_state.last_fetch_at = datetime.now(timezone.utc)
        st.session_state.page = 1
    else:
        terms = [t.strip() for t in st.session_state.terms_text.splitlines() if t.strip()]
        terms = terms[:TERMS_MAX]
        if len(terms) < TERMS_MIN:
            terms = list(dict.fromkeys(terms + ["python", "sql"]))[:TERMS_MIN]

        st.write("**Термины для HH:**", ", ".join(terms))

        with st.spinner("Ранжирование через глобальный индекс (FAISS, быстро)..."):
            fast_df = _rank_from_global_index(
                area_id=int(area_id),
                period_days=int(period_days),
                update_hours=int(update_hours),
                resume_text=resume_text,
                terms=terms,
            )

        if fast_df is not None and not fast_df.empty:
            df = fast_df
        else:
            # Fallback: old behavior (multi-query fetch -> embed -> FAISS) if global index is unavailable
            with st.spinner("Fetching вакансий по терминам (multi-query)..."):
                batches = [_fetch_term(int(area_id), term, PER_TERM, int(period_days), int(st.session_state.refresh_nonce)) for term in terms]
                merged = _dedupe_merge(batches)
                df = _items_to_df(merged)

            model = SentenceTransformer(MODEL_NAME)
            with st.spinner("Эмбеддинги вакансий (reuse by vacancy_id) + FAISS ранжирование..."):
                job_embs = _build_embeddings_for_df(df, model)
                q = model.encode([resume_text], normalize_embeddings=True)
                q = np.asarray(q, dtype=np.float32)
                scores = _rank_with_faiss(job_embs, q)
                df["similarity_score"] = scores
                df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

        # ---- SAVE HISTORY (resume-based search) ----
        try:
            if resume_source == "Created resume":
                resume_id_to_save = int(rid) if rid is not None else None
                resume_key = f"rid:{resume_id_to_save}" if resume_id_to_save is not None else "rid:unknown"
                resume_label = st.session_state.get("selected_resume_label")
            else:
                fp = hashlib.sha256((resume_text or "").encode("utf-8", errors="ignore")).hexdigest()
                resume_id_to_save = None
                resume_key = f"pdf:{fp}"
                resume_label = "PDF"

            search_id, created_new = create_or_get_saved_search(
                user_id=user_id,
                resume_key=resume_key,
                area_id=int(area_id),
                timeframe_days=int(period_days),
                resume_id=resume_id_to_save,
                resume_label=resume_label,
                update_interval_hours=int(update_hours),
                refresh_window_hours=int(update_hours),
            )

            enforce_saved_search_limit(user_id=user_id, keep_n=3)

        except Exception as e:
            st.warning(f"Не удалось сохранить историю поиска: {e}")
            search_id = None

        if search_id is not None:
            out_rows = []
            for _r in df.to_dict(orient="records"):
                out_rows.append(
                    {
                        "vacancy_id": str(_r.get(COL_JOB_ID, "")),
                        "published_at": str(_r.get("published_at", "")),
                        "title": str(_r.get(COL_POSITION, "")),
                        "employer": str(_r.get(COL_WORKPLACE, "")),
                        "url": str(_r.get("alternate_url", "")),
                        "snippet_req": str(_r.get(COL_SKILLS, "")),
                        "snippet_resp": str(_r.get(COL_DUTIES, "")),
                        "salary_text": str(_r.get(COL_SALARY, "")),
                        "score": None if pd.isna(_r.get("similarity_score")) else float(_r.get("similarity_score")),
                    }
                )

            try:
                upsert_saved_search_results(search_id, out_rows)
                touch_ranked(search_id)
                touch_refreshed(search_id)
            except Exception as e:
                st.warning(f"Не удалось сохранить результаты: {e}")

        st.session_state.last_results_df = df
        st.session_state.last_results_meta = {"mode": "resume_ranked", "area_id": int(area_id)}
        st.session_state.last_fetch_at = datetime.now(timezone.utc)
        st.session_state.page = 1


# ---------- AUTO-UPDATE ----------
# Only applies to last shown results; if stale, refresh default-latest view.
try:
    if st.session_state.last_fetch_at is not None:
        age = datetime.now(timezone.utc) - st.session_state.last_fetch_at
        if age >= timedelta(hours=int(update_hours)) and st.session_state.last_results_meta is not None:
            meta = st.session_state.last_results_meta
            # auto-refresh only for default / explicit (no resume); for resume-based we expect manual "Поиск"
            if meta.get("mode") in ("default_latest", "explicit_params"):
                with st.spinner("Авто-обновление ленты (последние вакансии)..."):
                    items = _fetch_default_startup(int(area_id), int(period_days), int(st.session_state.refresh_nonce))
                    df0 = _items_to_df(items)
                    df0["similarity_score"] = pd.NA
                    st.session_state.last_results_df = df0
                    st.session_state.last_results_meta = {"mode": meta.get("mode"), "area_id": int(area_id)}
                    st.session_state.last_fetch_at = datetime.now(timezone.utc)
                    st.session_state.page = 1
except Exception:
    pass


# ---------- render ----------
df_show = st.session_state.last_results_df
show_scores = df_show is not None and "similarity_score" in df_show.columns and df_show["similarity_score"].notna().any()
_render_jobs(df_show, favorites, show_scores)

# global cleanup (keeps last 3 searches etc.)
try:
    enforce_limit_and_cleanup(user_id=user_id, keep_n=3)
except Exception:
    pass
