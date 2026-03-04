import math
import re
import html as _html
import hashlib
import os
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
from faiss_search_index import delete_index_dir

# ---------- constants ----------
COL_JOB_ID = "Job Id"
COL_WORKPLACE = "workplace"
COL_MODE = "working_mode"
COL_SALARY = "salary"
COL_POSITION = "position"
COL_DUTIES = "job_role_and_duties"
COL_SKILLS = "requisite_skill"
COL_DESC = "offer_details"

DEFAULT_QUERY = "Python"
DEFAULT_STARTUP_LIMIT = 500

PER_TERM = 50
TERMS_MIN = 6
TERMS_MAX = 10

CACHE_TTL_SECONDS = 60 * 60  # 60 min
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_DESC_CHARS = 2500

# ---------- page ----------
st.set_page_config(page_title="HH Job Recommender", page_icon="💼", layout="wide")
init_db()
init_store()

# ---------- styles ----------
st.markdown(
    """
    <style>
      .center-wrap { max-width: 520px; margin: 0 auto; }
      .card { border: 1px solid rgba(49,51,63,.15); border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; }
      .pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
              border: 1px solid rgba(49,51,63,.2); margin-right: 6px; margin-top: 6px; font-size: 0.85rem; }
      .pill-strong { font-weight: 600; }
      .snippet { color: rgba(49,51,63,.82); font-size: 0.95rem; margin-top: 8px; }
      .skill-chip { display: inline-block; margin: 4px 6px 0 0; padding: 2px 8px; border-radius: 999px;
                    background: rgba(49,51,63,.06); font-size: 0.85rem; }
      .muted { color: rgba(49,51,63,.65); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- state ----------
if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = 1
    st.session_state.last_fetch_at = datetime.now(timezone.utc)
if "page_size" not in st.session_state:
    st.session_state.page_size = 20

if "resume_source" not in st.session_state:
    st.session_state.resume_source = "None"
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "details_cache" not in st.session_state:
    st.session_state.details_cache = {}  # vacancy_id -> full_desc_text

# terms state
if "terms_text" not in st.session_state:
    st.session_state.terms_text = ""
if "resume_hash_for_terms" not in st.session_state:
    st.session_state.resume_hash_for_terms = ""

# manual fetch results state
if "last_fetch_at" not in st.session_state:
    st.session_state.last_fetch_at = None
if "refresh_nonce" not in st.session_state:
    st.session_state.refresh_nonce = 0

if "last_results_df" not in st.session_state:
    st.session_state.last_results_df = None
if "last_results_meta" not in st.session_state:
    st.session_state.last_results_meta = {}

# bootstrap default vacancies once per session
if "did_bootstrap_default" not in st.session_state:
    st.session_state.did_bootstrap_default = False

# ---------- persistent login (URL token) ----------
# If user refreshes the page, Streamlit starts a new session.
# We restore user from token in query params.
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


def _items_to_df(items: List[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        vid = str(it.get("id", "")).strip()
        if not vid:
            continue
        title = (it.get("name") or "").strip()

        employer = ""
        emp = it.get("employer") or {}
        if isinstance(emp, dict):
            employer = emp.get("name", "") or ""

        schedule = it.get("schedule") or {}
        working_mode = schedule.get("name", "") if isinstance(schedule, dict) else ""

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
        duties = ""
        skills = ""
        if isinstance(snippet, dict):
            duties = (snippet.get("responsibility") or "").strip()
            skills = (snippet.get("requirement") or "").strip()

        rows.append(
            {
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
        )

    df = pd.DataFrame(rows)
    if len(df):
        df["job_text"] = df.apply(_job_text, axis=1)
    else:
        df["job_text"] = ""
    return df




def _load_default_timeline_from_history(user_id: int, favorites_set: set) -> pd.DataFrame:
    # Default timeline = ALL saved vacancies across last 2-3 saved searches,
    # merged/deduped, using MOST RECENT search score per vacancy (see db.list_default_timeline).
    rows = list_default_timeline(user_id=user_id, limit=5000)
    if not rows:
        return pd.DataFrame()

    all_rows = []
    for r in rows:
        vid = str(r.get("vacancy_id", "")).strip()
        if not vid:
            continue
        all_rows.append(
            {
                COL_JOB_ID: vid,
                COL_WORKPLACE: r.get("employer", "") or "",
                COL_MODE: "",  # not stored in saved results
                COL_SALARY: r.get("salary_text", "") or "",
                COL_POSITION: r.get("title", "") or "",
                COL_DUTIES: r.get("snippet_resp", "") or "",
                COL_SKILLS: r.get("snippet_req", "") or "",
                COL_DESC: "",
                "alternate_url": r.get("url", "") or "",
                "published_at": r.get("published_at", "") or "",
                "similarity_score": r.get("score", None),
            }
        )

    df = pd.DataFrame(all_rows)
    if len(df):
        df["job_text"] = df.apply(_job_text, axis=1)
        df["is_favorite"] = df[COL_JOB_ID].apply(lambda x: 1 if str(x) in favorites_set else 0)
        # favorites pinned first in DEFAULT view, then by similarity (desc)
        # Use published_at as a stable tie-breaker.
        df = df.sort_values(
            ["is_favorite", "similarity_score", "published_at"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return df


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)  # 24h
def _areas_cached():
    tree = fetch_areas_tree()
    regions, cities_by_region_id = list_regions_and_cities(tree, country_name="Россия")
    return regions, cities_by_region_id

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_default_startup(area_id: int, period_days: int, refresh_nonce: int) -> List[dict]:
    # refresh_nonce is used only to bust Streamlit cache when needed
    return fetch_vacancies(
        text=DEFAULT_QUERY,
        area=area_id,
        max_items=DEFAULT_STARTUP_LIMIT,
        per_page=50,
        period_days=period_days,
        order_by="publication_time",
    )
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_term(area_id: int, term: str, per_term: int, period_days: int, refresh_nonce: int) -> List[dict]:
    # refresh_nonce is used only to bust Streamlit cache when needed
    return fetch_vacancies(
        text=term,
        area=area_id,
        max_items=per_term,
        per_page=per_term,
        period_days=period_days,
        order_by="publication_time",
    )
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_details(vacancy_id: str) -> str:
    full = vacancy_details(vacancy_id)
    return _strip_html(full.get("description") or "")

# ---------- GLOBAL PRE-INDEX (area + period_days) ----------
# Stores:
#   artifacts/global_index/area_<id>/days_<n>/meta.pkl  (DataFrame with job_text + basic fields)
#   artifacts/global_index/area_<id>/days_<n>/faiss.index
# This makes resume searches much faster by reusing a global pool of recent vacancies.

GLOBAL_MAX_ITEMS = 5000  # global pool size per (area_id, period_days)
GLOBAL_TOPK = 2000       # candidates retrieved from FAISS before optional term-filtering
GLOBAL_RESULTS_KEEP = 300  # results saved to history

def _global_dir(area_id: int, period_days: int) -> str:
    return os.path.join("artifacts", "global_index", f"area_{int(area_id)}", f"days_{int(period_days)}")

def _global_meta_path(area_id: int, period_days: int) -> str:
    return os.path.join(_global_dir(area_id, period_days), "meta.pkl")

def _global_faiss_path(area_id: int, period_days: int) -> str:
    return os.path.join(_global_dir(area_id, period_days), "faiss.index")

def _global_is_fresh(area_id: int, period_days: int, refresh_hours: int) -> bool:
    mp = _global_meta_path(area_id, period_days)
    ip = _global_faiss_path(area_id, period_days)
    if not (os.path.exists(mp) and os.path.exists(ip)):
        return False
    try:
        mtime = max(os.path.getmtime(mp), os.path.getmtime(ip))
        age_sec = max(0.0, (datetime.now(timezone.utc).timestamp() - mtime))
        return age_sec < float(refresh_hours) * 3600.0
    except Exception:
        return False

def _global_load(area_id: int, period_days: int):
    import faiss  # local import
    mp = _global_meta_path(area_id, period_days)
    ip = _global_faiss_path(area_id, period_days)
    if not (os.path.exists(mp) and os.path.exists(ip)):
        return None, None
    try:
        df = pd.read_pickle(mp)
        index = faiss.read_index(ip)
        return df, index
    except Exception:
        return None, None

def _global_build(area_id: int, period_days: int, refresh_nonce: int, model: SentenceTransformer):
    """
    Build / rebuild global pool + FAISS index for (area_id, period_days).
    Uses embeddings cache by vacancy_id to avoid recomputing.
    """
    import faiss  # local import
    os.makedirs(_global_dir(area_id, period_days), exist_ok=True)

    # Fetch WITHOUT keyword limit: latest vacancies for timeframe/area.
    # (hh_client.fetch_vacancies supports text=None and will omit 'text' param.)
    items = fetch_vacancies(
        text=None,
        area=int(area_id),
        max_items=int(GLOBAL_MAX_ITEMS),
        per_page=50,
        period_days=int(period_days),
        order_by="publication_time",
    )
    df = _items_to_df(items)
    if df is None or df.empty:
        return df, None

    # Build embeddings (cached) and FAISS index
    job_embs = _build_embeddings_for_df(df, model)
    d = job_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(job_embs.astype(np.float32))

    # Persist
    df.to_pickle(_global_meta_path(area_id, period_days))
    faiss.write_index(index, _global_faiss_path(area_id, period_days))
    return df, index

def _global_get_or_build(area_id: int, period_days: int, refresh_hours: int, refresh_nonce: int, model: SentenceTransformer):
    """
    Load global index if fresh; otherwise rebuild.
    Returns (df, faiss_index) or (None, None) if unable.
    """
    if _global_is_fresh(area_id, period_days, refresh_hours):
        df, index = _global_load(area_id, period_days)
        if df is not None and index is not None and len(df):
            return df, index
    # rebuild
    try:
        return _global_build(area_id, period_days, refresh_nonce, model)
    except Exception:
        return None, None

def _term_match_mask(df: pd.DataFrame, terms: List[str]) -> pd.Series:
    """
    Lightweight filter: keep rows whose job_text contains ANY term (case-insensitive).
    Applied after FAISS retrieval (post-filter).
    """
    if df is None or df.empty or not terms:
        return pd.Series([True] * (len(df) if df is not None else 0))
    tset = [t.strip().lower() for t in terms if t.strip()]
    if not tset:
        return pd.Series([True] * len(df))
    jt = df.get("job_text", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
    mask = False
    for t in tset:
        mask = mask | jt.str.contains(re.escape(t), na=False)
    return mask


) or "")

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
            missing_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        new_emb = np.asarray(new_emb, dtype=np.float32)
        for j, i in enumerate(missing_idx):
            embs[i] = new_emb[j]
            put_embedding(str(df.loc[i, COL_JOB_ID]), MODEL_NAME, new_emb[j])

    denom = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / denom

def _rank_with_faiss(embs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    import faiss
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))
    scores, idx = index.search(query_vec.astype(np.float32), embs.shape[0])
    score_arr = np.zeros((embs.shape[0],), dtype=np.float32)
    score_arr[idx[0]] = scores[0]
    return score_arr

def _snippet(s: str, n: int = 230) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"

def _chips(skills_text: str, limit: int = 10) -> List[str]:
    if not skills_text:
        return []
    parts = re.split(r"[;,]\s*|\n+", skills_text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
        if len(out) >= limit:
            break
    return out

# ---------- auth UI (centered) ----------
def auth_screen():
    st.markdown("<div class='center-wrap'>", unsafe_allow_html=True)
    st.markdown("## 🔐 Вход / Регистрация")
    t1, t2 = st.tabs(["Вход", "Регистрация"])

    with t1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти", use_container_width=True):
            user = authenticate(email, password)
            if not user:
                st.error("Неверный email или пароль.")
            else:
                # Create persistent session token and store in URL
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

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.user is None:
    auth_screen()
    st.stop()

user_id = int(st.session_state.user["id"])

# ---------- sidebar ----------
st.sidebar.title("⚙️ Настройки")

# Logout: delete session token and clear query param
if st.sidebar.button("🚪 Выйти", use_container_width=True):
    tok = st.query_params.get("token", "")
    if tok:
        delete_session(tok)
    st.query_params.clear()
    st.session_state.user = None
    st.session_state.last_results_df = None
    st.session_state.did_bootstrap_default = False
    st.rerun()

st.sidebar.subheader("Локация (HH Areas)")
regions, cities_by_region_id = _areas_cached()

region_names = [r["name"] for r in regions]
_default_region_idx = 0
if "Новосибирская область" in region_names:
    _default_region_idx = region_names.index("Новосибирская область")
region_name = st.sidebar.selectbox("Регион", region_names, index=_default_region_idx if region_names else 0)
region_obj = next((r for r in regions if r["name"] == region_name), None)
region_id = region_obj["id"] if region_obj else None

cities = cities_by_region_id.get(str(region_id), []) if region_id else []
city_names = [c["name"] for c in cities]
default_city_idx = 0
if "Новосибирск" in city_names:
    default_city_idx = city_names.index("Новосибирск")
city_name = st.sidebar.selectbox("Город", city_names, index=default_city_idx if city_names else 0)
city_obj = next((c for c in cities if c["name"] == city_name), None)
area_id = int(city_obj["id"]) if city_obj else 1

st.sidebar.subheader("Время")
period_days = st.sidebar.selectbox("Период вакансий (дней)", [30, 60, 90, 180], index=1)
update_hours = st.sidebar.selectbox("Авто-обновление (часы)", [6, 12, 24], index=2)

st.sidebar.subheader("Резюме")
resume_source = st.sidebar.radio("Источник резюме", ["None", "PDF resume", "Created resume"], index=0)
st.session_state.resume_source = resume_source

if resume_source == "PDF resume":
    pdf = st.sidebar.file_uploader("Загрузите PDF", type=["pdf"])
    if pdf is not None:
        with st.spinner("Читаем PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(pdf.read())

resumes = list_resumes(user_id)
selected_resume_text = ""
if resume_source == "Created resume" and resumes:
    opts = {f'{r["name"]} (#{r["id"]})': r["id"] for r in resumes}
    label = st.sidebar.selectbox("Выберите резюме", list(opts.keys()))
    st.session_state.selected_resume_label = label
    rid = opts[label]
    sel = next((x for x in resumes if x["id"] == rid), None)
    selected_resume_text = sel["text"] if sel else ""

resume_text = ""
if resume_source == "PDF resume":
    resume_text = st.session_state.pdf_text
elif resume_source == "Created resume":
    resume_text = selected_resume_text
has_resume = bool((resume_text or "").strip())

st.sidebar.subheader("Термины (TF-IDF)")
# auto-generate TF-IDF terms when resume changes (no HH calls)
if has_resume:
    rh = hashlib.sha256(resume_text.encode("utf-8", errors="ignore")).hexdigest()
    if rh != st.session_state.resume_hash_for_terms:
        auto_terms = extract_terms(resume_text, top_k=TERMS_MAX)
        auto_terms = auto_terms[:TERMS_MAX]
        if len(auto_terms) < TERMS_MIN:
            auto_terms = list(dict.fromkeys(auto_terms + ["python", "sql"]))[:TERMS_MIN]
        st.session_state.terms_text = "\n".join(auto_terms)
        st.session_state.resume_hash_for_terms = rh

    st.sidebar.caption("Термины выбраны автоматически. Отредактируйте и нажмите **Поиск**.")
    st.session_state.terms_text = st.sidebar.text_area(
        "Термины (по одному в строке)",
        value=st.session_state.terms_text,
        height=160,
    )

    add_term = st.sidebar.text_input("Добавить термин", value="")
    if st.sidebar.button("➕ Добавить", use_container_width=True):
        t = add_term.strip()
        if t:
            current = [x.strip() for x in st.session_state.terms_text.splitlines() if x.strip()]
            current.append(t)
            seen = set()
            deduped = []
            for x in current:
                k = x.lower()
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(x)
            st.session_state.terms_text = "\n".join(deduped[:TERMS_MAX])
            st.rerun()
else:
    st.sidebar.info("Загрузите/выберите резюме — термины TF-IDF появятся автоматически.")

st.sidebar.subheader("Показ")
page_size = st.sidebar.selectbox("На странице", [10, 20, 50, 100], index=1)
st.session_state.page_size = int(page_size)

# ✅ Manual trigger for HH fetch (after bootstrap)
do_search = st.sidebar.button("Поиск", use_container_width=True)

# ---------- header ----------
st.title("💼 HH.ru Job Recommender")
st.caption("По умолчанию показываем 500 вакансий. Все последующие запросы к HH — только по кнопке **Поиск**.")

favorites = set(list_favorites(user_id))

# ---------- ACTION: manual search ----------
if do_search:
    # clear lazy details on new search
    st.session_state.details_cache = {}
    st.session_state.refresh_nonce += 1

    if not has_resume:
        with st.spinner("Загружаем дефолтные вакансии (без эмбеддингов)..."):
            items = _fetch_default_startup(int(area_id), int(period_days), int(st.session_state.refresh_nonce))
            df = _items_to_df(items)
        df["similarity_score"] = pd.NA
        st.session_state.last_results_df = df
        st.session_state.last_results_meta = {"mode": "explicit_params", "area_id": int(area_id)}
    else:
        terms = [t.strip() for t in st.session_state.terms_text.splitlines() if t.strip()]
        terms = terms[:TERMS_MAX]
        if len(terms) < TERMS_MIN:
            terms = list(dict.fromkeys(terms + ["python", "sql"]))[:TERMS_MIN]

        st.write("**Термины для HH:**", ", ".join(terms))

        model = SentenceTransformer(MODEL_NAME)

        # ---- FAST PATH: Global pre-index (area_id + period_days) ----
        # We load (or rebuild) a global pool of recent vacancies once per refresh window,
        # then rank instantly via FAISS.
        with st.spinner("Подготавливаем глобальный индекс (ускорение)..."):
            g_df, g_index = _global_get_or_build(
                int(area_id),
                int(period_days),
                refresh_hours=int(update_hours),
                refresh_nonce=int(st.session_state.refresh_nonce),
                model=model,
            )

        if g_df is not None and g_index is not None and len(g_df):
            with st.spinner("Ранжирование по глобальному индексу (FAISS)..."):
                q = model.encode([resume_text], normalize_embeddings=True)
                q = np.asarray(q, dtype=np.float32)

                topk = min(int(GLOBAL_TOPK), int(len(g_df)))
                scores, idxs = g_index.search(q.astype(np.float32), topk)

                cand = g_df.iloc[idxs[0]].copy()
                cand["similarity_score"] = scores[0]

                # Optional post-filter by TF-IDF terms (keeps UX: terms still matter)
                mask = _term_match_mask(cand, terms)
                filtered = cand[mask].copy()
                if len(filtered) >= 50:
                    df = filtered
                else:
                    df = cand

                df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
        else:
            # ---- FALLBACK: multi-query HH fetch (previous behavior) ----
            with st.spinner("Fetching вакансий по терминам (multi-query)..."):
                batches = [
                    _fetch_term(int(area_id), term, PER_TERM, int(period_days), int(st.session_state.refresh_nonce))
                    for term in terms
                ]
                merged = _dedupe_merge(batches)
                df = _items_to_df(merged)

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
                resume_id_to_save = int(rid)  # type: ignore[name-defined]
                resume_key = f"rid:{resume_id_to_save}"
                resume_label = st.session_state.get("selected_resume_label")
            else:
                # PDF resume: save by fingerprint of extracted text
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

            # keep only last 3 searches per user to limit storage
            enforce_saved_search_limit(user_id=user_id, keep_n=3)
        
        except Exception as e:
            st.warning(f"Не удалось сохранить историю поиска: {e}")
            search_id = None
            created_new = False

        # save results with scores (if history was saved)
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
            upsert_saved_search_results(search_id, out_rows)
            touch_ranked(search_id)
            touch_refreshed(search_id)

        top_n = min(10, len(df))
        with st.spinner("Подгружаем полные описания только для TOP-10..."):
            for i in range(top_n):
                vid = str(df.loc[i, COL_JOB_ID])
                if vid and vid not in st.session_state.details_cache:
                    try:
                        st.session_state.details_cache[vid] = _fetch_details(vid)
                    except Exception:
                        st.session_state.details_cache[vid] = ""

        st.session_state.last_results_df = df
        st.session_state.last_results_meta = {"mode": "ranked_manual", "area_id": int(area_id), "terms": terms}

    st.session_state.page = 1
    st.session_state.last_fetch_at = datetime.now(timezone.utc)


# ---------- DEFAULT VIEW (search history) ----------
# Default vacancies = union of vacancies from saved resume-based searches (history).
# If user has no saved searches yet, ask them to run a resume-based search first.
if st.session_state.last_results_df is None:
    hist_df = _load_default_timeline_from_history(user_id=user_id, favorites_set=favorites)
    if hist_df is None or hist_df.empty:
        st.info("Пока нет истории поисков с резюме. Выберите резюме и нажмите **Поиск**, чтобы сохранить результаты.")
        st.stop()
    st.session_state.last_results_df = hist_df
    st.session_state.last_results_meta = {"mode": "default_history"}
    st.session_state.page = 1
    st.session_state.last_fetch_at = datetime.now(timezone.utc)


# ---------- AUTO_REFRESH ----------
try:
    _now = datetime.now(timezone.utc)
    _last = st.session_state.last_fetch_at
    if _last is not None and (_now - _last) >= timedelta(hours=int(update_hours)):
        st.session_state.refresh_nonce += 1
        st.session_state.details_cache = {}
        if st.session_state.last_results_meta.get("mode", "") == "default_history":
            hist_df = _load_default_timeline_from_history(user_id=user_id, favorites_set=favorites)
            if hist_df is not None and not hist_df.empty:
                st.session_state.last_results_df = hist_df
        elif st.session_state.last_results_meta.get("mode", "").startswith("ranked") and has_resume:
            # Prefer global pre-index during auto-refresh (much faster, fewer HH calls).
            model = SentenceTransformer(MODEL_NAME)
            g_df, g_index = _global_get_or_build(
                int(area_id),
                int(period_days),
                refresh_hours=int(update_hours),
                refresh_nonce=int(st.session_state.refresh_nonce),
                model=model,
            )
            if g_df is not None and g_index is not None and len(g_df):
                q = model.encode([resume_text], normalize_embeddings=True)
                q = np.asarray(q, dtype=np.float32)
                topk = min(int(GLOBAL_TOPK), int(len(g_df)))
                scores, idxs = g_index.search(q.astype(np.float32), topk)
                cand = g_df.iloc[idxs[0]].copy()
                cand["similarity_score"] = scores[0]

                terms = [t.strip() for t in st.session_state.terms_text.splitlines() if t.strip()]
                terms = terms[:TERMS_MAX]
                if len(terms) < TERMS_MIN:
                    terms = list(dict.fromkeys(terms + ["python", "sql"]))[:TERMS_MIN]

                mask = _term_match_mask(cand, terms)
                filtered = cand[mask].copy()
                st.session_state.last_results_df = (
                    filtered if len(filtered) >= 50 else cand
                ).sort_values("similarity_score", ascending=False).reset_index(drop=True)
            else:
                # fallback to previous multi-query refresh if global index missing
                terms = [t.strip() for t in st.session_state.terms_text.splitlines() if t.strip()]
                terms = terms[:TERMS_MAX]
                if len(terms) < TERMS_MIN:
                    terms = list(dict.fromkeys(terms + ["python", "sql"]))[:TERMS_MIN]
                batches = [_fetch_term(int(area_id), term, PER_TERM, int(period_days), int(st.session_state.refresh_nonce)) for term in terms]
                merged = _dedupe_merge(batches)
                _df = _items_to_df(merged)

                job_embs = _build_embeddings_for_df(_df, model)
                q = model.encode([resume_text], normalize_embeddings=True)
                q = np.asarray(q, dtype=np.float32)
                scores = _rank_with_faiss(job_embs, q)
                _df["similarity_score"] = scores
                _df = _df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
                st.session_state.last_results_df = _df
        else:
            items = _fetch_default_startup(int(area_id), int(period_days), int(st.session_state.refresh_nonce))
            _df = _items_to_df(items)
            _df["similarity_score"] = pd.NA
            st.session_state.last_results_df = _df
        st.session_state.last_fetch_at = _now
except Exception:
    pass

df = st.session_state.last_results_df.copy()

# ---------- render ----------
def render_job(row: Dict, idx: int):
    vid = str(row.get(COL_JOB_ID, "") or "")
    title = str(row.get(COL_POSITION, "") or "Untitled")
    company = str(row.get(COL_WORKPLACE, "") or "")
    mode_ = str(row.get(COL_MODE, "") or "")
    sal = str(row.get(COL_SALARY, "") or "")
    url = str(row.get("alternate_url", "") or "")
    pub = str(row.get("published_at", "") or "")
    score = row.get("similarity_score", None)

    pct = None
    if score is not None and pd.notna(score):
        pct = max(0, min(100, int(round(float(score) * 100))))

    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([4, 1.2, 1.2])
    with c1:
        st.markdown(f"### {idx}. {title}", unsafe_allow_html=True)
        pills = []
        if company:
            pills.append(f'<span class="pill pill-strong">{company}</span>')
        if mode_:
            pills.append(f'<span class="pill">{mode_}</span>')
        if sal:
            pills.append(f'<span class="pill">{sal}</span>')
        if pub:
            pills.append(f'<span class="pill">{pub[:10]}</span>')
        if url:
            pills.append(f'<span class="pill"><a href="{url}" target="_blank">hh.ru</a></span>')
        if pills:
            st.markdown(" ".join(pills), unsafe_allow_html=True)

        st.markdown(f"<div class='snippet'>{_snippet(row.get('job_text',''))}</div>", unsafe_allow_html=True)

        chips = _chips(str(row.get(COL_SKILLS, "") or ""), limit=10)
        if chips:
            st.markdown("".join([f"<span class='skill-chip'>{c}</span>" for c in chips]), unsafe_allow_html=True)

    with c2:
        if pct is not None:
            st.metric("Сходство", f"{pct}%")
            st.progress(pct)
        else:
            st.metric("Сходство", "—")

    with c3:
        if vid:
            is_fav = vid in favorites
            label = "⭐ Удалить" if is_fav else "☆ В избранное"
            if st.button(label, use_container_width=True, key=f"fav_{vid}_{idx}"):
                if is_fav:
                    remove_favorite(user_id, vid)
                else:
                    add_favorite(user_id, vid)
                st.rerun()

    with st.expander("Подробнее"):
        full_desc = ""
        if vid and vid in st.session_state.details_cache:
            full_desc = st.session_state.details_cache[vid]
        elif vid and st.session_state.last_results_meta.get("mode", "").startswith("ranked") and idx > 10:
            with st.spinner("Подгружаем полное описание..."):
                try:
                    full_desc = _fetch_details(vid)
                except Exception:
                    full_desc = ""
                st.session_state.details_cache[vid] = full_desc

        if full_desc:
            st.write(full_desc)
        else:
            st.caption("Полное описание не загружено (или отсутствует).")

    st.markdown("</div>", unsafe_allow_html=True)

total = len(df)
st.caption(f"Вакансий: {total}")

total_pages = max(1, math.ceil(total / st.session_state.page_size))
st.session_state.page = max(1, min(st.session_state.page, total_pages))

start = (st.session_state.page - 1) * st.session_state.page_size
end = start + st.session_state.page_size
page_df = df.iloc[start:end]

for i, row in enumerate(page_df.to_dict(orient="records"), start=1):
    render_job(row, start + i)

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    if st.button("⬅️", disabled=st.session_state.page <= 1):
        st.session_state.page -= 1
        st.rerun()
with c2:
    st.write(f"Страница {st.session_state.page} / {total_pages}")
with c3:
    if st.button("➡️", disabled=st.session_state.page >= total_pages):
        st.session_state.page += 1
        st.rerun()
