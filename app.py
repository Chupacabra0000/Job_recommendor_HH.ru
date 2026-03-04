# app.py
import re
import html as _html
import hashlib
from typing import Dict, List
from datetime import datetime, timezone

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from db import (
    init_db,
    create_user,
    authenticate,
    list_resumes,
    list_favorites,
    add_favorite,
    remove_favorite,
    create_session,
    get_user_by_token,
    delete_session,
    create_or_get_saved_search,
    upsert_saved_search_results,
    touch_ranked,
    touch_refreshed,
    enforce_saved_search_limit,
    list_default_timeline,
    get_global_vacancy,
)
from hh_client import fetch_vacancies, vacancy_details
from tfidf_terms import extract_terms
from embedding_store import init_store, get_embedding, put_embedding
from hh_areas import fetch_areas_tree, list_regions_and_cities
from faiss_search_index import delete_index_dir
from search_cleanup import enforce_limit_and_cleanup

from global_index_manager import refresh_global_index, GlobalIndexConfig
from global_faiss_index import load_index as load_global_index, search as global_search


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
CACHE_TTL_SECONDS = 60 * 60  # 60 min

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
                COL_MODE: "",
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
    df["job_text"] = df.apply(_job_text, axis=1)
    df["is_favorite"] = df[COL_JOB_ID].apply(lambda x: 1 if str(x) in favorites_set else 0)

    df = df.sort_values(
        ["is_favorite", "similarity_score", "published_at"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return df


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _areas_cached():
    tree = fetch_areas_tree()
    regions, cities_by_region_id = list_regions_and_cities(tree, country_name="Россия")
    return regions, cities_by_region_id


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_details(vacancy_id: str) -> str:
    full = vacancy_details(vacancy_id)
    return _strip_html(full.get("description") or "")


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


def _rank_with_faiss_local(embs: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    import faiss

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))
    scores, idx = index.search(query_vec.astype(np.float32), embs.shape[0])
    score_arr = np.zeros((embs.shape[0],), dtype=np.float32)
    score_arr[idx[0]] = scores[0]
    return score_arr


def _resume_key_from_pdf(pdf_bytes: bytes) -> str:
    return "pdf:" + hashlib.sha256(pdf_bytes).hexdigest()


# ---------- auth UI ----------
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


# ---------- main UI ----------
user_id = int(st.session_state.user["id"])
favorites = set(list_favorites(user_id))

st.sidebar.markdown("### Параметры")
regions, cities_by_region_id = _areas_cached()

# defaults: Novosibirsk region (as you requested earlier)
default_region_name = "Новосибирская область"
region_names = [r["name"] for r in regions]
default_region_idx = region_names.index(default_region_name) if default_region_name in region_names else 0
region_name = st.sidebar.selectbox("Регион", region_names, index=default_region_idx)

region = next(r for r in regions if r["name"] == region_name)
region_id = int(region["id"])

cities = cities_by_region_id.get(region_id, [])
city_names = [c["name"] for c in cities] if cities else []
default_city = "Новосибирск"
city_idx = city_names.index(default_city) if default_city in city_names else 0
city_name = st.sidebar.selectbox("Город", city_names, index=city_idx) if city_names else None
city_id = int(next(c["id"] for c in cities if c["name"] == city_name)) if city_name else region_id

period_days = st.sidebar.selectbox("Период вакансий (дни)", [14, 30, 60, 90, 180], index=2)
auto_update_hours = st.sidebar.selectbox("Авто-обновление (часы)", [6, 12, 24], index=2)

st.sidebar.markdown("---")

if st.sidebar.button("Выйти", use_container_width=True):
    tok = st.query_params.get("token", "")
    if tok:
        delete_session(tok)
    st.query_params.clear()
    st.session_state.user = None
    st.rerun()

# Optional: allow refreshing global index
with st.sidebar.expander("⚡ Глобальный индекс (ускорение)"):
    st.caption("Если индекс уже построен, поиск по резюме работает намного быстрее.")
    if st.button("Обновить глобальный индекс сейчас", use_container_width=True):
        did, msg = refresh_global_index(
            GlobalIndexConfig(area_id=city_id, period_days=int(period_days), max_items=5000),
            force=True,
            min_hours_between_refresh=int(auto_update_hours),
        )
        st.success(msg) if did else st.info(msg)

# ---------- Resume selection ----------
st.markdown("# HH Job Recommender")

resumes = list_resumes(user_id)
resume_mode = st.radio("Источник резюме", ["Загрузить PDF", "Выбрать сохраненное"], horizontal=True)

resume_text = ""
resume_label = None
resume_key = None
resume_id = None

if resume_mode == "Загрузить PDF":
    up = st.file_uploader("Загрузите резюме (PDF)", type=["pdf"])
    if up:
        pdf_bytes = up.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
        resume_key = _resume_key_from_pdf(pdf_bytes)
        resume_label = up.name
else:
    if resumes:
        opt = st.selectbox("Выберите резюме", [f'{r["id"]}: {r["name"]}' for r in resumes])
        resume_id = int(opt.split(":")[0])
        r = next(x for x in resumes if int(x["id"]) == resume_id)
        resume_text = r["text"]
        resume_key = f"rid:{resume_id}"
        resume_label = r["name"]
    else:
        st.info("У вас нет сохраненных резюме. Загрузите PDF или создайте резюме.")

# ---------- Default timeline ----------
st.markdown("## Лента (сохраненные результаты)")
timeline_df = _load_default_timeline_from_history(user_id, favorites)
if timeline_df.empty:
    st.info("История пуста. Выполните поиск по резюме, чтобы сохранить результаты.")
else:
    for i, row in timeline_df.head(50).iterrows():
        vid = str(row[COL_JOB_ID])
        cols = st.columns([6, 1])
        with cols[0]:
            st.markdown(f"**{row[COL_POSITION]}** — {row[COL_WORKPLACE]}")
            if row.get("alternate_url"):
                st.markdown(row["alternate_url"])
            if row.get("similarity_score") is not None:
                st.caption(f"Score: {float(row['similarity_score']):.4f}")
        with cols[1]:
            if vid in favorites:
                if st.button("★", key=f"fav_rm_{vid}"):
                    remove_favorite(user_id, vid)
                    st.rerun()
            else:
                if st.button("☆", key=f"fav_add_{vid}"):
                    add_favorite(user_id, vid)
                    st.rerun()

        # lazy full description
        with st.expander("Описание"):
            if vid not in st.session_state.details_cache:
                st.session_state.details_cache[vid] = _fetch_details(vid)
            st.write(_truncate(st.session_state.details_cache[vid], 2500))

st.markdown("---")
st.markdown("## Поиск по резюме")

if not resume_text.strip():
    st.info("Выберите или загрузите резюме, затем нажмите Поиск.")
    st.stop()

# TF-IDF terms (editable)
res_hash = hashlib.sha256(resume_text.encode("utf-8")).hexdigest()
if st.session_state.resume_hash_for_terms != res_hash:
    terms = extract_terms(resume_text, top_k=TERMS_MAX)  # <-- compatible with current tfidf_terms.py
    terms = [t.strip() for t in terms if str(t).strip()]

# enforce min/max without changing tfidf_terms.py
if len(terms) < TERMS_MIN:
    # pad with generic tech terms (won't break; user can edit anyway)
    fallback = ["python", "sql", "аналитик", "разработчик", "backend", "data"]
    for t in fallback:
        if t not in terms:
            terms.append(t)
        if len(terms) >= TERMS_MIN:
            break

terms = terms[:TERMS_MAX]

st.session_state.terms_text = "\n".join(terms)
st.session_state.resume_hash_for_terms = res_hash

st.caption("Ключевые термины (можно редактировать перед поиском):")
terms_text = st.text_area("Термины", value=st.session_state.terms_text, height=120)
terms = [t.strip() for t in terms_text.splitlines() if t.strip()]

do_search = st.button("Поиск", type="primary")

if not do_search:
    st.stop()

# Create/get saved search (history entry)
search_id, _created = create_or_get_saved_search(
    user_id=user_id,
    resume_key=resume_key,
    resume_id=resume_id,
    resume_label=resume_label,
    area_id=city_id,
    timeframe_days=int(period_days),
    update_interval_hours=int(auto_update_hours),
    refresh_window_hours=int(auto_update_hours),
)

# Try fast path: global index
model = SentenceTransformer(MODEL_NAME)
q = model.encode([resume_text], normalize_embeddings=True)
q = np.asarray(q, dtype=np.float32)

global_index = load_global_index(city_id, int(period_days))

ranked_rows: List[Dict] = []

if global_index is not None:
    # Fast: search global FAISS, then load metadata from global_vacancies table
    scores, ids = global_search(global_index, q, top_k=500)
    for score, vid_int in zip(scores, ids):
        if int(vid_int) == -1:
            continue
        vid = str(int(vid_int))
        meta = get_global_vacancy(vid)
        if not meta:
            continue
        ranked_rows.append(
            dict(
                vacancy_id=vid,
                published_at=meta.get("published_at"),
                title=meta.get("title"),
                employer=meta.get("employer"),
                url=meta.get("url"),
                snippet_req=meta.get("snippet_req"),
                snippet_resp=meta.get("snippet_resp"),
                salary_text=meta.get("salary_text"),
                score=float(score),
            )
        )

    # Save + cleanup
    upsert_saved_search_results(search_id, ranked_rows)
    touch_ranked(search_id)
    touch_refreshed(search_id)
    deleted = enforce_saved_search_limit(user_id, keep_last=3)
    for sid in deleted:
        delete_index_dir(sid)

    st.success(f"Готово (глобальный индекс): {len(ranked_rows)} вакансий.")
else:
    # Fallback: your previous behavior (multi-term fetch -> embed -> local rank)
    st.warning("Глобальный индекс не найден. Использую обычный режим (медленнее).")

    term_lists = []
    for term in terms[:TERMS_MAX]:
        term_lists.append(
            fetch_vacancies(
                text=term,
                area=city_id,
                max_items=PER_TERM,
                per_page=PER_TERM,
                period_days=int(period_days),
                order_by="publication_time",
            )
        )
    merged = _dedupe_merge(term_lists)
    df = _items_to_df(merged)

    embs = _build_embeddings_for_df(df, model)
    scores = _rank_with_faiss_local(embs, q)

    df["similarity_score"] = scores
    df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    ranked_rows = []
    for _, row in df.head(300).iterrows():
        ranked_rows.append(
            dict(
                vacancy_id=str(row[COL_JOB_ID]),
                published_at=row.get("published_at"),
                title=row.get(COL_POSITION),
                employer=row.get(COL_WORKPLACE),
                url=row.get("alternate_url"),
                snippet_req=row.get(COL_SKILLS),
                snippet_resp=row.get(COL_DUTIES),
                salary_text=row.get(COL_SALARY),
                score=float(row.get("similarity_score") or 0.0),
            )
        )

    upsert_saved_search_results(search_id, ranked_rows)
    touch_ranked(search_id)
    touch_refreshed(search_id)
    deleted = enforce_saved_search_limit(user_id, keep_last=3)
    for sid in deleted:
        delete_index_dir(sid)

    st.success(f"Готово (обычный режим): {len(ranked_rows)} вакансий.")

# Render results (top 50)
st.markdown("### Результаты поиска")
out_df = pd.DataFrame(
    [
        {
            "vacancy_id": r["vacancy_id"],
            "title": r["title"],
            "employer": r["employer"],
            "score": r["score"],
            "url": r["url"],
        }
        for r in ranked_rows
    ]
)

for _, r in out_df.head(50).iterrows():
    vid = str(r["vacancy_id"])
    cols = st.columns([6, 1])
    with cols[0]:
        st.markdown(f"**{r['title']}** — {r['employer']}")
        if r.get("url"):
            st.markdown(r["url"])
        st.caption(f"Score: {float(r['score']):.4f}")
    with cols[1]:
        if vid in favorites:
            if st.button("★", key=f"fav_rm_res_{vid}"):
                remove_favorite(user_id, vid)
                st.rerun()
        else:
            if st.button("☆", key=f"fav_add_res_{vid}"):
                add_favorite(user_id, vid)
                st.rerun()

    with st.expander("Описание"):
        if vid not in st.session_state.details_cache:
            st.session_state.details_cache[vid] = _fetch_details(vid)
        st.write(_truncate(st.session_state.details_cache[vid], 2500))

# keep disk storage small
enforce_limit_and_cleanup(user_id=user_id)
