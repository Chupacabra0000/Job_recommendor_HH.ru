import math
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
    enforce_saved_search_limit, delete_saved_search, list_default_timeline,
    get_global_vacancies_by_ids
)

from hh_client import fetch_vacancies, vacancy_details
from sentence_transformers import SentenceTransformer

from tfidf_terms import extract_terms
from embedding_store import init_store, get_embedding, put_embedding
from hh_areas import fetch_areas_tree, list_regions_and_cities
from search_cleanup import enforce_limit_and_cleanup
from faiss_search_index import delete_index_dir
from global_index_manager import GlobalIndexConfig, refresh_global_index
from global_faiss_index import load_index as load_global_index, search as search_global_index

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
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TERMS_MIN = 6
TERMS_MAX = 18

SAVED_SEARCH_MAX = 3
DEFAULT_UPDATE_INTERVAL_HOURS = 24
DEFAULT_REFRESH_WINDOW_HOURS = 24

DETAILS_CACHE_MAX = 600

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

# ---------- helpers ----------
def _strip_html(s: str) -> str:
    # very light sanitize
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _truncate(s: str, n: int = 1200) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _snippet(job_text: str) -> str:
    job_text = job_text or ""
    job_text = job_text.strip()
    if not job_text:
        return ""
    # show first 240 chars as snippet
    return _html.escape(_truncate(job_text, 240))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _read_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(text_parts).strip()


def _dedupe_merge(batches: List[List[dict]]) -> List[dict]:
    seen = set()
    out = []
    for b in batches:
        for it in b:
            vid = str(it.get("id", "")).strip()
            if not vid or vid in seen:
                continue
            seen.add(vid)
            out.append(it)
    return out


def _job_text(row: pd.Series) -> str:
    parts = [
        str(row.get(COL_POSITION, "") or ""),
        str(row.get(COL_WORKPLACE, "") or ""),
        str(row.get(COL_MODE, "") or ""),
        str(row.get(COL_DUTIES, "") or ""),
        str(row.get(COL_SKILLS, "") or ""),
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


def _build_embeddings_for_df(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    embs = []
    for _, r in df.iterrows():
        vid = str(r.get(COL_JOB_ID, "")).strip()
        if not vid:
            embs.append(np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32))
            continue

        cached = get_embedding(vid, MODEL_NAME)
        if cached is not None:
            embs.append(cached.astype(np.float32))
            continue

        txt = str(r.get("job_text", "") or "")
        vec = model.encode([txt], normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)[0]
        put_embedding(vid, MODEL_NAME, vec)
        embs.append(vec)

    return np.vstack(embs).astype(np.float32)


def _rank_with_faiss(job_embs: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    # brute cosine since job_embs already normalized
    job_embs = job_embs.astype(np.float32)
    query_emb = query_emb.astype(np.float32)
    return (job_embs @ query_emb[0]).astype(np.float32)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_term(area_id: int, term: str, per_term: int, period_days: int, refresh_nonce: int) -> List[dict]:
    return fetch_vacancies(
        text=term,
        area=area_id,
        max_items=per_term,
        per_page=min(50, per_term),
        period_days=period_days,
        order_by="publication_time",
    )


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_default_startup(area_id: int, period_days: int, refresh_nonce: int) -> List[dict]:
    return fetch_vacancies(
        text=None,
        area=area_id,
        max_items=DEFAULT_STARTUP_LIMIT,
        per_page=50,
        period_days=period_days,
        order_by="publication_time",
    )


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_details(vacancy_id: str) -> Dict:
    return vacancy_details(vacancy_id)


def _details_cached(details_cache: Dict[str, Dict], vacancy_id: str) -> Dict:
    vid = str(vacancy_id).strip()
    if not vid:
        return {}
    if vid in details_cache:
        return details_cache[vid]
    try:
        d = _fetch_details(vid)
    except Exception:
        d = {}
    if len(details_cache) >= DETAILS_CACHE_MAX:
        # drop an arbitrary item to keep bounded
        details_cache.pop(next(iter(details_cache)))
    details_cache[vid] = d
    return d


def _extract_skills_from_full(full: Dict) -> List[str]:
    # HH details: key_skills is list of dicts {"name": "..."}
    ks = full.get("key_skills") or []
    out = []
    if isinstance(ks, list):
        for x in ks:
            if isinstance(x, dict) and x.get("name"):
                out.append(str(x["name"]).strip())
    return out[:20]


def _extract_description_from_full(full: Dict) -> str:
    desc = full.get("description") or ""
    return _strip_html(str(desc))


def _ensure_session_state_defaults():
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
        st.session_state.details_cache = {}

    if "refresh_nonce" not in st.session_state:
        st.session_state.refresh_nonce = 0

    if "terms_text" not in st.session_state:
        st.session_state.terms_text = ""

    if "selected_resume_id" not in st.session_state:
        st.session_state.selected_resume_id = None
    if "selected_resume_label" not in st.session_state:
        st.session_state.selected_resume_label = None

    if "last_results_df" not in st.session_state:
        st.session_state.last_results_df = pd.DataFrame()

    if "last_global_refresh_nonce" not in st.session_state:
        st.session_state.last_global_refresh_nonce = -1


_ensure_session_state_defaults()

# ---------- auth ----------
def _auth_block():
    st.title("💼 HH Job Recommender")

    token = st.session_state.get("session_token", None)
    user = None
    if token:
        user = get_user_by_token(token)

    if user:
        st.session_state.user = user
        return

    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            u = authenticate(email, password)
            if u:
                token = create_session(u["id"])
                st.session_state.session_token = token
                st.session_state.user = u
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Create account"):
            try:
                create_user(email, password)
                st.success("Account created. Please login.")
            except Exception as e:
                st.error(f"Could not create user: {e}")


if not st.session_state.user:
    _auth_block()
    st.stop()

user_id = int(st.session_state.user["id"])

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    if st.button("Logout"):
        token = st.session_state.get("session_token", None)
        if token:
            delete_session(token)
        st.session_state.session_token = None
        st.session_state.user = None
        st.rerun()

    st.markdown("---")

    areas_tree = fetch_areas_tree()
    regions, cities = list_regions_and_cities(areas_tree)

    region = st.selectbox("Регион", regions, index=min(0, len(regions) - 1), key="region_sel")
    city_list = cities.get(region, [])
    city = st.selectbox("Город", city_list if city_list else ["—"], key="city_sel")

    # city -> area_id
    # In hh_areas.py, cities list contains tuples? If list is strings, we map by searching tree.
    # The helper already provides city name; we will pass it back through hh_areas search
    # For stability, keep existing behavior (it worked in main).
    # We keep a text input fallback if not found.
    area_id = st.text_input("Area ID (если нужно)", value="", key="area_id_txt")

    period_days = st.selectbox("Период (дни)", [7, 14, 30, 60], index=3, key="period_days_sel")
    page_size = st.selectbox("Размер страницы", [10, 20, 30, 50], index=1, key="page_size_sel")
    st.session_state.page_size = int(page_size)

    if st.button("🔄 Обновить вакансии"):
        st.session_state.refresh_nonce += 1
        st.success("Обновление запрошено. Перезапустите поиск/ленту.")

    st.markdown("---")
    st.markdown("## 🧾 Резюме")

    resumes = list_resumes(user_id)
    resume_opts = ["— None —"] + [f'{r["id"]}: {r["name"]}' for r in resumes]
    sel = st.selectbox("Выбрать резюме", resume_opts, index=0, key="resume_pick")

    resume_source = st.radio("Источник", ["None", "Created resume", "PDF resume"], index=0, key="resume_source")

    if resume_source == "Created resume":
        if sel != "— None —":
            rid = int(sel.split(":")[0])
            st.session_state.selected_resume_id = rid
            st.session_state.selected_resume_label = sel
    elif resume_source == "PDF resume":
        up = st.file_uploader("Загрузить PDF", type=["pdf"])
        if up is not None:
            try:
                st.session_state.pdf_text = _read_pdf_text(up.read())
                st.success("PDF прочитан")
            except Exception as e:
                st.error(f"Не удалось прочитать PDF: {e}")

    st.markdown("---")
    st.markdown("## 🔎 Термины (TF-IDF)")
    if st.button("Extract terms"):
        # extract from active resume
        txt = ""
        if resume_source == "Created resume" and st.session_state.get("selected_resume_id"):
            rid = int(st.session_state.selected_resume_id)
            rmatch = next((r for r in resumes if int(r["id"]) == rid), None)
            if rmatch:
                txt = rmatch.get("text", "") or ""
        elif resume_source == "PDF resume":
            txt = st.session_state.get("pdf_text", "") or ""
        if not txt.strip():
            st.warning("Сначала выберите/загрузите резюме")
        else:
            terms = extract_terms(txt, top_k=TERMS_MAX)
            st.session_state.terms_text = "\n".join(terms)

    st.text_area("Термины (каждый с новой строки)", height=180, key="terms_text")


# ---------- area_id resolution ----------
# keep the original behavior:
# if user didn't supply area_id, try to infer by city name search in tree
if not area_id.strip():
    # best-effort find city id by name
    # fetch_areas_tree provides dict structure; try to find matching city name
    def _find_area_id(tree, name: str):
        if not name or name == "—":
            return None
        name_l = name.strip().lower()

        def walk(node):
            if isinstance(node, dict):
                if str(node.get("name", "")).strip().lower() == name_l and node.get("id") is not None:
                    return int(node["id"])
                for ch in node.get("areas", []) or []:
                    r = walk(ch)
                    if r is not None:
                        return r
            elif isinstance(node, list):
                for x in node:
                    r = walk(x)
                    if r is not None:
                        return r
            return None

        return walk(tree)

    inferred = _find_area_id(areas_tree, city)
    if inferred is not None:
        area_id = str(inferred)
    else:
        # fallback to Novosibirsk if nothing found (user requirement earlier)
        area_id = "4"  # Novosibirsk area_id on HH is commonly 4

# ---------- favorites ----------
favorites = list_favorites(user_id)
favorites_set = {str(x["job_id"]) for x in favorites}


# ---------- timeline default from history ----------
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
                COL_MODE: r.get("working_mode", "") or "",
                COL_SALARY: r.get("salary_text", "") or "",
                COL_POSITION: r.get("title", "") or "",
                COL_DUTIES: r.get("snippet_resp", "") or "",
                COL_SKILLS: r.get("snippet_req", "") or "",
                COL_DESC: "",
                "alternate_url": r.get("url", "") or "",
                "published_at": r.get("published_at", "") or "",
                "similarity_score": r.get("similarity_score", pd.NA),
            }
        )

    df = pd.DataFrame(all_rows)
    if len(df):
        df["job_text"] = df.apply(_job_text, axis=1)
    else:
        df["job_text"] = ""
    return df


# ---------- main layout ----------
st.title("💼 HH Job Recommender")

colA, colB = st.columns([1.2, 1])

with colA:
    st.markdown("### Результаты")
with colB:
    st.markdown("### История / Избранное")

# ---------- saved searches list ----------
saved = list_saved_searches(user_id)
latest = get_latest_saved_search(user_id)

with colB:
    st.markdown("**Сохраненные поиски (последние 2-3):**")
    if saved:
        for s in saved:
            sid = int(s["id"])
            label = s.get("resume_label") or s.get("resume_key")
            st.write(f'• #{sid} — {label} — area={s["area_id"]} days={s["timeframe_days"]}')
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"🗑️ Удалить #{sid}", key=f"del_search_{sid}"):
                    delete_saved_search(sid)
                    st.rerun()
            with c2:
                st.caption(f'last_ranked={s.get("last_ranked_at") or "—"} | last_refresh={s.get("last_refresh_at") or "—"}')
    else:
        st.info("Нет сохраненных поисков. Выполните поиск с резюме, чтобы сохранить результаты.")

# ---------- compute resume_text + terms ----------
resume_text = ""
resume_source = st.session_state.get("resume_source", "None")

if resume_source == "Created resume" and st.session_state.get("selected_resume_id"):
    rid = int(st.session_state.selected_resume_id)
    rmatch = next((r for r in resumes if int(r["id"]) == rid), None)
    if rmatch:
        resume_text = str(rmatch.get("text", "") or "")
elif resume_source == "PDF resume":
    resume_text = str(st.session_state.get("pdf_text", "") or "")

terms = [t.strip() for t in st.session_state.terms_text.splitlines() if t.strip()]
terms = terms[:TERMS_MAX]
if len(terms) < TERMS_MIN:
    terms = list(dict.fromkeys(terms + ["python", "sql"]))[:TERMS_MIN]


# ---------- default behavior (no resume): show history timeline if exists, else show city+time feed ----------
try:
    _now = _now_utc()
    last_fetch_at = st.session_state.get("last_fetch_at", _now - timedelta(days=365))

    need_refresh = (_now - last_fetch_at) > timedelta(minutes=10) or (st.session_state.refresh_nonce > 0)

    if need_refresh:
        if resume_text.strip():
            # resume-based search: run ranking and save search results
            st.write("**Термины для HH:**", ", ".join(terms))

            with st.spinner("Глобальный индекс: обновление/загрузка..."):
                # If user clicked refresh button, refresh_nonce changes; force index refresh for this (area, days).
                nonce = int(st.session_state.get("refresh_nonce", 0))
                last_nonce = int(st.session_state.get("last_global_refresh_nonce", -1))
                force_refresh = nonce != last_nonce
                st.session_state.last_global_refresh_nonce = nonce

                did_refresh, msg = refresh_global_index(
                    GlobalIndexConfig(area_id=int(area_id), period_days=int(period_days), max_items=5000, per_page=50),
                    force=force_refresh,
                    min_hours_between_refresh=6,
                )
                st.caption(msg)

                g_index = load_global_index(int(area_id), int(period_days))

                # If index doesn't exist yet (first run), force refresh once.
                if g_index is None:
                    did_refresh, msg = refresh_global_index(
                        GlobalIndexConfig(area_id=int(area_id), period_days=int(period_days), max_items=5000, per_page=50),
                        force=True,
                        min_hours_between_refresh=0,
                    )
                    st.caption(msg)
                    g_index = load_global_index(int(area_id), int(period_days))

            if g_index is None:
                st.error("Не удалось загрузить глобальный индекс (проверьте HH API / User-Agent и права на запись в artifacts).")
                df = pd.DataFrame()
            else:
                model = SentenceTransformer(MODEL_NAME)
                with st.spinner("FAISS поиск по глобальному индексу + пост-фильтрация..."):
                    q = model.encode([resume_text], normalize_embeddings=True)
                    q = np.asarray(q, dtype=np.float32)

                    scores, ids = search_global_index(g_index, q, top_k=1200)
                    vid_list = [str(int(i)) for i in ids if int(i) != -1]

                    rows = get_global_vacancies_by_ids(vid_list)

                    out_rows = []
                    score_map = {str(int(i)): float(s) for s, i in zip(scores, ids) if int(i) != -1}
                    for r in rows:
                        vid = str(r.get("vacancy_id", "")).strip()
                        if not vid:
                            continue
                        out_rows.append(
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
                                "similarity_score": score_map.get(vid, pd.NA),
                            }
                        )

                    df = pd.DataFrame(out_rows)
                    if len(df):
                        df["job_text"] = df.apply(_job_text, axis=1)
                    else:
                        df["job_text"] = ""

                    # Preserve old "terms" semantics: post-filter results by term mentions.
                    if len(df) and terms:
                        pat = "|".join(re.escape(t.lower()) for t in terms if t)
                        if pat:
                            mask = df["job_text"].astype(str).str.lower().str.contains(pat, regex=True, na=False)
                            filtered = df[mask].copy()
                            if len(filtered) >= 15:
                                df = filtered

                    df = df.sort_values("similarity_score", ascending=False, na_position="last").reset_index(drop=True)

            # ---- SAVE HISTORY (resume-based search) ----
            try:
                if resume_source == "Created resume":
                    resume_id_to_save = int(rid)  # type: ignore[name-defined]
                    resume_key = f"rid:{resume_id_to_save}"
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
                    update_interval_hours=DEFAULT_UPDATE_INTERVAL_HOURS,
                    refresh_window_hours=DEFAULT_REFRESH_WINDOW_HOURS,
                )

                # keep only last 2-3 searches
                enforce_saved_search_limit(user_id=user_id, limit=SAVED_SEARCH_MAX)

                if len(df):
                    upsert_saved_search_results(search_id, df)

                touch_ranked(search_id)
                touch_refreshed(search_id)

            except Exception:
                pass

            st.session_state.last_results_df = df

        else:
            # if no resume: prefer history timeline; otherwise show default feed for area+time
            hist_df = _load_default_timeline_from_history(user_id, favorites_set)
            if len(hist_df):
                st.session_state.last_results_df = hist_df
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

    with c2:
        is_fav = vid in favorites_set
        if is_fav:
            if st.button("★", key=f"fav_{vid}"):
                remove_favorite(user_id, vid)
                st.rerun()
        else:
            if st.button("☆", key=f"fav_{vid}"):
                add_favorite(user_id, vid)
                st.rerun()

    with c3:
        if pct is not None:
            st.metric("Match", f"{pct}%")
        else:
            st.write("")

    # details expand
    with st.expander("Подробнее"):
        full = _details_cached(st.session_state.details_cache, vid)
        if full:
            desc = _extract_description_from_full(full)
            skills = _extract_skills_from_full(full)
            if skills:
                st.markdown("**Навыки:**")
                st.markdown(" ".join([f"<span class='skill-chip'>{_html.escape(s)}</span>" for s in skills]), unsafe_allow_html=True)
            if desc:
                st.markdown("**Описание:**")
                st.write(desc)
        else:
            st.write("Нет подробностей (ошибка загрузки или лимиты HH).")

    st.markdown("</div>", unsafe_allow_html=True)


with colA:
    if df is None or not len(df):
        st.info("Нет результатов. Выберите резюме и выполните поиск, либо обновите ленту.")
    else:
        total = len(df)
        page_size = int(st.session_state.page_size)
        pages = max(1, math.ceil(total / page_size))

        # page controls
        pcols = st.columns([1, 2, 2, 1])
        with pcols[0]:
            if st.button("◀"):
                st.session_state.page = max(1, int(st.session_state.page) - 1)
        with pcols[1]:
            st.write(f"Page {st.session_state.page} / {pages}")
        with pcols[2]:
            st.slider("Перейти", min_value=1, max_value=pages, value=int(st.session_state.page), key="page_slider")
            st.session_state.page = int(st.session_state.page_slider)
        with pcols[3]:
            if st.button("▶"):
                st.session_state.page = min(pages, int(st.session_state.page) + 1)

        start = (int(st.session_state.page) - 1) * page_size
        end = min(total, start + page_size)
        view = df.iloc[start:end].to_dict(orient="records")

        for i, r in enumerate(view, start=start + 1):
            render_job(r, i)
