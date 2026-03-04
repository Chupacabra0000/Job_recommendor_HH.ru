from __future__ import annotations

import os
import sqlite3
import hashlib
import base64
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import pandas as pd  # ✅ needed because we use pd.DataFrame in annotations

DB_PATH = os.getenv("DB_PATH", "app.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Better concurrency behavior for Streamlit
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # ---- core tables ----
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, job_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # ---- embeddings cache (global, reused across searches/users) ----
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vacancy_embeddings (
            vacancy_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            dim INTEGER NOT NULL,
            emb_blob BLOB NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (vacancy_id, model_name)
        )
        """
    )

    # ---- saved searches ----
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            resume_id INTEGER,
            resume_key TEXT NOT NULL,
            resume_label TEXT,
            area_id INTEGER NOT NULL,
            timeframe_days INTEGER NOT NULL,
            update_interval_hours INTEGER NOT NULL DEFAULT 24,
            refresh_window_hours INTEGER NOT NULL DEFAULT 24,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_ranked_at TEXT,
            last_refresh_at TEXT,
            UNIQUE(user_id, resume_key, area_id, timeframe_days),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(resume_id) REFERENCES resumes(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_search_results (
            search_id INTEGER NOT NULL,
            vacancy_id TEXT NOT NULL,
            similarity_score REAL,
            title TEXT,
            employer TEXT,
            url TEXT,
            published_at TEXT,
            snippet_req TEXT,
            snippet_resp TEXT,
            salary_text TEXT,
            working_mode TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (search_id, vacancy_id),
            FOREIGN KEY(search_id) REFERENCES saved_searches(id)
        )
        """
    )

    # ---- global vacancy metadata ----
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS global_vacancies (
            vacancy_id TEXT PRIMARY KEY,
            area_id INTEGER NOT NULL,
            published_at TEXT,
            title TEXT,
            employer TEXT,
            url TEXT,
            snippet_req TEXT,
            snippet_resp TEXT,
            salary_text TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # ---- global index state ----
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS global_index_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    conn.commit()
    conn.close()


# ---------------- Password hashing (PBKDF2) ----------------
# stored format: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>

def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 200_000
    dk = _pbkdf2_hash(password, salt, iterations)
    return "pbkdf2_sha256$%d$%s$%s" % (
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, it_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(it_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
        dk = _pbkdf2_hash(password, salt, iterations)
        return dk == expected
    except Exception:
        return False


# ---------------- Users ----------------

def create_user(email: str, password: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    pw_hash = hash_password(password)
    cur.execute("INSERT INTO users(email, password_hash) VALUES (?, ?)", (email, pw_hash))
    conn.commit()
    uid = int(cur.lastrowid)
    conn.close()
    return uid


def authenticate(email: str, password: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if verify_password(password, row["password_hash"]):
        return dict(row)
    return None


# ---------------- Sessions ----------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_session(user_id: int, ttl_hours: int = 24 * 14) -> str:
    token = base64.urlsafe_b64encode(os.urandom(24)).decode("ascii").rstrip("=")
    expires = _utcnow() + timedelta(hours=int(ttl_hours))
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions(token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, int(user_id), expires.isoformat().replace("+00:00", "Z")),
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE token=?", (token,))
    s = cur.fetchone()
    if not s:
        conn.close()
        return None

    exp = s["expires_at"] or ""
    try:
        exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        if _utcnow() > exp_dt:
            conn.close()
            return None
    except Exception:
        pass

    cur.execute("SELECT * FROM users WHERE id=?", (int(s["user_id"]),))
    u = cur.fetchone()
    conn.close()
    return dict(u) if u else None


def delete_session(token: str) -> None:
    if not token:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()


# ---------------- Resumes ----------------

def list_resumes(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM resumes WHERE user_id=? ORDER BY created_at DESC", (int(user_id),))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ---------------- Favorites ----------------

def list_favorites(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM favorites WHERE user_id=? ORDER BY created_at DESC", (int(user_id),))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def add_favorite(user_id: int, job_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO favorites(user_id, job_id) VALUES (?, ?)",
        (int(user_id), str(job_id)),
    )
    conn.commit()
    conn.close()


def remove_favorite(user_id: int, job_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM favorites WHERE user_id=? AND job_id=?", (int(user_id), str(job_id)))
    conn.commit()
    conn.close()


# ---------------- Saved searches ----------------

def list_saved_searches(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM saved_searches WHERE user_id=? ORDER BY created_at DESC",
        (int(user_id),),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_latest_saved_search(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM saved_searches WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
        (int(user_id),),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_or_get_saved_search(
    *,
    user_id: int,
    resume_key: str,
    area_id: int,
    timeframe_days: int,
    resume_id: Optional[int] = None,
    resume_label: Optional[str] = None,
    update_interval_hours: int = 24,
    refresh_window_hours: int = 24,
) -> Tuple[int, bool]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id FROM saved_searches
        WHERE user_id=? AND resume_key=? AND area_id=? AND timeframe_days=?
        """,
        (int(user_id), str(resume_key), int(area_id), int(timeframe_days)),
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return int(row["id"]), False

    cur.execute(
        """
        INSERT INTO saved_searches
          (user_id, resume_id, resume_key, resume_label, area_id, timeframe_days,
           update_interval_hours, refresh_window_hours)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            int(resume_id) if resume_id is not None else None,
            str(resume_key),
            str(resume_label) if resume_label is not None else None,
            int(area_id),
            int(timeframe_days),
            int(update_interval_hours),
            int(refresh_window_hours),
        ),
    )
    conn.commit()
    sid = int(cur.lastrowid)
    conn.close()
    return sid, True


def enforce_saved_search_limit(user_id: int, limit: int = 3) -> None:
    limit = int(limit)
    if limit <= 0:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM saved_searches WHERE user_id=? ORDER BY created_at DESC",
        (int(user_id),),
    )
    ids = [int(r["id"]) for r in cur.fetchall()]
    if len(ids) <= limit:
        conn.close()
        return

    to_delete = ids[limit:]
    for sid in to_delete:
        cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (int(sid),))
        cur.execute("DELETE FROM saved_searches WHERE id=?", (int(sid),))

    conn.commit()
    conn.close()


def delete_saved_search(search_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (int(search_id),))
    cur.execute("DELETE FROM saved_searches WHERE id=?", (int(search_id),))
    conn.commit()
    conn.close()


def upsert_saved_search_results(search_id: int, df: pd.DataFrame) -> None:
    if df is None or not len(df):
        return

    rows = []
    for _, r in df.iterrows():
        vid = str(r.get("Job Id", "")).strip()
        if not vid:
            continue
        rows.append(
            (
                int(search_id),
                vid,
                float(r["similarity_score"]) if "similarity_score" in r and pd.notna(r["similarity_score"]) else None,
                r.get("position"),
                r.get("workplace"),
                r.get("alternate_url"),
                r.get("published_at"),
                r.get("requisite_skill"),
                r.get("job_role_and_duties"),
                r.get("salary"),
                r.get("working_mode"),
            )
        )

    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO saved_search_results
          (search_id, vacancy_id, similarity_score, title, employer, url,
           published_at, snippet_req, snippet_resp, salary_text, working_mode, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(search_id, vacancy_id) DO UPDATE SET
          similarity_score=excluded.similarity_score,
          title=excluded.title,
          employer=excluded.employer,
          url=excluded.url,
          published_at=excluded.published_at,
          snippet_req=excluded.snippet_req,
          snippet_resp=excluded.snippet_resp,
          salary_text=excluded.salary_text,
          working_mode=excluded.working_mode,
          updated_at=CURRENT_TIMESTAMP
        """,
        rows,
    )
    conn.commit()
    conn.close()


def touch_ranked(search_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE saved_searches SET last_ranked_at=CURRENT_TIMESTAMP WHERE id=?",
        (int(search_id),),
    )
    conn.commit()
    conn.close()


def touch_refreshed(search_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE saved_searches SET last_refresh_at=CURRENT_TIMESTAMP WHERE id=?",
        (int(search_id),),
    )
    conn.commit()
    conn.close()


def list_default_timeline(user_id: int, limit: int = 5000) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.*
        FROM saved_search_results r
        JOIN (
          SELECT vacancy_id, MAX(updated_at) AS maxu
          FROM saved_search_results
          WHERE search_id IN (SELECT id FROM saved_searches WHERE user_id=?)
          GROUP BY vacancy_id
        ) x
        ON r.vacancy_id=x.vacancy_id AND r.updated_at=x.maxu
        ORDER BY r.updated_at DESC
        LIMIT ?
        """,
        (int(user_id), int(limit)),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ------------ Global vacancies metadata ----------------
def upsert_global_vacancies(rows: List[Dict[str, Any]]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO global_vacancies
          (vacancy_id, area_id, published_at, title, employer, url,
           snippet_req, snippet_resp, salary_text, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(vacancy_id)
        DO UPDATE SET
          area_id=excluded.area_id,
          published_at=excluded.published_at,
          title=excluded.title,
          employer=excluded.employer,
          url=excluded.url,
          snippet_req=excluded.snippet_req,
          snippet_resp=excluded.snippet_resp,
          salary_text=excluded.salary_text,
          updated_at=CURRENT_TIMESTAMP
        """,
        [
            (
                str(r["vacancy_id"]),
                int(r["area_id"]),
                r.get("published_at"),
                r.get("title"),
                r.get("employer"),
                r.get("url"),
                r.get("snippet_req"),
                r.get("snippet_resp"),
                r.get("salary_text"),
            )
            for r in rows
        ],
    )
    conn.commit()
    conn.close()


def get_global_vacancies_by_ids(vacancy_ids: List[str]) -> List[Dict[str, Any]]:
    ids = [str(v).strip() for v in vacancy_ids if str(v).strip()]
    if not ids:
        return []

    conn = get_conn()
    cur = conn.cursor()

    found: Dict[str, Dict[str, Any]] = {}
    chunk = 900
    for i in range(0, len(ids), chunk):
        part = ids[i: i + chunk]
        qs = ",".join("?" for _ in part)
        cur.execute(f"SELECT * FROM global_vacancies WHERE vacancy_id IN ({qs})", tuple(part))
        for row in cur.fetchall():
            d = dict(row)
            found[str(d.get("vacancy_id"))] = d

    conn.close()

    out: List[Dict[str, Any]] = []
    for vid in ids:
        r = found.get(str(vid))
        if r:
            out.append(r)
    return out


# ---------------- Global index state ----------------
def get_global_index_state(key: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM global_index_state WHERE key=?", (str(key),))
    row = cur.fetchone()
    conn.close()
    return str(row["value"]) if row else None


def set_global_index_state(key: str, value: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO global_index_state(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (str(key), str(value)),
    )
    conn.commit()
    conn.close()


# ---------------- Embeddings for embedding_store.py wrapper ----------------
def get_embedding_db(vacancy_id: str, model_name: str) -> Optional[bytes]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT emb_blob FROM vacancy_embeddings WHERE vacancy_id=? AND model_name=?",
        (str(vacancy_id), str(model_name)),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return row["emb_blob"]


def put_embedding_db(vacancy_id: str, model_name: str, dim: int, emb_blob: bytes) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO vacancy_embeddings(vacancy_id, model_name, dim, emb_blob, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(vacancy_id, model_name) DO UPDATE SET
          dim=excluded.dim,
          emb_blob=excluded.emb_blob,
          updated_at=CURRENT_TIMESTAMP
        """,
        (str(vacancy_id), str(model_name), int(dim), sqlite3.Binary(emb_blob)),
    )
    conn.commit()
    conn.close()
