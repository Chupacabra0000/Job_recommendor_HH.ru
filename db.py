# db.py
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.getenv("APP_DB_PATH", "app.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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

    # ---- embeddings cache (global) ----
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

    # ---- saved searches (per user history) ----
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
          published_at TEXT,
          title TEXT,
          employer TEXT,
          url TEXT,
          snippet_req TEXT,
          snippet_resp TEXT,
          salary_text TEXT,
          score REAL,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (search_id, vacancy_id),
          FOREIGN KEY(search_id) REFERENCES saved_searches(id)
        )
        """
    )

    # ---- NEW: global vacancy metadata (for global index + fast UI) ----
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

    # ---- NEW: global index state ----
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


# ---------------- Password hashing ----------------
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
        algo, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
        dk = _pbkdf2_hash(password, salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ---------------- Users ----------------
def create_user(email: str, password: str) -> Tuple[bool, str]:
    if len(password) < 6:
        return False, "Пароль слишком короткий (мин. 6 символов)."
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users(email, password_hash) VALUES(?, ?)",
            (email.strip().lower(), hash_password(password)),
        )
        conn.commit()
        return True, "Пользователь создан."
    except sqlite3.IntegrityError:
        return False, "Email уже зарегистрирован."
    finally:
        conn.close()


def authenticate(email: str, password: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return dict(row)


# ---------------- Resumes ----------------
def list_resumes(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, text, created_at FROM resumes WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_resume(user_id: int, name: str, text: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO resumes(user_id, name, text) VALUES(?, ?, ?)",
        (user_id, name, text),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return int(rid)


def delete_resume(user_id: int, resume_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM resumes WHERE user_id = ? AND id = ?", (user_id, resume_id))
    conn.commit()
    conn.close()


# ---------------- Favorites ----------------
def list_favorites(user_id: int) -> List[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT job_id FROM favorites WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [str(r["job_id"]) for r in rows]


def add_favorite(user_id: int, job_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO favorites(user_id, job_id) VALUES(?, ?)",
        (user_id, str(job_id)),
    )
    conn.commit()
    conn.close()


def remove_favorite(user_id: int, job_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM favorites WHERE user_id = ? AND job_id = ?", (user_id, str(job_id)))
    conn.commit()
    conn.close()


# ---------------- Sessions (login survives refresh) ----------------
def _rand_token(nbytes: int = 24) -> str:
    return base64.urlsafe_b64encode(os.urandom(nbytes)).decode("ascii").rstrip("=")


def create_session(user_id: int, days_valid: int = 30) -> str:
    token = _rand_token()
    now = datetime.datetime.utcnow()
    exp = now + datetime.timedelta(days=days_valid)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions(token, user_id, expires_at) VALUES(?, ?, ?)",
        (token, user_id, exp.isoformat() + "Z"),
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.*, s.expires_at AS expires_at
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token = ?
        """,
        (token,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        exp = datetime.datetime.fromisoformat(str(row["expires_at"]).replace("Z", ""))
        if datetime.datetime.utcnow() > exp:
            delete_session(token)
            return None
    except Exception:
        pass
    d = dict(row)
    d.pop("expires_at", None)
    return d


def delete_session(token: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


# ---------------- Embeddings cache ----------------
def get_embedding(vacancy_id: str, model_name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT dim, emb_blob FROM vacancy_embeddings WHERE vacancy_id = ? AND model_name = ?",
        (str(vacancy_id), str(model_name)),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    # emb_blob is stored as raw float32 bytes
    import numpy as np

    dim = int(row["dim"])
    vec = np.frombuffer(row["emb_blob"], dtype=np.float32)
    if vec.size != dim:
        return None
    return vec


def put_embedding(vacancy_id: str, model_name: str, vec) -> None:
    import numpy as np

    v = np.asarray(vec, dtype=np.float32)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO vacancy_embeddings(vacancy_id, model_name, dim, emb_blob, updated_at)
        VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(vacancy_id, model_name)
        DO UPDATE SET dim=excluded.dim, emb_blob=excluded.emb_blob, updated_at=CURRENT_TIMESTAMP
        """,
        (str(vacancy_id), str(model_name), int(v.size), sqlite3.Binary(v.tobytes())),
    )
    conn.commit()
    conn.close()


# ---------------- Saved searches (history) ----------------
def list_saved_searches(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ss.*, r.name AS resume_name
        FROM saved_searches ss
        LEFT JOIN resumes r ON r.id = ss.resume_id
        WHERE ss.user_id = ?
        ORDER BY ss.created_at DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_saved_search(user_id: int) -> Optional[Dict[str, Any]]:
    rows = list_saved_searches(user_id)
    return rows[0] if rows else None


def create_or_get_saved_search(
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
        (user_id, resume_key, int(area_id), int(timeframe_days)),
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
            int(resume_id) if resume_id else None,
            str(resume_key),
            resume_label,
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


def upsert_saved_search_results(search_id: int, rows: List[Dict[str, Any]]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO saved_search_results
          (search_id, vacancy_id, published_at, title, employer, url,
           snippet_req, snippet_resp, salary_text, score, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(search_id, vacancy_id)
        DO UPDATE SET
          published_at=excluded.published_at,
          title=excluded.title,
          employer=excluded.employer,
          url=excluded.url,
          snippet_req=excluded.snippet_req,
          snippet_resp=excluded.snippet_resp,
          salary_text=excluded.salary_text,
          score=excluded.score,
          updated_at=CURRENT_TIMESTAMP
        """,
        [
            (
                int(search_id),
                str(r.get("vacancy_id")),
                r.get("published_at"),
                r.get("title"),
                r.get("employer"),
                r.get("url"),
                r.get("snippet_req"),
                r.get("snippet_resp"),
                r.get("salary_text"),
                float(r.get("score")) if r.get("score") is not None else None,
            )
            for r in rows
        ],
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


def enforce_saved_search_limit(user_id: int, keep_last: int = 3) -> List[int]:
    """
    Keeps only last N saved searches per user.
    Returns list of deleted search_ids.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM saved_searches WHERE user_id=? ORDER BY created_at DESC",
        (int(user_id),),
    )
    ids = [int(r["id"]) for r in cur.fetchall()]
    to_delete = ids[keep_last:]
    for sid in to_delete:
        cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (sid,))
        cur.execute("DELETE FROM saved_searches WHERE id=?", (sid,))
    conn.commit()
    conn.close()
    return to_delete


def delete_saved_search(user_id: int, search_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (int(search_id),))
    cur.execute("DELETE FROM saved_searches WHERE id=? AND user_id=?", (int(search_id), int(user_id)))
    conn.commit()
    conn.close()


def list_default_timeline(user_id: int, limit: int = 5000) -> List[Dict[str, Any]]:
    """
    Timeline = ALL vacancies stored across last 2-3 searches,
    deduped by vacancy_id, keeping MOST RECENT score.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        WITH last_searches AS (
          SELECT id, created_at
          FROM saved_searches
          WHERE user_id = ?
          ORDER BY created_at DESC
          LIMIT 3
        ),
        scored AS (
          SELECT r.*
          FROM saved_search_results r
          JOIN last_searches s ON s.id = r.search_id
        ),
        ranked AS (
          SELECT
            vacancy_id,
            published_at, title, employer, url, snippet_req, snippet_resp, salary_text,
            score,
            updated_at,
            ROW_NUMBER() OVER (PARTITION BY vacancy_id ORDER BY updated_at DESC) AS rn
          FROM scored
        )
        SELECT * FROM ranked
        WHERE rn = 1
        ORDER BY score DESC
        LIMIT ?
        """,
        (int(user_id), int(limit)),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- NEW: Global vacancy metadata ----------------
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


def get_global_vacancy(vacancy_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM global_vacancies WHERE vacancy_id=?", (str(vacancy_id),))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------- NEW: Global index state ----------------
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
