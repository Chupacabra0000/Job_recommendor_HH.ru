import sqlite3
import os
import base64
import hashlib
import hmac
import datetime
from typing import Optional, List, Dict, Any, Tuple

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

    # ---- embedding cache (global, reused across searches/users) ----
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

    # ---- resume-based saved searches (search history) ----
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

    # =========================
    # ADDED FOR GLOBAL PRE-INDEX
    # =========================

    # ---- global vacancy pool (for global FAISS index) ----
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

    # ---- key/value storage for global index refresh timestamps ----
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS global_index_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


# ---------------- Password hashing (stdlib: PBKDF2) ----------------
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
    """Deletes resume row only. Call search cleanup separately."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM resumes WHERE user_id = ? AND id = ?", (user_id, resume_id))

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

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

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


def remove_favorite(user_id: int, job_id: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM favorites WHERE user_id = ? AND job_id = ?", (user_id, str(job_id)))

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


# ---------------- Persistent sessions (login survive refresh) ----------------
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

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

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
        SELECT u.* FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token = ?
        """,
        (token,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    # expiry check
    cur.execute("SELECT expires_at FROM sessions WHERE token = ?", (token,))
    srow = cur.fetchone()
    conn.close()
    try:
        exp = datetime.datetime.fromisoformat(str(srow["expires_at"]).replace("Z", ""))
        if datetime.datetime.utcnow() > exp:
            delete_session(token)
            return None
    except Exception:
        return None

    return dict(row)


def delete_session(token: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


# ---------------- Vacancy embeddings cache ----------------
def get_embedding(vacancy_id: str, model_name: str) -> Optional[Tuple[int, bytes]]:
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
    return int(row["dim"]), row["emb_blob"]


def put_embedding(vacancy_id: str, model_name: str, dim: int, emb_blob: bytes) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO vacancy_embeddings(vacancy_id, model_name, dim, emb_blob, updated_at)
        VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(vacancy_id, model_name) DO UPDATE SET
            dim=excluded.dim,
            emb_blob=excluded.emb_blob,
            updated_at=CURRENT_TIMESTAMP
        """,
        (str(vacancy_id), str(model_name), int(dim), sqlite3.Binary(emb_blob)),
    )

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


# ---------------- Saved searches (resume-based history) ----------------
def list_saved_searches(user_id: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ss.*, r.name AS resume_name
        FROM saved_searches ss
        JOIN resumes r ON r.id = ss.resume_id
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
    """
    Returns (search_id, created_new).
    Unique by (user_id,resume_key,area_id,timeframe_days).
    resume_key examples:
      - "rid:123" for stored resumes
      - "pdf:<sha256>" for PDF text searches
    """
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
        sid = int(row["id"])
        cur.execute(
            """
            UPDATE saved_searches
            SET resume_id=?, resume_label=?, update_interval_hours=?, refresh_window_hours=?
            WHERE id=?
            """,
            (resume_id, resume_label, int(update_interval_hours), int(refresh_window_hours), sid),
        )
        conn.commit()
        conn.close()
        return sid, False

    cur.execute(
        """
        INSERT INTO saved_searches (user_id, resume_id, resume_key, resume_label, area_id, timeframe_days, update_interval_hours, refresh_window_hours)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            int(user_id),
            resume_id,
            str(resume_key),
            resume_label,
            int(area_id),
            int(timeframe_days),
            int(update_interval_hours),
            int(refresh_window_hours),
        ),
    )
    sid = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return sid, True


def touch_ranked(search_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE saved_searches SET last_ranked_at=CURRENT_TIMESTAMP WHERE id=?",
        (int(search_id),),
    )

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


def touch_refreshed(search_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE saved_searches SET last_refresh_at=CURRENT_TIMESTAMP WHERE id=?",
        (int(search_id),),
    )

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


def delete_saved_search(search_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (int(search_id),))
    cur.execute("DELETE FROM saved_searches WHERE id=?", (int(search_id),))

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


def delete_saved_searches_for_resume(user_id: int, resume_id: int) -> List[int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM saved_searches WHERE user_id=? AND resume_id=?",
        (user_id, resume_id),
    )
    rows = cur.fetchall()
    ids = [int(r["id"]) for r in rows]
    for sid in ids:
        cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (sid,))
        cur.execute("DELETE FROM saved_searches WHERE id=?", (sid,))

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()
    return ids


def enforce_saved_search_limit(user_id: int, keep_n: int = 3) -> List[int]:
    """
    Keep only last N searches for user. Returns deleted search_ids.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM saved_searches WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    )
    ids = [int(r["id"]) for r in cur.fetchall()]
    to_delete = ids[keep_n:]
    for sid in to_delete:
        cur.execute("DELETE FROM saved_search_results WHERE search_id=?", (sid,))
        cur.execute("DELETE FROM saved_searches WHERE id=?", (sid,))

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()
    return to_delete


# ---------------- Saved search results ----------------
def upsert_saved_search_results(search_id: int, rows: List[Dict[str, Any]]) -> None:
    """
    rows entries must include:
      vacancy_id, published_at, title, employer, url, snippet_req, snippet_resp, salary_text
    score can be provided (optional).
    """
    conn = get_conn()
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO saved_search_results(
              search_id, vacancy_id, published_at, title, employer, url,
              snippet_req, snippet_resp, salary_text, score, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(search_id, vacancy_id) DO UPDATE SET
              published_at=excluded.published_at,
              title=excluded.title,
              employer=excluded.employer,
              url=excluded.url,
              snippet_req=excluded.snippet_req,
              snippet_resp=excluded.snippet_resp,
              salary_text=excluded.salary_text,
              score=COALESCE(excluded.score, saved_search_results.score),
              updated_at=CURRENT_TIMESTAMP
            """,
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
                r.get("score"),
            ),
        )

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


def set_saved_search_scores(search_id: int, scores: Dict[str, float]) -> None:
    """
    Update score for given vacancy_ids. Does not touch others.
    """
    conn = get_conn()
    cur = conn.cursor()
    for vid, sc in scores.items():
        cur.execute(
            "UPDATE saved_search_results SET score=?, updated_at=CURRENT_TIMESTAMP WHERE search_id=? AND vacancy_id=?",
            (float(sc), int(search_id), str(vid)),
        )

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()


def prune_saved_search_results(search_id: int, cutoff_iso: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM saved_search_results WHERE search_id=? AND published_at IS NOT NULL AND published_at < ?",
        (int(search_id), str(cutoff_iso)),
    )
    deleted = cur.rowcount

    # ---- migration: add resume_key to saved_searches (support PDF searches) ----
    try:
        cur.execute("PRAGMA table_info(saved_searches)")
        cols = [r[1] for r in cur.fetchall()]
        if "resume_key" not in cols:
            cur.execute("ALTER TABLE saved_searches RENAME TO saved_searches_old;")
            cur.execute(
                """
                CREATE TABLE saved_searches (
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
                );
                """
            )
            cur.execute(
                """
                INSERT INTO saved_searches (
                    id,user_id,resume_id,resume_key,resume_label,area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                )
                SELECT
                    id,user_id,resume_id,
                    ('rid:' || resume_id) AS resume_key,
                    NULL AS resume_label,
                    area_id,timeframe_days,
                    update_interval_hours,refresh_window_hours,created_at,last_ranked_at,last_refresh_at
                FROM saved_searches_old;
                """
            )
            cur.execute("DROP TABLE saved_searches_old;")
    except Exception:
        pass

    conn.commit()
    conn.close()
    return int(deleted)


def list_saved_search_results(search_id: int, order_by_score: bool = True) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    if order_by_score:
        cur.execute(
            """
            SELECT * FROM saved_search_results
            WHERE search_id=?
            ORDER BY (score IS NULL) ASC, score DESC, published_at DESC
            """,
            (int(search_id),),
        )
    else:
        cur.execute(
            """
            SELECT * FROM saved_search_results
            WHERE search_id=?
            ORDER BY published_at DESC
            """,
            (int(search_id),),
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_default_timeline(user_id: int, limit: int = 5000) -> List[Dict[str, Any]]:
    """Merged saved vacancies across user's saved searches using MOST RECENT search score per vacancy."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT vacancy_id, published_at, title, employer, url, snippet_req, snippet_resp, salary_text, score
        FROM (
            SELECT
                r.*,
                s.last_ranked_at,
                s.created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY r.vacancy_id
                    ORDER BY COALESCE(s.last_ranked_at, s.created_at) DESC
                ) AS rn
            FROM saved_search_results r
            JOIN saved_searches s ON s.id = r.search_id
            WHERE s.user_id = ?
        )
        WHERE rn = 1
        ORDER BY (score IS NULL) ASC, score DESC
        LIMIT ?
        """,
        (int(user_id), int(limit)),
    )
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return rows


# =========================
# ADDED FOR GLOBAL PRE-INDEX
# =========================

def upsert_global_vacancies(rows: List[Dict[str, Any]]) -> None:
    """
    rows entries must include:
      vacancy_id, area_id, published_at, title, employer, url, snippet_req, snippet_resp, salary_text
    """
    conn = get_conn()
    cur = conn.cursor()

    for r in rows:
        cur.execute(
            """
            INSERT INTO global_vacancies(
                vacancy_id, area_id, published_at, title,
                employer, url, snippet_req, snippet_resp, salary_text, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(vacancy_id) DO UPDATE SET
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
            (
                str(r.get("vacancy_id")),
                int(r.get("area_id")) if r.get("area_id") is not None else None,
                r.get("published_at"),
                r.get("title"),
                r.get("employer"),
                r.get("url"),
                r.get("snippet_req"),
                r.get("snippet_resp"),
                r.get("salary_text"),
            ),
        )

    conn.commit()
    conn.close()


def get_global_index_state(key: str) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM global_index_state WHERE key=?", (str(key),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return row["value"]


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


def global_has_vacancy_ids(ids: List[str]) -> set:
    """Return set of vacancy_ids that already exist in global_vacancies."""
    ids = [str(x).strip() for x in ids if str(x).strip()]
    if not ids:
        return set()

    conn = get_conn()
    cur = conn.cursor()
    out = set()

    CHUNK = 500
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        placeholders = ",".join(["?"] * len(chunk))
        cur.execute(
            f"SELECT vacancy_id FROM global_vacancies WHERE vacancy_id IN ({placeholders})",
            tuple(chunk),
        )
        out |= {str(r["vacancy_id"]) for r in cur.fetchall()}

    conn.close()
    return out


def set_global_index_state_if_newer(key: str, value_iso_z: str) -> None:
    """
    Convenience: store only if newer lexicographically/ISO-wise.
    ISO Z timestamps compare correctly as strings in practice.
    """
    old = get_global_index_state(key)
    if (old is None) or (str(value_iso_z) > str(old)):
        set_global_index_state(key, value_iso_z)





def get_max_global_published_at(area_id: int) -> Optional[str]:
    """Newest published_at we have for this area_id (ISO string)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT MAX(published_at) AS mx FROM global_vacancies WHERE area_id=? AND published_at IS NOT NULL",
        (int(area_id),),
    )
    row = cur.fetchone()
    conn.close()
    if not row or not row["mx"]:
        return None
    return str(row["mx"])
