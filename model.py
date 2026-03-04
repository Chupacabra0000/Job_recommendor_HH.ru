import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    # L2 normalize rows
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


class JobRecommendationSystem:
    """
    SentenceTransformer-based matcher (primary).
    TF-IDF is computed lazily only for explanations.
    """

    def __init__(
        self,
        jobs_csv: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        enable_explanations: bool = True,
        stop_words: str = "english",
    ):
        self.jobs_csv = jobs_csv
        self.model_name = model_name
        self.batch_size = batch_size
        self.enable_explanations = enable_explanations
        self.stop_words = stop_words

        _ensure_dir(ARTIFACT_DIR)
        self.jobs_path = os.path.join(ARTIFACT_DIR, "jobs_clean.parquet")
        self.emb_path = os.path.join(ARTIFACT_DIR, "job_embeddings.npy")

        self.model = SentenceTransformer(self.model_name)

        self.jobs_df = self._load_or_prepare_jobs()
        self.embeddings = self._load_or_build_embeddings()

        # lazy TF-IDF components (fit-on-demand)
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None

    # ---------------- Data prep ----------------
    def _build_job_text(self, df: pd.DataFrame) -> pd.Series:
        cols = [
            "workplace",
            "working_mode",
            "position",
            "job_role_and_duties",
            "requisite_skill",
            "offer_details",
            "salary",
        ]
        present = [c for c in cols if c in df.columns]
        # join with spaces; handle NaNs
        text = df[present].fillna("").astype(str).agg(" ".join, axis=1)
        text = text.str.replace(r"\s+", " ", regex=True).str.strip()
        return text

    def _load_or_prepare_jobs(self) -> pd.DataFrame:
        if os.path.exists(self.jobs_path):
            df = pd.read_parquet(self.jobs_path)
            # ensure job_text exists
            if "job_text" not in df.columns:
                df["job_text"] = self._build_job_text(df)
                df.to_parquet(self.jobs_path, index=False)
            return df

        df = pd.read_csv(self.jobs_csv)
        df["job_text"] = self._build_job_text(df)
        df.to_parquet(self.jobs_path, index=False)
        return df

    def _load_or_build_embeddings(self) -> np.ndarray:
        if os.path.exists(self.emb_path):
            emb = np.load(self.emb_path)
            # safety: ensure float32 and normalized
            emb = emb.astype("float32")
            emb = _normalize_rows(emb)
            return emb

        texts = self.jobs_df["job_text"].fillna("").astype(str).tolist()
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # already L2 normalized
        )
        emb = np.asarray(emb, dtype="float32")
        np.save(self.emb_path, emb)
        return emb

    # ---------------- Scoring ----------------
    def score_all_jobs(self, resume_text: str) -> pd.DataFrame:
        """
        Returns a copy of jobs_df with 'similarity_score' column (0..1).
        """
        resume_text = (resume_text or "").strip()
        df = self.jobs_df.copy()
        if not resume_text:
            df["similarity_score"] = np.nan
            return df

        q = self.model.encode([resume_text], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")  # (1, dim)
        scores = (self.embeddings @ q.T).reshape(-1)  # cosine similarity
        df["similarity_score"] = scores
        return df

    # ---------------- Explainability (lazy TF-IDF) ----------------
    def _get_vectorizer(self) -> TfidfVectorizer:
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(stop_words=self.stop_words, ngram_range=(1, 2), max_features=5000)
        return self._tfidf_vectorizer

    def explain_match(self, resume_text: str, job_text: str, top_k: int = 10) -> Dict[str, List[str]]:
        """
        Lightweight explainer:
        - Fit TF-IDF on [resume, job] only (lazy per call) to get top keywords
        - Return intersection as matched keywords
        """
        resume_text = (resume_text or "").strip()
        job_text = (job_text or "").strip()
        if not resume_text or not job_text:
            return {"resume_keywords": [], "job_keywords": [], "matched_keywords": []}

        vec = TfidfVectorizer(stop_words=self.stop_words, ngram_range=(1, 2), max_features=3000)
        X = vec.fit_transform([resume_text, job_text])
        terms = np.array(vec.get_feature_names_out())

        def top_terms(row_idx: int) -> List[str]:
            row = X[row_idx].toarray().ravel()
            if row.sum() <= 0:
                return []
            top = row.argsort()[-top_k:][::-1]
            return [t for t in terms[top] if row[top[list(terms[top]).index(t)] if False else True] is not None][:top_k]

        # Better: use sparse indices
        def top_terms_sparse(row_idx: int) -> List[str]:
            row = X[row_idx]
            if row.nnz == 0:
                return []
            data = row.data
            idx = row.indices
            order = np.argsort(data)[-top_k:][::-1]
            return [terms[idx[i]] for i in order]

        rk = top_terms_sparse(0)
        jk = top_terms_sparse(1)
        matched = [t for t in rk if t in set(jk)]
        return {"resume_keywords": rk, "job_keywords": jk, "matched_keywords": matched[:top_k]}
