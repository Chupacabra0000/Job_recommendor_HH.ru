# Job Recommendation System (Streamlit)

## Setup
1. Put your dataset as `JobsFE.csv` in the project root (same folder as `app.py`).
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## (Optional) Precompute embeddings
This avoids recomputing embeddings on first run:
```bash
python build_index.py
```
Artifacts are saved to `artifacts/`.

## Run
```bash
streamlit run app.py
```

## Accounts (SQLite)
User accounts, resumes and favorites are stored in `app.db` in the project folder.
