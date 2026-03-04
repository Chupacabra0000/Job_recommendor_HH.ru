# Job Recommender (HH.ru API Based)

A resume-driven job recommendation system that fetches vacancies from **hh.ru** and ranks them using **semantic similarity (SentenceTransformers + FAISS)**.

The system analyzes a user's resume, extracts meaningful terms with **TF-IDF**, retrieves relevant vacancies from **HH.ru API**, embeds them, and ranks them based on similarity to the resume.

The application is built using **Python + Streamlit**.

---

# Main Features

### Resume-based job search

Users can:

• Upload a PDF resume
• Create resumes inside the system
• Use stored resumes for repeated searches

The resume text is used to find the most relevant vacancies.

---

### TF-IDF keyword extraction

From the resume the system automatically extracts **6–10 most meaningful keywords**.

Example:

Resume text → TF-IDF → keywords

```
python
machine learning
sql
data analysis
pandas
etl
```

Each keyword is sent as a separate search query to **HH.ru API**.

---

### Multi-query vacancy retrieval

The system sends multiple API queries:

```
keyword1 → HH API
keyword2 → HH API
keyword3 → HH API
...
```

Results are then:

• merged
• deduplicated
• converted to structured format

---

### Semantic ranking

Vacancies are ranked using:

```
SentenceTransformer embeddings
+
FAISS similarity search
```

Embedding model:

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This supports **Russian and English resumes**.

---

### Lazy loading vacancy descriptions

To improve performance:

• Full vacancy descriptions are **not fetched initially**

Instead:

1. Top 10 results load full descriptions automatically
2. Other results load descriptions only when the user expands them

This dramatically reduces API calls.

---

### Default vacancy timeline

When the user opens the application:

The system shows a **timeline of all stored vacancies from past searches**.

Rules:

• Favorites always appear at the top
• Remaining vacancies sorted by **similarity score**
• Only **last 2–3 searches are stored** to prevent storage overflow

---

### Favorites system

Users can mark vacancies as favorites.

Favorites:

• Always appear **at the top of the timeline**
• Persist across sessions
• Are stored in the SQLite database

---

### Smart vacancy storage

To avoid excessive storage usage:

Only **resume-based searches are stored**.

Search types:

| Search Type          | Stored |
| -------------------- | ------ |
| City + Time only     | ❌ No   |
| City + Time + Resume | ✅ Yes  |

Old searches are automatically removed.

---

# Project Architecture

```
app.py
```

Main Streamlit interface.

Handles:

• login
• resume upload
• vacancy search
• ranking
• timeline display

---

```
hh_client.py
```

Handles all communication with **HH.ru API**.

Functions:

• vacancy search
• vacancy detail retrieval

---

```
tfidf_terms.py
```

Extracts important keywords from resumes using TF-IDF.

---

```
embedding_store.py
```

Stores vacancy embeddings in SQLite.

Prevents re-embedding the same vacancy multiple times.

---

```
faiss_search_index.py
```

Handles FAISS index creation and removal.

---

```
search_cleanup.py
```

Maintains storage limits:

• keeps last 3 searches
• deletes outdated FAISS indexes

---

```
hh_areas.py
```

Fetches regions and cities from HH API.

Used to populate UI selectors.

---

```
db.py
```

Handles SQLite database operations:

• users
• resumes
• favorites
• saved searches
• sessions

---

# Requirements

Python version:

```
Python 3.10+
```

Required libraries:

```
streamlit
requests
numpy
pandas
scikit-learn
faiss-cpu
sentence-transformers
pymupdf
```

---

# Installation

Clone repository:

```
git clone https://github.com/Chupacabra0000/Job-Recommendor
cd Job-Recommendor
```

Create virtual environment (recommended):

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Linux / Mac:

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the application

Start Streamlit server:

```
streamlit run app.py
```

The application will open in your browser.

Default address:

```
http://localhost:8501
```

---

# HH API Notes

HH.ru requires a proper **User-Agent header**.

The project already sets:

```
HH-User-Agent: Job-Recommendor
```

If necessary you can override:

```
export HH_USER_AGENT="Job-Recommendor (your_email@example.com)"
```

---

# Performance Optimizations

The project includes several optimizations:

• lazy vacancy description loading
• embedding caching
• FAISS similarity search
• search result deduplication
• history size limits

These allow the system to process **thousands of vacancies quickly**.

---

# Storage

The project stores data in:

```
app.db
```

SQLite database storing:

• users
• resumes
• favorites
• search history

---

Embeddings stored in:

```
artifacts/embeddings.sqlite3
```

FAISS indexes stored in:

```
artifacts/faiss/
```

---

# Future Improvements

Possible improvements include:

• asynchronous HH API requests
• Redis caching
• background vacancy updates
• vector database integration (Qdrant / Pinecone)

---

# License

MIT License

---

