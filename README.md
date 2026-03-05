

# HH.ru Semantic Job Recommender

A **semantic job recommendation system** built for the HH.ru job platform using **vector search, transformer embeddings, and FAISS indexing**.

Instead of traditional keyword matching, the system ranks job vacancies by **semantic similarity to a user's resume**, providing more relevant recommendations.

The project demonstrates **modern ML search architecture** including:

* transformer embeddings
* vector indexing
* incremental indexing
* memory-mapped vector storage
* scalable search pipelines

---

# Demo Overview

Workflow:

```id="dcyoaj"
User Resume
      │
      ▼
TF-IDF keyword extraction
      │
      ▼
HH API vacancy retrieval
      │
      ▼
SentenceTransformer embeddings
      │
      ▼
FAISS vector similarity search
      │
      ▼
Ranked job recommendations
```

---

# Key Features

## Semantic job matching

Resumes and job descriptions are converted into **vector embeddings** using:

```id="frt9p6"
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This allows the system to match **meaning**, not just keywords.

Example:

| Resume Skill         | Matching Vacancy |
| -------------------- | ---------------- |
| Python data analysis | Data engineer    |
| Machine learning     | AI engineer      |
| Backend APIs         | Python developer |

---

## Global vacancy vector index

Vacancies are stored in a **shared global FAISS index**, allowing extremely fast search across thousands of jobs.

Architecture:

```id="ikxhb6"
HH API
   │
   ▼
Global Vacancy Store
(SQLite metadata)
   │
   ▼
Vector Store (memmap)
   │
   ▼
FAISS Index
```

All users share the same index, making the system scalable.

---

## Incremental index updates

The system uses **Policy A indexing strategy**:

| Update type        | Description                               |
| ------------------ | ----------------------------------------- |
| Incremental update | Only new vacancies are embedded and added |
| Full rebuild       | Performed every 24 hours                  |

This reduces compute costs while keeping the index fresh.

---

## Memory-mapped vector storage

Instead of storing embeddings in a database, vectors are stored in a **memory-mapped file**.

Advantages:

* extremely fast loading
* minimal memory overhead
* scalable to large vector sets

Structure:

```id="r2ut87"
artifacts/
   vector_store/
      model_name/
         vecs.f32
         ids.npy
         meta.json
```

This allows fast retrieval of vectors without loading the entire dataset into memory.

---

## FAISS vector search

Similarity search uses **FAISS** with inner-product similarity.

Benefits:

* extremely fast nearest-neighbor search
* optimized for large vector datasets
* widely used in production ML systems

Typical search latency:

```id="uifj0r"
50-150 ms
```

---

## Resume-driven multi-query retrieval

The system extracts keywords from resumes using **TF-IDF**.

Example:

```id="st5vfe"
Resume:
Python, SQL, machine learning

Extracted search terms:
python
machine learning
data analysis
pandas
numpy
sql
```

Each term triggers an HH API search and results are merged.

---

## Search history & timeline

User searches are stored in a database.

The system automatically generates a **default vacancy timeline** based on:

* recent searches
* resume similarity
* favorites

---

## Favorites system

Users can bookmark interesting vacancies.

Favorites are prioritized in the recommendation view.

---

# System Architecture

High-level architecture:

```id="uzfy2t"
                ┌───────────────┐
                │   User Resume │
                └───────┬───────┘
                        │
                        ▼
           ┌────────────────────────┐
           │ SentenceTransformer ML │
           └────────────┬───────────┘
                        │
                        ▼
                Resume Vector
                        │
                        ▼
               FAISS Vector Search
                        │
                        ▼
             Top Similar Vacancy IDs
                        │
                        ▼
               SQLite Metadata Store
                        │
                        ▼
                  Streamlit UI
```

---

# Global Index Pipeline

Vacancy indexing pipeline:

```id="wjgsoj"
HH API
   │
   ▼
Vacancy Fetch
   │
   ▼
Text Preprocessing
   │
   ▼
Embedding Generation
   │
   ▼
Vector Store (memmap)
   │
   ▼
FAISS Index Update
```

This pipeline runs periodically to keep the index up to date.

---

# Project Structure

```id="elq8hr"
project/
│
├── app.py
│   Streamlit application UI
│
├── db.py
│   SQLite database interface
│
├── hh_client.py
│   HH.ru API client
│
├── hh_areas.py
│   Region and city retrieval
│
├── tfidf_terms.py
│   Resume keyword extraction
│
├── search_cleanup.py
│   Search history cleanup
│
├── embedding_store.py
│   Legacy embedding cache
│
├── vector_store.py
│   Memory-mapped vector storage
│
├── global_index_manager.py
│   Global vacancy index manager
│
├── global_faiss_index.py
│   FAISS index utilities
│
└── artifacts/
    ├── vector_store/
    └── global_index/
```

---

# Installation

Clone repository:

```bash id="y9cn04"
git clone https://github.com/yourusername/job_recommendor_hh.ru
cd job_recommendor_hh.ru
```

Create virtual environment:

```id="2m1ycy"
python -m venv venv
```

Activate environment:

Linux / Mac

```id="4j4fck"
source venv/bin/activate
```

Windows

```id="gllh8x"
venv\Scripts\activate
```

Install dependencies:

```id="k2b7u0"
pip install -r requirements.txt
```

---

# Running the Application

Start the Streamlit app:

```id="4l3lf3"
streamlit run app.py
```

Open browser:

```id="zw47f8"
http://localhost:8501
```

---

# Performance

Performance improvements achieved through vector indexing:

| Operation             | Traditional approach | This system |
| --------------------- | -------------------- | ----------- |
| Vacancy ranking       | 10–25 seconds        | <200 ms     |
| Embedding computation | every search         | cached      |
| API calls             | many                 | minimal     |
| Search complexity     | O(N²)                | FAISS ANN   |

---

# Engineering Highlights

This project demonstrates several **modern ML system design patterns**:

### Vector search architecture

Widely used in:

* recommendation systems
* semantic search engines
* retrieval-augmented generation (RAG)

### Incremental indexing

Avoids expensive full rebuilds.

### Memory-mapped vector storage

Allows large embedding datasets with low memory usage.

### Hybrid storage design

```id="jqu7gp"
Vectors → memmap
Metadata → SQLite
Index → FAISS
```

---

# Limitations

* Depends on HH.ru API availability
* Job descriptions from snippets may be incomplete
* Index size currently optimized for ~5000–10000 vacancies

---

# Future Improvements

Possible extensions:

* background worker for indexing
* distributed vector search
* larger FAISS indexes (IVF/HNSW)
* improved resume parsing
* job deduplication

---

# Technologies Used

* Python
* Streamlit
* Sentence Transformers
* FAISS
* NumPy
* SQLite
* Requests




