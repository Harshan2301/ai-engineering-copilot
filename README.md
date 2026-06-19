# 🤖 AI Engineering Copilot

> A production-ready **Retrieval-Augmented Generation (RAG)** assistant for engineers.  
> Upload PDFs → Ask questions → Semantic search → Summarize — powered by **Gemini** + **ChromaDB** + **FastAPI** + **Streamlit**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Upload** | Upload engineering PDFs; text is chunked and embedded automatically |
| 💬 **RAG Q&A** | Ask natural-language questions with cited, grounded answers |
| 🔍 **Semantic Search** | Search by meaning across all documents with cosine-similarity scores |
| 📝 **Summarization** | Generate technical, executive, or bullet-point summaries |
| 🌑 **Dark Mode UI** | Premium dark-mode Streamlit frontend with Inter & JetBrains Mono fonts |

---

## 🗂️ Project Structure

```
ai-engineering-copilot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app factory + lifespan
│   │   ├── config.py            # Pydantic settings from .env
│   │   ├── core/
│   │   │   ├── gemini.py        # Gemini API client (generate + embed)
│   │   │   ├── chroma.py        # ChromaDB persistent client
│   │   │   └── logging_config.py
│   │   ├── routers/
│   │   │   ├── documents.py     # /api/v1/documents/*
│   │   │   └── rag.py           # /api/v1/rag/*
│   │   ├── schemas/
│   │   │   ├── documents.py
│   │   │   └── rag.py
│   │   └── services/
│   │       ├── document_service.py  # PDF parse → chunk → embed → upsert
│   │       └── rag_service.py       # Q&A / search / summarize
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Home dashboard
│   ├── pages/
│   │   ├── 1_📄_Documents.py
│   │   ├── 2_🔍_Search.py
│   │   ├── 3_💬_Ask.py
│   │   └── 4_📝_Summarize.py
│   ├── utils/
│   │   └── api_client.py        # httpx wrapper for all backend calls
│   ├── assets/
│   │   └── style.css            # Full dark-mode theme
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.12+
- A [Gemini API key](https://aistudio.google.com/app/apikey)
- Docker + Docker Compose (for containerised setup)

---

### 🐳 Option A — Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ai-engineering-copilot.git
cd ai-engineering-copilot

# 2. Create your .env file
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 3. Build and run
docker-compose up --build

# 4. Open the app
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

---

### 💻 Option B — Local Development

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp ../.env.example ../.env
# Edit .env and add your GEMINI_API_KEY

# Run the API server
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend

# Install dependencies (use the same venv or a new one)
pip install -r requirements.txt

# Point to the backend
export BACKEND_URL=http://localhost:8000/api/v1   # Windows: set BACKEND_URL=...

# Run Streamlit
streamlit run app.py
```

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Your Google Gemini API key |
| `GEMINI_MODEL` | | `gemini-1.5-pro` | Generative model name |
| `GEMINI_EMBED_MODEL` | | `models/text-embedding-004` | Embedding model |
| `CHROMA_PERSIST_DIR` | | `./chroma_data` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | | `engineering_docs` | Collection name |
| `UPLOAD_DIR` | | `./uploads` | File upload directory |
| `MAX_FILE_SIZE_MB` | | `50` | Max PDF size in MB |
| `CHUNK_SIZE` | | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | | `100` | Overlap between chunks |
| `SECRET_KEY` | ✅ | — | JWT / app secret key |
| `LOG_LEVEL` | | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 📡 API Reference

The FastAPI backend auto-generates interactive docs:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload a PDF |
| `GET` | `/api/v1/documents/` | List all documents |
| `DELETE` | `/api/v1/documents/{doc_id}` | Delete a document |
| `POST` | `/api/v1/rag/ask` | RAG question answering |
| `POST` | `/api/v1/rag/search` | Semantic similarity search |
| `POST` | `/api/v1/rag/summarize` | Document summarization |
| `GET` | `/health` | Backend health check |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Streamlit Frontend             │
│  Dashboard · Documents · Search · Ask   │
│            · Summarize                  │
└────────────────┬────────────────────────┘
                 │ HTTP (httpx)
┌────────────────▼────────────────────────┐
│           FastAPI Backend                │
│  /documents   /rag/ask                  │
│  /rag/search  /rag/summarize            │
└────────┬────────────────┬───────────────┘
         │                │
┌────────▼───────┐ ┌──────▼──────────────┐
│   ChromaDB     │ │    Gemini API        │
│ Vector Store   │ │ Generate · Embed     │
│ (persistent)   │ │ (google-generativeai)│
└────────────────┘ └─────────────────────┘
```

**RAG Pipeline:**
1. User uploads PDF → `pypdf` extracts text per page
2. Text → sliding-window chunks (800 chars, 100 overlap)
3. Each chunk → `text-embedding-004` → 768-dim vector
4. Vectors upserted to ChromaDB with metadata (doc_id, filename, page)
5. On query → embed question → cosine search → top-k chunks
6. Chunks + question → `gemini-1.5-pro` → grounded, cited answer

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM / Embeddings** | Google Gemini 1.5 Pro + text-embedding-004 |
| **Vector DB** | ChromaDB (persistent, cosine similarity) |
| **Backend** | FastAPI 0.111 + Uvicorn |
| **Frontend** | Streamlit 1.35 |
| **PDF Parsing** | pypdf 4.x |
| **HTTP Client** | httpx |
| **Config** | Pydantic Settings |
| **Containerisation** | Docker + Docker Compose |
| **Language** | Python 3.12 |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
