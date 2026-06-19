"""
AI Engineering Copilot — Streamlit main entry point.
Home page: health status dashboard + overview.
"""
import os
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Engineering Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject dark-mode CSS ────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.api_client import health_check, list_documents

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            <div style="font-size:2.5rem">🤖</div>
            <div style="font-size:1.1rem; font-weight:700; color:#a78bfa;">Engineering Copilot</div>
            <div style="font-size:0.75rem; color:#8b949e; margin-top:4px;">Powered by Gemini + ChromaDB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 📋 Navigation")
    st.page_link("app.py",                        label="🏠 Dashboard",          icon=None)
    st.page_link("pages/1_📄_Documents.py",       label="📄 Documents",          icon=None)
    st.page_link("pages/2_🔍_Search.py",          label="🔍 Semantic Search",    icon=None)
    st.page_link("pages/3_💬_Ask.py",             label="💬 Ask (RAG)",          icon=None)
    st.page_link("pages/4_📝_Summarize.py",       label="📝 Summarize",          icon=None)
    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem;color:#484f58;text-align:center;">v1.0.0 · MIT License</div>',
        unsafe_allow_html=True,
    )

# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 AI Engineering Copilot</h1>
        <p>Upload PDFs → Ask questions → Semantic search → Summarize — all powered by Gemini + ChromaDB.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Health Status ─────────────────────────────────────────────────────────────
with st.spinner("Checking backend health…"):
    health = health_check()

status = health.get("status", "unreachable")
badge_class = "healthy" if status == "healthy" else ("degraded" if status == "degraded" else "error")
badge_icon  = "✅" if status == "healthy" else ("⚠️" if status == "degraded" else "❌")

st.markdown(
    f'<span class="status-badge {badge_class}">{badge_icon} Backend: {status.upper()}</span>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

# Fetch document count
try:
    docs = list_documents()
    doc_count = len(docs)
except Exception:
    docs, doc_count = [], 0

chroma_info = health.get("chromadb", {})
gemini_info = health.get("gemini", {})

with col1:
    st.markdown(
        f"""
        <div class="stat-tile">
            <div class="stat-value">{doc_count}</div>
            <div class="stat-label">Documents Indexed</div>
        </div>
        """, unsafe_allow_html=True,
    )

with col2:
    chunk_count = chroma_info.get("total_chunks", "–")
    st.markdown(
        f"""
        <div class="stat-tile">
            <div class="stat-value">{chunk_count}</div>
            <div class="stat-label">Vector Chunks</div>
        </div>
        """, unsafe_allow_html=True,
    )

with col3:
    chroma_status = "🟢 OK" if chroma_info.get("ok") else "🔴 Down"
    st.markdown(
        f"""
        <div class="stat-tile">
            <div class="stat-value" style="font-size:1.4rem;">{chroma_status}</div>
            <div class="stat-label">ChromaDB</div>
        </div>
        """, unsafe_allow_html=True,
    )

with col4:
    gemini_status = "🟢 OK" if gemini_info.get("ok") else "🔴 Down"
    st.markdown(
        f"""
        <div class="stat-tile">
            <div class="stat-value" style="font-size:1.4rem;">{gemini_status}</div>
            <div class="stat-label">Gemini API</div>
        </div>
        """, unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Cards ─────────────────────────────────────────────────────────────
st.markdown("### 🚀 Features")

fc1, fc2 = st.columns(2)
with fc1:
    st.markdown(
        """
        <div class="copilot-card">
            <h4 style="color:#a78bfa; margin:0 0 0.5rem;">📄 PDF Ingestion</h4>
            <p style="color:#8b949e; font-size:0.88rem; margin:0;">
                Upload engineering PDFs. Text is extracted, chunked, embedded with
                Gemini, and stored in ChromaDB for instant retrieval.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="copilot-card">
            <h4 style="color:#58a6ff; margin:0 0 0.5rem;">💬 RAG Question Answering</h4>
            <p style="color:#8b949e; font-size:0.88rem; margin:0;">
                Ask natural-language questions. The copilot retrieves relevant chunks
                and generates grounded, cited answers using Gemini.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )

with fc2:
    st.markdown(
        """
        <div class="copilot-card">
            <h4 style="color:#3fb950; margin:0 0 0.5rem;">🔍 Semantic Search</h4>
            <p style="color:#8b949e; font-size:0.88rem; margin:0;">
                Search across all documents by meaning, not just keywords.
                Results are ranked by cosine similarity with relevance scores.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="copilot-card">
            <h4 style="color:#d29922; margin:0 0 0.5rem;">📝 Document Summarization</h4>
            <p style="color:#8b949e; font-size:0.88rem; margin:0;">
                Generate technical, executive, or bullet-point summaries of any
                uploaded document with a single click.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )

# ── Recent Documents ──────────────────────────────────────────────────────────
if docs:
    st.markdown("### 📂 Recent Documents")
    for doc in docs[:5]:
        st.markdown(
            f"""
            <div class="doc-row">
                <div>
                    <div class="doc-name">📄 {doc.get('filename','unknown')}</div>
                    <div class="doc-meta">{doc.get('chunks',0)} chunks · {doc.get('pages',0)} pages</div>
                </div>
                <div class="doc-meta">{doc.get('uploaded_at','')[:10]}</div>
            </div>
            """, unsafe_allow_html=True,
        )
else:
    st.info("No documents yet. Go to **📄 Documents** to upload your first PDF.", icon="💡")
