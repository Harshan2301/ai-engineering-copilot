"""
Page 1 — PDF Document Management
Upload, view, and delete documents.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import upload_document, list_documents, delete_document

st.set_page_config(page_title="Documents · Copilot", page_icon="📄", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📄 Document Manager</h1>
        <p>Upload PDF files to index them in ChromaDB for RAG and semantic search.</p>
    </div>
    """, unsafe_allow_html=True,
)

# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown("### ⬆️ Upload PDF")
uploaded_file = st.file_uploader(
    "Drop a PDF here or click to browse",
    type=["pdf"],
    help="Max file size: 50 MB",
)

if uploaded_file is not None:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.markdown(
            f"""
            <div class="copilot-card" style="margin:0;">
                <div class="doc-name">📎 {uploaded_file.name}</div>
                <div class="doc-meta">{size_mb:.2f} MB · PDF</div>
            </div>
            """, unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner("Parsing PDF, generating embeddings, storing in ChromaDB…"):
                try:
                    result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
                    st.success(
                        f"✅ **{result['filename']}** ingested successfully!\n\n"
                        f"📊 **{result['pages']} pages** · **{result['chunks']} chunks** stored.",
                    )
                    st.balloons()
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Upload failed: {exc}")

st.divider()

# ── Document Library ──────────────────────────────────────────────────────────
st.markdown("### 📚 Document Library")

with st.spinner("Loading documents…"):
    try:
        docs = list_documents()
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        docs = []

if not docs:
    st.info("No documents indexed yet. Upload your first PDF above.", icon="📭")
else:
    st.markdown(f"**{len(docs)} document(s)** in the knowledge base")
    st.markdown("<br>", unsafe_allow_html=True)

    for doc in docs:
        col_icon, col_info, col_actions = st.columns([0.5, 6, 2])

        with col_icon:
            st.markdown(
                '<div style="font-size:2rem; text-align:center; padding-top:0.5rem;">📄</div>',
                unsafe_allow_html=True,
            )

        with col_info:
            st.markdown(
                f"""
                <div style="padding: 0.5rem 0;">
                    <div class="doc-name">{doc.get('filename', 'unknown')}</div>
                    <div class="doc-meta" style="margin-top:4px;">
                        🗂 {doc.get('chunks', 0)} chunks &nbsp;·&nbsp;
                        📖 {doc.get('pages', 0)} pages &nbsp;·&nbsp;
                        🆔 <code style="font-size:0.7rem; color:#8b949e;">{doc.get('doc_id','')[:16]}…</code>
                    </div>
                </div>
                """, unsafe_allow_html=True,
            )

        with col_actions:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "🗑️ Delete",
                key=f"del_{doc.get('doc_id')}",
                use_container_width=True,
            ):
                with st.spinner("Deleting…"):
                    try:
                        delete_document(doc["doc_id"])
                        st.success(f"Deleted **{doc['filename']}**")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

        st.divider()
