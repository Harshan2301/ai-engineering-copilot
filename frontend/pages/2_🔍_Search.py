"""
Page 2 — Semantic Search
Search across document chunks by meaning using vector similarity.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import semantic_search, list_documents

st.set_page_config(page_title="Semantic Search · Copilot", page_icon="🔍", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 Semantic Search</h1>
        <p>Search across all documents by meaning — not just keywords. Results ranked by cosine similarity.</p>
    </div>
    """, unsafe_allow_html=True,
)

# ── Controls ────────────────────────────────────────────────────────────────
col_query, col_opts = st.columns([3, 1])

with col_query:
    query = st.text_input(
        "Search query",
        placeholder="e.g. 'thermal runaway protection in battery management systems'",
        label_visibility="collapsed",
    )

with col_opts:
    top_k = st.selectbox("Results", [5, 8, 10, 15, 20], index=1)

# Document filter
try:
    docs = list_documents()
    doc_options = {"All Documents": None} | {
        d.get("filename", d["doc_id"]): d["doc_id"] for d in docs
    }
except Exception:
    doc_options = {"All Documents": None}

selected_doc_label = st.selectbox(
    "Filter by document",
    list(doc_options.keys()),
)
selected_doc_id = doc_options[selected_doc_label]

search_btn = st.button("🔍 Search", use_container_width=False, type="primary")

# ── Results ────────────────────────────────────────────────────────────────
if search_btn and query.strip():
    with st.spinner("Embedding query and searching vector store…"):
        try:
            result = semantic_search(query.strip(), doc_id=selected_doc_id, top_k=top_k)
            chunks = result.get("results", [])
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            chunks = []

    if not chunks:
        st.warning("No results found. Try a different query or upload more documents.", icon="🔎")
    else:
        st.markdown(
            f"<br>Found **{len(chunks)} results** for: *{query}*",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for i, chunk in enumerate(chunks, 1):
            score = chunk.get("score", 0.0)
            score_pct = int(score * 100)
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page", 0)
            text = chunk.get("text", "")

            st.markdown(
                f"""
                <div class="search-result">
                    <div class="result-header">
                        <span class="filename">📄 {filename}</span>
                        <span class="page-badge">Page {page}</span>
                    </div>
                    <div class="score-bar-wrap">
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width:{score_pct}%;"></div>
                        </div>
                        <span class="score-label">{score:.3f}</span>
                    </div>
                    <div class="result-text">{text[:600]}{"…" if len(text) > 600 else ""}</div>
                </div>
                """, unsafe_allow_html=True,
            )

elif search_btn and not query.strip():
    st.warning("Please enter a search query.", icon="⚠️")

elif not search_btn:
    st.markdown(
        """
        <div class="copilot-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
            <div style="color:#8b949e;">Enter a query above to search your document knowledge base.</div>
        </div>
        """, unsafe_allow_html=True,
    )
