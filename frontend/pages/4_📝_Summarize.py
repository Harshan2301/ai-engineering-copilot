"""
Page 4 — Document Summarization
Generate AI-powered technical, executive, or bullet-point summaries.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import summarize_document, list_documents

st.set_page_config(page_title="Summarize · Copilot", page_icon="📝", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📝 Document Summarizer</h1>
        <p>Generate AI-powered summaries of any uploaded document. Choose from technical, executive, or bullet-point styles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load documents ────────────────────────────────────────────────────────────
try:
    docs = list_documents()
except Exception as exc:
    st.error(f"Failed to load documents: {exc}")
    docs = []

if not docs:
    st.info(
        "No documents found. Please upload PDFs on the **📄 Documents** page first.",
        icon="📭",
    )
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
col_doc, col_style, col_btn = st.columns([3, 2, 1])

with col_doc:
    doc_options = {d.get("filename", d["doc_id"]): d["doc_id"] for d in docs}
    selected_filename = st.selectbox(
        "Select document",
        list(doc_options.keys()),
        help="Choose the document you want to summarize.",
    )
    selected_doc_id = doc_options[selected_filename]

with col_style:
    style = st.selectbox(
        "Summary style",
        ["technical", "executive", "bullet"],
        format_func=lambda s: {
            "technical": "🔧 Technical",
            "executive": "📊 Executive",
            "bullet":    "• Bullet Points",
        }[s],
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("✨ Generate", use_container_width=True, type="primary")

# ── Style description ─────────────────────────────────────────────────────────
style_descriptions = {
    "technical":  "Covers architecture, algorithms, data flows, APIs, and implementation details.",
    "executive":  "3-5 key bullet points covering purpose, findings, risks, and recommendations.",
    "bullet":     "Structured bullet-point list grouped by topic — comprehensive yet concise.",
}
st.markdown(
    f'<div style="font-size:0.83rem; color:#8b949e; margin-bottom:1.5rem;">'
    f'ℹ️ {style_descriptions[style]}</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Summary output ────────────────────────────────────────────────────────────
if generate_btn:
    with st.spinner(f"Reading all chunks and generating {style} summary with Gemini…"):
        try:
            result = summarize_document(selected_doc_id, style=style)

            filename    = result.get("filename", selected_filename)
            summary     = result.get("summary", "")
            word_count  = result.get("word_count", 0)
            style_out   = result.get("style", style)

            # Meta badges
            style_labels = {"technical": "🔧 Technical", "executive": "📊 Executive", "bullet": "• Bullet"}
            st.markdown(
                f"""
                <div class="summary-box">
                    <div class="summary-meta">
                        <span class="meta-badge">📄 {filename}</span>
                        <span class="meta-badge">{style_labels.get(style_out, style_out)}</span>
                        <span class="meta-badge">📝 {word_count} words</span>
                    </div>
                    <div class="summary-content">{summary}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Download button
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Summary (.txt)",
                data=f"Document: {filename}\nStyle: {style_out}\nWords: {word_count}\n\n{summary}",
                file_name=f"{Path(filename).stem}_{style_out}_summary.txt",
                mime="text/plain",
            )

        except Exception as exc:
            st.error(f"❌ Summarization failed: {exc}", icon="🚨")
else:
    # Empty state
    st.markdown(
        """
        <div class="copilot-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3.5rem; margin-bottom:1rem;">📝</div>
            <div style="color:#8b949e; font-size:0.95rem;">
                Select a document and style above, then click <strong style="color:#a78bfa;">✨ Generate</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
