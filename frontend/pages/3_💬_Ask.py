"""
Page 3 — RAG Question Answering (Ask)
Chat-style interface backed by Retrieval-Augmented Generation.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import ask_question, list_documents

st.set_page_config(page_title="Ask · Copilot", page_icon="💬", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state for chat history ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, sources}

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>💬 Ask the Copilot</h1>
        <p>Ask engineering questions. The copilot retrieves relevant context and generates grounded, cited answers.</p>
    </div>
    """, unsafe_allow_html=True,
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ RAG Settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=20, value=5)

    try:
        docs = list_documents()
        doc_options = {"All Documents": None} | {
            d.get("filename", d["doc_id"]): d["doc_id"] for d in docs
        }
    except Exception:
        doc_options = {"All Documents": None}

    selected_doc_label = st.selectbox("Filter by document", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_doc_label]

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat history display ─────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                f'<div class="answer-box">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Show sources
            sources = msg.get("sources", [])
            if sources:
                chips_html = ""
                for s in sources:
                    chips_html += (
                        f'<span class="source-chip">'
                        f'📄 {s.get("filename","?")} · p{s.get("page","?")}'
                        f'<span class="score">{s.get("score",0):.2f}</span>'
                        f'</span>'
                    )
                st.markdown(
                    f'<div style="margin-top:0.75rem;">{chips_html}</div>',
                    unsafe_allow_html=True,
                )

# ── Input ────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask an engineering question…")

if question:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving context and generating answer…"):
            try:
                result = ask_question(
                    question,
                    doc_id=selected_doc_id,
                    top_k=top_k,
                )
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                model = result.get("model", "gemini")

                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True,
                )

                if sources:
                    chips_html = ""
                    for s in sources:
                        chips_html += (
                            f'<span class="source-chip">'
                            f'📄 {s.get("filename","?")} · p{s.get("page","?")}'
                            f'<span class="score">{s.get("score",0):.2f}</span>'
                            f'</span>'
                        )
                    st.markdown(
                        f'<div style="margin-top:0.75rem;">'
                        f'<span style="font-size:0.78rem;color:#8b949e;">Sources: </span>'
                        f'{chips_html}</div>',
                        unsafe_allow_html=True,
                    )

                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                err_msg = f"❌ Error: {exc}"
                st.error(err_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": err_msg,
                    "sources": [],
                })
