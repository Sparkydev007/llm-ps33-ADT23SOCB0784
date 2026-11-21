"""Streamlit UI for Textbook Doubt Solver.

One chapter per session. Answers come only from the uploaded PDF.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import streamlit as st

from ingest import EmptyPdfError, load_chunks, page_distribution
from index import build_or_load, sha256_file
from prompts import REFUSAL_STRING
from rag import answer, explain_simpler, retrieve

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SAMPLE_PDF = Path("data/ncert_class10_science_ch6_life_processes.pdf")
UPLOAD_DIR = Path("cache/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Textbook Doubt Solver",
    page_icon="📖",
    layout="centered",
)


def init_state() -> None:
    """Create session keys used to cache the indexed chapter and last Q&A."""
    defaults = {
        "chunks": None,
        "index": None,
        "pdf_hash": None,
        "pdf_name": None,
        "last_answer": None,
        "last_question": None,
        "last_retrieved": None,
        "last_result": None,
        "simpler": None,
        "from_cache": False,
        "index_seconds": None,
        "page_dist": None,
        "selected_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Clear the chapter and the last answer (new chapter = new session)."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def save_upload(uploaded) -> Path:
    """Write the uploaded PDF to disk so pypdf and the hash cache can read it."""
    dest = UPLOAD_DIR / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    return dest


def index_pdf(target_path: Path) -> None:
    """Extract pages, generate embeddings, and build/load FAISS cache."""
    with st.spinner("Indexing chapter — extracting pages, embedding, building FAISS..."):
        t0 = time.perf_counter()
        try:
            pdf_hash = sha256_file(target_path)
            chunks = load_chunks(target_path)
            index, chunks, pdf_hash, from_cache = build_or_load(target_path, chunks)
        except EmptyPdfError as exc:
            st.error(str(exc))
            st.stop()
        elapsed = time.perf_counter() - t0

    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.pdf_hash = pdf_hash
    st.session_state.pdf_name = target_path.name
    st.session_state.from_cache = from_cache
    st.session_state.index_seconds = elapsed
    st.session_state.page_dist = page_distribution(chunks)
    st.session_state.last_answer = None
    st.session_state.last_question = None
    st.session_state.last_retrieved = None
    st.session_state.last_result = None
    st.session_state.simpler = None

    if from_cache:
        st.success(
            f"⚡ Cache hit for this PDF (SHA-256 `{pdf_hash[:10]}…`). "
            f"Loaded {len(chunks)} chunks instantly in {elapsed:.2f}s."
        )
    else:
        st.success(
            f"✅ Indexed {len(chunks)} chunks from {target_path.name} in {elapsed:.2f}s."
        )


def render_answer_card(text: str) -> None:
    """Show the three required sections with clean visual badges."""
    stripped = text.strip()
    if stripped == REFUSAL_STRING:
        st.warning(
            f"🛡️ **Out-of-Scope Chapter Refusal:**\n\n"
            f"> *\"{stripped}\"*\n\n"
            f"*(This question asks for facts not covered in this chapter; the model refused cleanly.)*"
        )
        return

    short = _section(text, "Short answer")
    simple = _section(text, "In simple words")
    textbook = _section(text, "From the textbook")

    if short or simple or textbook:
        if short:
            st.markdown(
                f"""<div style="background-color: #f0f7ff; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #0066cc;">
                <strong style="color: #004c99; font-size: 15px;">📌 Short Answer:</strong><br>
                <span style="font-size: 14.5px; line-height: 1.5;">{short}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        if simple:
            st.markdown(
                f"""<div style="background-color: #f2fbf4; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #2e7d32;">
                <strong style="color: #1b5e20; font-size: 15px;">🎈 In Simple Words:</strong><br>
                <span style="font-size: 14.5px; line-height: 1.5;">{simple}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        if textbook:
            st.markdown(
                f"""<div style="background-color: #fffdf2; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #f57c00;">
                <strong style="color: #e65100; font-size: 15px;">📖 From the Textbook (Verbatim Quote):</strong><br>
                <em style="font-size: 14.5px; line-height: 1.5;">"{textbook}"</em>
                </div>""",
                unsafe_allow_html=True,
            )
        leftover = _leftover(text)
        if leftover:
            st.info(leftover)
    else:
        st.write(text)


def _section(text: str, title: str) -> str | None:
    pattern = rf"\*\*{title}:\*\*\s*(.+?)(?=\n\s*\*\*|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _leftover(text: str) -> str | None:
    extra = re.sub(
        r"\*\*Short answer:\*\*.*?(?=\n\s*\*\*|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    extra = re.sub(
        r"\*\*In simple words:\*\*.*?(?=\n\s*\*\*|$)",
        "",
        extra,
        flags=re.IGNORECASE | re.DOTALL,
    )
    extra = re.sub(
        r"\*\*From the textbook:\*\*.*",
        "",
        extra,
        flags=re.IGNORECASE | re.DOTALL,
    )
    extra = extra.strip()
    return extra or None


init_state()

st.title("📖 Textbook Doubt Solver")
st.caption(
    "**Prathmesh Bunde** · LY Core 4 · Roll 33 · ADT23SOCB0784 — answers ONLY from the active chapter."
)

with st.sidebar:
    st.header("📚 Chapter Source")
    uploaded = st.file_uploader("Upload ONE chapter (PDF)", type=["pdf"])
    use_sample = st.checkbox(
        "Use bundled sample: NCERT Class 10 'Life Processes'",
        value=SAMPLE_PDF.exists() and uploaded is None,
    )
    index_clicked = st.button("🚀 Index Chapter", type="primary", use_container_width=True)
    if st.button("🔄 Clear / Reset Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown(
        "1. **Upload or Select** a textbook chapter\n"
        "2. Click **Index Chapter** (instant cache hit on re-upload)\n"
        "3. **Ask doubts** or click any quick-test prompt\n"
        "4. Check the **page-cited verbatim quote**\n"
        "5. Click **Explain simpler** for a 12-year-old analogy"
    )

    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        st.error("⚠️ GROQ_API_KEY not found in `.env` file.")

# Determine target PDF path
pdf_path: Path | None = None
if uploaded is not None:
    pdf_path = save_upload(uploaded)
elif use_sample and SAMPLE_PDF.exists():
    pdf_path = SAMPLE_PDF

# Handle Index action
if index_clicked:
    if pdf_path is None:
        st.warning("Please upload a PDF or select the sample chapter first.")
    else:
        index_pdf(pdf_path)

# If no chapter is indexed yet, display a welcoming quick-start card
if st.session_state.chunks is None:
    st.markdown("---")
    st.markdown(
        """
        ### Welcome to the Grounded Textbook Tutor!
        This RAG assistant guarantees that answers come **strictly from your textbook chapter**, 
        providing simple explanations along with **verbatim quotes and exact page citations**.
        """
    )
    if SAMPLE_PDF.exists():
        st.info("💡 You can start immediately with the bundled NCERT Class 10 sample chapter.")
        if st.button("⚡ 1-Click Quick Start (Load Sample Chapter)", type="primary", use_container_width=True):
            index_pdf(SAMPLE_PDF)
            st.rerun()
    else:
        st.info("👈 Upload a textbook chapter PDF in the sidebar to get started.")
    st.stop()

# Display active chapter metadata badge
st.markdown(
    f"📘 **Active Chapter:** `{st.session_state.pdf_name}` · "
    f"**{len(st.session_state.chunks)} chunks** · "
    f"SHA-256: `{st.session_state.pdf_hash[:10]}…`"
)
if st.session_state.from_cache:
    st.caption("⚡ Loaded from SHA-256 disk cache (zero re-embedding latency).")

st.markdown("---")

# Quick-ask Viva chips for ease of use
st.markdown("**💡 Quick Doubts (Click to auto-fill):**")
chip1, chip2, chip3, chip4 = st.columns(4)
if chip1.button("🌿 Transport (T1)", use_container_width=True):
    st.session_state["selected_question"] = "Why do plants need a transport system?"
    st.rerun()
if chip2.button("🫀 Heart Pump (T4)", use_container_width=True):
    st.session_state["selected_question"] = "How does the heart pump blood?"
    st.rerun()
if chip3.button("🍃 Stomata (T2)", use_container_width=True):
    st.session_state["selected_question"] = "What is the role of stomata?"
    st.rerun()
if chip4.button("🚫 Chlorophyll (T7)", use_container_width=True):
    st.session_state["selected_question"] = "What is the chemical formula of chlorophyll?"
    st.rerun()

current_val = st.session_state.get("selected_question", "")
question = st.text_area(
    "Your doubt:",
    value=current_val,
    placeholder="e.g. Why do plants need a transport system?",
    height=85,
)
ask_disabled = not question or not question.strip()

col_a, col_b = st.columns([1, 1])
with col_a:
    ask = st.button("🔍 Ask Doubt", type="primary", disabled=ask_disabled, use_container_width=True)
with col_b:
    simpler_btn = st.button(
        "✨ Explain Simpler",
        disabled=st.session_state.last_answer is None,
        use_container_width=True,
    )

if ask_disabled and not question:
    st.caption("Type a question above or click a quick doubt to ask.")

# Execute Question
if ask and question.strip():
    st.session_state["selected_question"] = question.strip()
    st.session_state.simpler = None
    with st.spinner("Searching chapter and generating grounded answer..."):
        retrieved = retrieve(
            question.strip(),
            st.session_state.index,
            st.session_state.chunks,
            k=4,
        )
        result = answer(question.strip(), retrieved)

    st.session_state.last_question = question.strip()
    st.session_state.last_retrieved = retrieved
    st.session_state.last_result = result
    st.session_state.last_answer = result["answer"]

# Execute Explain Simpler
if simpler_btn and st.session_state.last_answer and st.session_state.last_retrieved:
    with st.spinner("Simplifying answer using the SAME retrieved chunks..."):
        simpler = explain_simpler(
            st.session_state.last_question,
            st.session_state.last_answer,
            st.session_state.last_retrieved,
        )
    st.session_state.simpler = simpler

# Render Main Answer Card
if st.session_state.last_answer:
    st.markdown("---")
    st.subheader("💬 Tutor Answer")
    result = st.session_state.last_result or {}
    render_answer_card(st.session_state.last_answer)

    # Metrics row
    meta_cols = st.columns(3)
    meta_cols[0].metric("⚡ Latency", f"{result.get('latency_s', 0):.2f}s")
    is_grounded = result.get("verified", False)
    meta_cols[1].metric("✅ Quote Grounded", "Verified (≥0.85)" if is_grounded else "Refusal / Fallback")
    path_name = "Labelled Fallback" if result.get("fallback") else ("Retried Once" if result.get("retried") else "Direct Match")
    meta_cols[2].metric("🎯 Verification Path", path_name)

    # Source expander
    retrieved = st.session_state.last_retrieved or []
    with st.expander(f"📑 View {len(retrieved)} Retrieved Source Chunks with Page Numbers"):
        if not retrieved:
            st.write("No chunks retrieved.")
        for i, item in enumerate(retrieved, start=1):
            st.markdown(
                f"**Chunk {i} · Page {item['page']} · Distance Score: `{item['score']:.4f}`**"
            )
            st.code(item["text"], language="text")

# Render Explain Simpler Card
if st.session_state.simpler:
    st.markdown("---")
    st.subheader("✨ Simplified Explanation (Same Chunks, No New Facts)")
    render_answer_card(st.session_state.simpler["answer"])
    extra = st.session_state.simpler.get("new_facts") or []
    if extra:
        st.warning("⚠️ Flagged terms/numbers in simplified answer: " + ", ".join(extra))
    else:
        st.success("✅ Fact-Check Passed: No extraneous facts or numbers introduced.")
    st.caption(f"Latency: {st.session_state.simpler.get('latency_s', 0):.2f}s")
