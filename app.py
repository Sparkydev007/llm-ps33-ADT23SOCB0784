"""Streamlit UI for Textbook Doubt Solver.

One chapter per session. Answers come only from the uploaded PDF.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from ingest import EmptyPdfError, load_chunks, page_distribution
from index import build_or_load, sha256_file
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


def render_answer_card(text: str) -> None:
    """Show the three required sections when the model followed the format."""
    short = _section(text, "Short answer")
    simple = _section(text, "In simple words")
    textbook = _section(text, "From the textbook")

    if short or simple or textbook:
        if short:
            st.markdown("**Short answer:**")
            st.write(short)
        if simple:
            st.markdown("**In simple words:**")
            st.write(simple)
        if textbook:
            st.markdown("**From the textbook:**")
            st.write(textbook)
        leftover = _leftover(text)
        if leftover:
            st.info(leftover)
    else:
        st.write(text)


def _section(text: str, title: str) -> str | None:
    import re

    pattern = rf"\*\*{title}:\*\*\s*(.+?)(?=\n\s*\*\*|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _leftover(text: str) -> str | None:
    import re

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
    "Prathmesh Bunde · LY Core 4 · Roll 33 — answers only from ONE uploaded chapter."
)

with st.sidebar:
    st.header("Chapter")
    uploaded = st.file_uploader("Upload ONE textbook chapter (PDF)", type=["pdf"])
    use_sample = st.checkbox(
        "Use sample: NCERT Class 10 Science Ch. 6 Life Processes",
        value=SAMPLE_PDF.exists() and uploaded is None,
    )
    index_clicked = st.button("Index chapter", type="primary", use_container_width=True)
    if st.button("Clear / reset session", use_container_width=True):
        reset_session()
        st.rerun()
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload one chapter PDF\n"
        "2. Click **Index chapter**\n"
        "3. Ask a doubt\n"
        "4. Check the page-cited quote\n"
        "5. Optional: **Explain simpler**"
    )
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        st.error("GROQ_API_KEY is not set. Add it to a `.env` file.")

pdf_path: Path | None = None
if uploaded is not None:
    pdf_path = save_upload(uploaded)
elif use_sample and SAMPLE_PDF.exists():
    pdf_path = SAMPLE_PDF

if index_clicked:
    if pdf_path is None:
        st.warning("Upload a PDF (or enable the sample chapter) first.")
    else:
        with st.spinner("Indexing chapter — extracting pages, embedding, building FAISS..."):
            t0 = time.perf_counter()
            try:
                pdf_hash = sha256_file(pdf_path)
                chunks = load_chunks(pdf_path)
                index, chunks, pdf_hash, from_cache = build_or_load(pdf_path, chunks)
            except EmptyPdfError as exc:
                st.error(str(exc))
                st.stop()
            elapsed = time.perf_counter() - t0
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.pdf_hash = pdf_hash
        st.session_state.pdf_name = pdf_path.name
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
                f"Cache hit for this PDF (SHA-256 {pdf_hash[:10]}…). "
                f"Loaded {len(chunks)} chunks in {elapsed:.2f}s."
            )
        else:
            st.success(
                f"Indexed {len(chunks)} chunks from {pdf_path.name} in {elapsed:.2f}s."
            )

if st.session_state.chunks is None:
    st.info("Upload a chapter and click **Index chapter** to start asking doubts.")
    st.stop()

st.markdown(
    f"Indexed **{st.session_state.pdf_name}** · "
    f"{len(st.session_state.chunks)} chunks · "
    f"hash `{st.session_state.pdf_hash[:12]}…`"
)
if st.session_state.from_cache:
    st.caption("Loaded from SHA-256 cache (re-upload was instant).")

question = st.text_area("Your doubt", placeholder="e.g. Why do plants need a transport system?", height=90)
ask_disabled = not question or not question.strip()

col_a, col_b = st.columns(2)
with col_a:
    ask = st.button("Ask", type="primary", disabled=ask_disabled, use_container_width=True)
with col_b:
    simpler_btn = st.button(
        "Explain simpler",
        disabled=st.session_state.last_answer is None,
        use_container_width=True,
    )

if ask_disabled:
    st.caption("Type a question to enable Ask. Empty or whitespace is ignored.")

if ask and question.strip():
    st.session_state.simpler = None
    with st.spinner("Retrieving from this chapter and asking the tutor..."):
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

if simpler_btn and st.session_state.last_answer and st.session_state.last_retrieved:
    with st.spinner("Re-explaining with the same retrieved chunks..."):
        simpler = explain_simpler(
            st.session_state.last_question,
            st.session_state.last_answer,
            st.session_state.last_retrieved,
        )
    st.session_state.simpler = simpler

if st.session_state.last_answer:
    st.markdown("---")
    st.subheader("Answer")
    result = st.session_state.last_result or {}
    render_answer_card(st.session_state.last_answer)

    meta_cols = st.columns(3)
    meta_cols[0].metric("Latency", f"{result.get('latency_s', 0):.2f}s")
    meta_cols[1].metric("Quote grounded", "Yes" if result.get("verified") else "No")
    meta_cols[2].metric(
        "Path",
        "fallback" if result.get("fallback") else ("retry" if result.get("retried") else "first try"),
    )

    retrieved = st.session_state.last_retrieved or []
    with st.expander("Source chunks (retrieved from this chapter)"):
        if not retrieved:
            st.write("No chunks retrieved.")
        for i, item in enumerate(retrieved, start=1):
            st.markdown(
                f"**Chunk {i} · page {item['page']} · L2 score {item['score']:.4f}**"
            )
            st.write(item["text"])

if st.session_state.simpler:
    st.markdown("---")
    st.subheader("Explain simpler (same chunks, no new facts)")
    render_answer_card(st.session_state.simpler["answer"])
    extra = st.session_state.simpler.get("new_facts") or []
    if extra:
        st.warning("Possible new facts in the simpler answer: " + ", ".join(extra))
    else:
        st.caption("No extra numbers or named entities were added.")
    st.caption(f"Latency {st.session_state.simpler.get('latency_s', 0):.2f}s")
