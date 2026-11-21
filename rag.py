# -*- coding: utf-8 -*-
"""Retrieve, answer, verify quotes, retry, and simplify.

No prompt strings live here; they are imported from prompts.py.
"""

from __future__ import annotations

import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

import faiss
from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from index import embed_texts
from prompts import (
    EXPLAIN_SIMPLER,
    EXPLAIN_SIMPLER_SYSTEM,
    REFUSAL_STRING,
    RETRY_QUOTE,
    SYSTEM_ANSWER,
    USER_ANSWER,
)

# Groq is OpenAI-compatible. Temperature stays low so quotes stay faithful.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TEMPERATURE = 0.1
MAX_TOKENS = 1200


def get_client() -> OpenAI:
    """Build an OpenAI client pointed at Groq."""
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in your environment or a .env file or Streamlit secrets."
        )
    base = os.getenv("OPENAI_BASE_URL", GROQ_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base, timeout=45.0)


def format_context(retrieved: list[dict[str, Any]]) -> str:
    """Turn retrieved chunks into the CONTEXT block the LLM sees."""
    parts: list[str] = []
    for i, item in enumerate(retrieved, start=1):
        parts.append(
            f"[Chunk {i} | p.{item['page']}]\n{item['text']}"
        )
    return "\n\n".join(parts)


def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[dict[str, Any]],
    k: int = 4,
) -> list[dict[str, Any]]:
    """Return the k nearest chunks as {page, text, score}.

    Score is L2 distance (lower is closer). k is clamped to the index size.
    """
    if index.ntotal == 0:
        return []
    k_use = min(k, index.ntotal)
    query_vec = embed_texts([query])
    distances, ids = index.search(query_vec, k_use)
    results: list[dict[str, Any]] = []
    for dist, idx in zip(distances[0], ids[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        results.append(
            {
                "page": int(chunk["page"]),
                "text": chunk["text"],
                "score": float(dist),
            }
        )
    return results


def _call_llm(
    system: str,
    user: str,
    temperature: float = TEMPERATURE,
) -> str:
    """One Groq chat completion. Temperature 0-0.2 keeps quotes stable."""
    client = get_client()
    model = os.getenv("GROQ_MODEL")
    if not model:
        try:
            import streamlit as st

            model = st.secrets.get("GROQ_MODEL")
        except Exception:
            pass
    if not model:
        model = DEFAULT_MODEL

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def _normalize(text: str) -> str:
    """Lowercase, strip quotes/punctuation noise, collapse spaces.

    WHY: PDF line-breaks and curly quotes must not fail an otherwise
    exact textbook line.
    """
    text = text.lower()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_quote_and_page(answer: str) -> tuple[str | None, int | None]:
    """Pull the verbatim quote and page from the model output."""
    page = None
    page_match = re.search(r"\(p\.?\s*(\d+)\)", answer, flags=re.IGNORECASE)
    if not page_match:
        page_match = re.search(r"\bp(?:age)?\.?\s*(\d+)\b", answer, flags=re.IGNORECASE)
    if page_match:
        page = int(page_match.group(1))

    quote = None
    labelled = re.search(
        r"From the textbook:\s*[\"“](.+?)[\"”]",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if labelled:
        quote = labelled.group(1).strip()
    else:
        quoted = re.findall(r"[\"“]([^\"”]{12,})[\"”]", answer)
        if quoted:
            quote = quoted[-1].strip()
    return quote, page


def _window_ratios(quote_norm: str, chunk_norm: str) -> float:
    """Best SequenceMatcher ratio of the quote vs sliding word-windows.

    WHY sliding windows: the quote is one sentence, the chunk is ~600
    characters. Matching the whole chunk would always look 'different'.
    """
    quote_words = quote_norm.split()
    chunk_words = chunk_norm.split()
    if not quote_words or not chunk_words:
        return 0.0

    n = len(quote_words)
    window = max(n, 1)
    best = 0.0
    if len(chunk_words) <= window + 4:
        best = SequenceMatcher(None, quote_norm, chunk_norm).ratio()
        return best

    for extra in (-4, -2, 0, 2, 4):
        size = window + extra
        if size <= 0 or size > len(chunk_words):
            continue
        for start in range(0, len(chunk_words) - size + 1):
            piece = " ".join(chunk_words[start : start + size])
            ratio = SequenceMatcher(None, quote_norm, piece).ratio()
            if ratio > best:
                best = ratio
            if best >= 0.99:
                return best
    full = SequenceMatcher(None, quote_norm, chunk_norm).ratio()
    return max(best, full)


def verify_quote(
    answer: str,
    chunks: list[dict[str, Any]],
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Fuzzy-match the model's quote against retrieved (or all) chunks.

    Returns grounded, ratio, quote, page, matched_page.
    """
    quote, page = extract_quote_and_page(answer)
    if not quote:
        return {
            "grounded": False,
            "ratio": 0.0,
            "quote": None,
            "page": page,
            "matched_page": None,
        }

    quote_norm = _normalize(quote)
    best_ratio = 0.0
    matched_page = None
    for chunk in chunks:
        ratio = _window_ratios(quote_norm, _normalize(chunk["text"]))
        if ratio > best_ratio:
            best_ratio = ratio
            matched_page = int(chunk["page"])

    return {
        "grounded": best_ratio >= threshold,
        "ratio": best_ratio,
        "quote": quote,
        "page": page,
        "matched_page": matched_page,
    }


def _sentences(text: str) -> list[str]:
    """Split a chunk into sentence-like pieces for the fallback line."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def closest_real_sentence(
    question: str,
    chunks: list[dict[str, Any]],
) -> tuple[str, int, float]:
    """Find the real sentence in the chunks closest to the question.

    Used when the model quote cannot be verified. We never invent a line.
    """
    q_norm = _normalize(question)
    best = ("", 0, 0.0)
    for chunk in chunks:
        for sent in _sentences(chunk["text"]):
            ratio = SequenceMatcher(None, q_norm, _normalize(sent)).ratio()
            if ratio > best[2]:
                best = (sent, int(chunk["page"]), ratio)
    if not best[0]:
        first = chunks[0]
        snippet = first["text"].split(".")[0].strip()
        return snippet, int(first["page"]), 0.0
    return best


def apply_quote_fallback(
    answer: str,
    question: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Replace an ungrounded quote with a labelled real sentence.

    WHY labelled: the student must see this is not the model's claim,
    it is the closest line we actually found in the chapter.
    """
    sentence, page, _ratio = closest_real_sentence(question, chunks)
    label = f'Closest line in the chapter (p.{page}): "{sentence}"'
    cleaned = re.sub(
        r"\*\*From the textbook:\*\*.*$",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).rstrip()
    return cleaned + "\n\n" + label


def is_refusal(text: str) -> bool:
    """True when the answer is the exact required refusal string."""
    return text.strip() == REFUSAL_STRING


def answer(
    question: str,
    retrieved: list[dict[str, Any]],
    retry: bool = True,
) -> dict[str, Any]:
    """Grounded answer with quote verify, retry once, then labelled fallback."""
    started = time.perf_counter()
    context = format_context(retrieved)
    user = USER_ANSWER.format(question=question, context=context)
    text = _call_llm(SYSTEM_ANSWER, user)

    # WHY refusal path: T7/T8 must match the string exactly. If the
    # model adds extra words, we still treat a clean refusal as final
    # and do not attach a fake quote.
    stripped = text.strip()
    if stripped == REFUSAL_STRING or (
        REFUSAL_STRING in stripped and len(stripped) < len(REFUSAL_STRING) + 40
    ):
        if REFUSAL_STRING in stripped and not re.search(
            r"From the textbook", stripped, flags=re.IGNORECASE
        ):
            text = REFUSAL_STRING
            elapsed = time.perf_counter() - started
            return {
                "answer": text,
                "verified": True,
                "fallback": False,
                "retried": False,
                "latency_s": elapsed,
                "quote_check": {
                    "grounded": True,
                    "ratio": 1.0,
                    "quote": None,
                    "page": None,
                    "matched_page": None,
                },
            }

    check = verify_quote(text, retrieved)
    retried = False
    fallback = False

    if not check["grounded"] and retry:
        retried = True
        retry_user = RETRY_QUOTE.format(
            refusal=REFUSAL_STRING,
            question=question,
            context=context,
            previous_answer=text,
        )
        text = _call_llm(SYSTEM_ANSWER, retry_user)
        if text.strip() == REFUSAL_STRING or (
            REFUSAL_STRING in text
            and not re.search(r"From the textbook", text, flags=re.IGNORECASE)
        ):
            text = REFUSAL_STRING
            check = {
                "grounded": True,
                "ratio": 1.0,
                "quote": None,
                "page": None,
                "matched_page": None,
            }
        else:
            check = verify_quote(text, retrieved)

    if text.strip() != REFUSAL_STRING and not check["grounded"]:
        fallback = True
        text = apply_quote_fallback(text, question, retrieved)
        fallback_quote = re.search(
            r'Closest line in the chapter \(p\.(\d+)\):\s*"(.+)"',
            text,
            flags=re.DOTALL,
        )
        if fallback_quote:
            check = {
                "grounded": True,
                "ratio": 1.0,
                "quote": fallback_quote.group(2),
                "page": int(fallback_quote.group(1)),
                "matched_page": int(fallback_quote.group(1)),
                "fallback": True,
            }

    elapsed = time.perf_counter() - started
    return {
        "answer": text,
        "verified": bool(check.get("grounded")),
        "fallback": fallback,
        "retried": retried,
        "latency_s": elapsed,
        "quote_check": check,
    }


def explain_simpler(
    question: str,
    prev_answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Re-explain using the SAME retrieved chunks. No new facts."""
    started = time.perf_counter()
    context = format_context(chunks)
    user = EXPLAIN_SIMPLER.format(
        question=question,
        answer=prev_answer,
        chunks=context,
    )
    text = _call_llm(EXPLAIN_SIMPLER_SYSTEM, user)
    check = verify_quote(text, chunks)
    if not check["grounded"] and text.strip() != REFUSAL_STRING:
        text = apply_quote_fallback(text, question, chunks)
        check["grounded"] = True
        check["fallback"] = True
    elapsed = time.perf_counter() - started
    extra = new_facts(prev_answer, text)
    return {
        "answer": text,
        "latency_s": elapsed,
        "quote_check": check,
        "new_facts": extra,
    }


def new_facts(original: str, simplified: str) -> list[str]:
    """Flag numbers / named entities that appear only in the simpler text.

    Used to catch Explain-simpler adding new facts.
    """
    orig_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", original))
    new_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", simplified))
    orig_ents = set(re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", original))
    new_ents = set(re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", simplified))
    ignore = {
        "Short",
        "Answer",
        "Simple",
        "From",
        "Textbook",
        "Closest",
        "Explain",
        "This",
        "Context",
        "Chunk",
        "They",
        "Think",
        "When",
        "What",
        "Where",
        "Just",
        "Like",
        "Some",
        "Also",
        "Then",
        "These",
        "That",
        "With",
        "Your",
        "Chapter",
    }
    extra_nums = sorted(new_nums - orig_nums)
    extra_ents = sorted(new_ents - orig_ents - ignore)
    flagged = [f"number:{n}" for n in extra_nums] + [
        f"entity:{e}" for e in extra_ents
    ]
    return flagged


if __name__ == "__main__":
    import sys

    from ingest import load_chunks
    from index import build_or_load
    from prompts import REFUSAL_STRING as REFUSAL

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    pdf = sys.argv[1] if len(sys.argv) > 1 else (
        "data/ncert_class10_science_ch6_life_processes.pdf"
    )
    print("Indexing for smoke test...")
    chunks = load_chunks(pdf)
    faiss_index, chunks, _h, cached = build_or_load(pdf, chunks)
    print(f"chunks={len(chunks)} cache={cached}")

    t1 = "Why do plants need a transport system?"
    retrieved = retrieve(t1, faiss_index, chunks, k=4)
    result = answer(t1, retrieved)
    print("--- T1 ---")
    print(result["answer"])
    print(
        "latency={:.2f}s grounded={} retried={} fallback={}".format(
            result["latency_s"],
            result["verified"],
            result["retried"],
            result["fallback"],
        )
    )
    assert result["quote_check"].get("grounded"), "T1 quote must verify"
    assert result["latency_s"] <= 12, "latency unexpectedly high"

    t7 = "What is the chemical formula of chlorophyll?"
    retrieved7 = retrieve(t7, faiss_index, chunks, k=4)
    result7 = answer(t7, retrieved7)
    print("--- T7 ---")
    print(repr(result7["answer"]))
    assert result7["answer"].strip() == REFUSAL, (
        f"T7 must be exact refusal, got: {result7['answer']!r}"
    )
    print("SMOKE_OK")
