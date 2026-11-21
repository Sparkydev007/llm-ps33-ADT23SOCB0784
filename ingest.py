"""PDF chapter ingestion: extract text and split into page-tagged chunks.

WHY this chunker exists:
- Retrieval works on chunks, not whole pages.
- A fact can sit on a paragraph boundary. We prepend a tail overlap
  from the previous chunk so that fact is present in both chunks.
- Small paragraphs (headings, one-liners) are merged so embeddings
  see enough context (~600 characters).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


class EmptyPdfError(ValueError):
    """Raised when pypdf finds no extractable text (often a scanned PDF)."""


def _normalize_page_text(raw: str) -> str:
    """Clean PDF extraction noise without changing the words."""
    text = raw.replace("\x00", " ")
    # Collapse many spaces but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(page_text: str) -> list[str]:
    """Split a page into paragraphs. Blank lines are the main boundary."""
    parts = re.split(r"\n\s*\n", page_text)
    paragraphs: list[str] = []
    for part in parts:
        cleaned = " ".join(part.split())
        if cleaned:
            paragraphs.append(cleaned)
    # If the page had no blank lines, still keep it as one block.
    if not paragraphs and page_text.strip():
        paragraphs.append(" ".join(page_text.split()))
    return paragraphs


def _word_tail(text: str, overlap_chars: int) -> str:
    """Return the last ~overlap_chars of text, starting at a word boundary.

    WHY word-aligned: cutting mid-word would create junk tokens in the
    next chunk and hurt both retrieval and verbatim quoting.
    """
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text
    tail = text[-overlap_chars:]
    # Drop a possible partial first word in the tail.
    space = tail.find(" ")
    if space != -1 and space < len(tail) - 1:
        tail = tail[space + 1 :]
    return tail.strip()


def _pack_paragraphs(
    paragraphs: list[str],
    target: int,
) -> list[str]:
    """Merge small paragraphs until each block is near `target` characters.

    WHY merge: a 20-character heading alone is a poor embedding.
    We do not split a paragraph unless it is already longer than target.
    """
    blocks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
            continue
        candidate = current + " " + para
        if len(candidate) <= target:
            current = candidate
        else:
            blocks.append(current)
            current = para
    if current:
        blocks.append(current)

    # Split oversized blocks so one giant paragraph cannot dominate.
    final: list[str] = []
    for block in blocks:
        if len(block) <= target * 2:
            final.append(block)
            continue
        words = block.split()
        buf: list[str] = []
        size = 0
        for word in words:
            extra = len(word) + (1 if buf else 0)
            if buf and size + extra > target:
                final.append(" ".join(buf))
                buf = [word]
                size = len(word)
            else:
                buf.append(word)
                size += extra
        if buf:
            final.append(" ".join(buf))
    return final


def extract_pages(path: str | Path) -> list[dict[str, Any]]:
    """Read every page. Empty pages are skipped, not treated as errors."""
    reader = PdfReader(str(path))
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = _normalize_page_text(raw)
        if not text:
            # WHY skip: blank or image-only pages add empty vectors.
            continue
        pages.append({"page": i + 1, "text": text})
    return pages


def load_chunks(
    path: str | Path,
    target: int = 600,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """Load a chapter PDF into page-tagged text chunks.

    Each chunk is {"page": int, "text": str}.
    `page` is the page where the *new* (non-overlap) text comes from.
    The overlap prefix may include the tail of the previous page.
    """
    pages = extract_pages(path)
    if not pages:
        raise EmptyPdfError(
            "This PDF has no extractable text; OCR is out of scope"
        )

    chunks: list[dict[str, Any]] = []
    previous_tail = ""

    for page_info in pages:
        page_num = page_info["page"]
        paragraphs = _split_paragraphs(page_info["text"])
        blocks = _pack_paragraphs(paragraphs, target=target)

        for block in blocks:
            # WHY prepend tail-overlap: a sentence split across two
            # packed blocks (or two pages) would otherwise sit on a
            # chunk boundary and retrieval might miss half of it.
            if previous_tail:
                text = previous_tail + " " + block
            else:
                text = block
            chunks.append({"page": page_num, "text": text.strip()})
            previous_tail = _word_tail(block, overlap)

    return chunks


def page_distribution(chunks: list[dict[str, Any]]) -> dict[int, int]:
    """Count how many chunks came from each page (for smoke tests)."""
    counts: dict[int, int] = {}
    for chunk in chunks:
        page = int(chunk["page"])
        counts[page] = counts.get(page, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/ncert_class10_science_ch6_life_processes.pdf"
    )
    print(f"Loading: {pdf_path}")
    try:
        built = load_chunks(pdf_path)
    except EmptyPdfError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    dist = page_distribution(built)
    print(f"chunk_count={len(built)}")
    print("page_distribution:")
    for page, count in dist.items():
        print(f"  page {page}: {count} chunk(s)")
    if built:
        preview = built[0]["text"][:180].replace("\n", " ")
        print(f"first_chunk_preview: {preview}...")
