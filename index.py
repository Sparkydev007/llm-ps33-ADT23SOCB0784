"""Embedding + FAISS index, cached by SHA-256 of the PDF bytes.

Re-uploading the same chapter is instant because we never re-embed
when the file hash already has a cache folder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path("cache")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# One model in process memory. Loading MiniLM on every Streamlit rerun
# would waste several seconds.
_MODEL: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 once (runs fully offline after first download)."""
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def sha256_file(path: str | Path) -> str:
    """Hash the PDF bytes so identical re-uploads hit the cache."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 64)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalized float32 embeddings (shape: n x 384).

    WHY normalize: IndexFlatL2 then ranks by a distance that matches
    cosine similarity, which works well for MiniLM sentence vectors.
    """
    model = get_embedder()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def build_index(chunks: list[dict[str, Any]]) -> faiss.IndexFlatL2:
    """Embed chunk texts and build a FAISS IndexFlatL2."""
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def cache_paths(pdf_hash: str, cache_dir: Path | None = None) -> dict[str, Path]:
    """Return paths for the FAISS file and the chunk JSON sidecar."""
    folder = (cache_dir or CACHE_DIR) / pdf_hash
    return {
        "dir": folder,
        "index": folder / "index.faiss",
        "chunks": folder / "chunks.json",
    }


def save_index(
    index: faiss.IndexFlatL2,
    chunks: list[dict[str, Any]],
    pdf_hash: str,
    cache_dir: Path | None = None,
) -> None:
    """Write the index and chunks so the next upload of this PDF is free."""
    paths = cache_paths(pdf_hash, cache_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(paths["index"]))
    with open(paths["chunks"], "w", encoding="utf-8") as handle:
        json.dump(chunks, handle, ensure_ascii=False, indent=2)


def load_index(
    pdf_hash: str,
    cache_dir: Path | None = None,
) -> tuple[faiss.IndexFlatL2, list[dict[str, Any]]] | None:
    """Load a cached index. Return None if this PDF was never indexed."""
    paths = cache_paths(pdf_hash, cache_dir)
    if not paths["index"].exists() or not paths["chunks"].exists():
        return None
    index = faiss.read_index(str(paths["index"]))
    with open(paths["chunks"], encoding="utf-8") as handle:
        chunks = json.load(handle)
    return index, chunks


def cache_hit(pdf_hash: str, cache_dir: Path | None = None) -> bool:
    """True when both the FAISS file and chunk JSON already exist."""
    paths = cache_paths(pdf_hash, cache_dir)
    return paths["index"].exists() and paths["chunks"].exists()


def build_or_load(
    pdf_path: str | Path,
    chunks: list[dict[str, Any]],
    cache_dir: Path | None = None,
) -> tuple[faiss.IndexFlatL2, list[dict[str, Any]], str, bool]:
    """Build a new index or load from SHA-256 cache.

    Returns (index, chunks, pdf_hash, from_cache).
    """
    pdf_hash = sha256_file(pdf_path)
    cached = load_index(pdf_hash, cache_dir)
    if cached is not None:
        return cached[0], cached[1], pdf_hash, True
    index = build_index(chunks)
    save_index(index, chunks, pdf_hash, cache_dir)
    return index, chunks, pdf_hash, False


if __name__ == "__main__":
    import sys

    from ingest import load_chunks

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/ncert_class10_science_ch6_life_processes.pdf"
    )
    chunks = load_chunks(pdf_path)
    index, stored, pdf_hash, from_cache = build_or_load(pdf_path, chunks)
    print(f"pdf_sha256={pdf_hash}")
    print(f"from_cache={from_cache}")
    print(f"ntotal={index.ntotal}")
    print(f"chunks={len(stored)}")
    # Second call must be a cache hit.
    _, _, _, hit_again = build_or_load(pdf_path, chunks)
    print(f"second_call_from_cache={hit_again}")
