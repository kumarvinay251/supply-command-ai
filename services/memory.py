"""
Supply Command AI — Memory / Vector Store Service
FAISS-backed RAG knowledge base.

LLM Usage Policy (strictly enforced):
    YES  build_vector_store() — 1 batch API call to embed all chunks
    YES  search()             — 1 API call to embed the query string
    NO   chunk_text()         — pure Python string splitting, zero tokens
    NO   load_documents()     — pure Python file I/O (pypdf + plain text)
    NO   load_vector_store()  — pure Python FAISS index load from disk

Similarity Guardrail:
    If best similarity score < SIMILARITY_THRESHOLD (0.75):
    → returns {"found": False, "reason": "not in knowledge base"}
    → the upstream agent MUST NOT pass this to the LLM
    → prevents hallucination from weak or absent context

Saved files:
    vector_store/
        index.faiss        — FAISS flat inner-product index
        chunks_meta.pkl    — list[dict] parallel to the FAISS vectors
"""

import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

from services.logger import get_logger

# ── Env + Logger ──────────────────────────────────────────────────────────────

load_dotenv()
log = get_logger("rag_agent")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR         = Path(__file__).parent.parent
DOCS_DIR         = BASE_DIR / "sample_data" / "documents"  # PDF files
KB_DIR           = BASE_DIR / "knowledge_base"              # Markdown / .txt files
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_META_FILE = VECTOR_STORE_DIR / "chunks_meta.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL      = "text-embedding-ada-002"
EMBEDDING_DIM        = 1536    # fixed output dimension for ada-002
SIMILARITY_THRESHOLD = 0.75    # guardrail: below this score → not found
EMBED_BATCH_SIZE     = 100     # max texts per OpenAI embeddings API call

# Matches tags like [SECTION: Introduction] or [SECTION:Overview]
SECTION_PATTERN = re.compile(r'\[SECTION:\s*(.+?)\]', re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CHUNK TEXT  —  pure Python, zero LLM tokens
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int    = 50,
    source: str     = "unknown",
) -> list[dict]:
    """
    Split raw document text into overlapping word-count chunks.

    Why overlapping?  So that answers that straddle two chunks are
    not lost — the `overlap` words appear in both the current chunk
    and the next one, preserving context continuity.

    [SECTION: name] tags are:
        • Used to track which section each chunk belongs to
        • Stripped from the stored text (stored in the 'section' field instead)
        • Carried forward to future chunks until a new tag is found

    Args:
        text:       Raw document text (string).
        chunk_size: Target number of words per chunk  (default 500).
        overlap:    Words shared between consecutive chunks (default 50).
        source:     Filename / doc identifier attached to every chunk dict.

    Returns:
        List of dicts:
            {
                "text":    str,   # cleaned chunk text
                "section": str,   # current [SECTION:] label or "General"
                "source":  str,   # filename / source identifier
            }
    """
    words           = text.split()
    chunks          = []
    current_section = "General"    # default until a [SECTION:] tag is found
    i               = 0

    while i < len(words):

        # ── Grab a window of words ────────────────────────────────────────────
        window         = words[i : i + chunk_size]
        chunk_raw      = " ".join(window)

        # ── Detect [SECTION: X] tags inside this window ───────────────────────
        # If multiple tags exist, take the LAST one as the active section
        section_hits = SECTION_PATTERN.findall(chunk_raw)
        if section_hits:
            current_section = section_hits[-1].strip()

        # ── Strip section tags from stored text (they live in metadata now) ───
        clean = SECTION_PATTERN.sub("", chunk_raw).strip()

        # ── Append non-empty chunks only ──────────────────────────────────────
        if clean:
            chunks.append({
                "text":    clean,
                "section": current_section,
                "source":  source,
            })

        # ── Advance by (chunk_size - overlap) ────────────────────────────────
        # This means the next chunk starts `overlap` words before where this
        # one ended, so critical context is never silently cut off.
        step = max(1, chunk_size - overlap)
        i   += step

    log.debug(
        f"chunk_text | source={source} | "
        f"{len(words)} words → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  2. LOAD DOCUMENTS  —  pure Python, zero LLM tokens
# ═══════════════════════════════════════════════════════════════════════════════

def load_documents() -> list[dict]:
    """
    Load and chunk all documents from both knowledge directories.

    Sources:
        DOCS_DIR  (sample_data/documents/)  — PDF files via pypdf
        KB_DIR    (knowledge_base/)          — Markdown (.md) and plain text (.txt)

    Missing directories are handled gracefully (logged as warnings, not errors).
    Individual file failures are logged and skipped so one bad file
    does not abort the entire load.

    Returns:
        Flat list of chunk dicts from chunk_text():
            [{"text": ..., "section": ..., "source": ...}, ...]
    """
    all_chunks: list[dict] = []

    # ── 1. PDF files ──────────────────────────────────────────────────────────
    if DOCS_DIR.exists():
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        log.info(f"load_documents | {len(pdf_files)} PDF(s) found in {DOCS_DIR}")

        for pdf_path in pdf_files:
            try:
                from pypdf import PdfReader

                reader   = PdfReader(str(pdf_path))
                # Join all page text with a single space — pypdf may return
                # None for scanned pages, so guard with `or ""`
                raw_text = " ".join(
                    page.extract_text() or "" for page in reader.pages
                )
                chunks = chunk_text(raw_text, source=pdf_path.name)
                all_chunks.extend(chunks)
                log.success(
                    f"load_documents | {pdf_path.name} → {len(chunks)} chunks"
                )
            except Exception as exc:
                log.error(
                    f"load_documents | failed to read {pdf_path.name}: {exc}"
                )
    else:
        log.warning(
            f"load_documents | DOCS_DIR not found: {DOCS_DIR} — skipping PDFs"
        )

    # ── 2. Markdown / plain-text files ────────────────────────────────────────
    if KB_DIR.exists():
        text_files = list(KB_DIR.glob("*.md")) + list(KB_DIR.glob("*.txt"))
        log.info(
            f"load_documents | {len(text_files)} text file(s) found in {KB_DIR}"
        )

        for txt_path in text_files:
            try:
                raw_text = txt_path.read_text(encoding="utf-8")
                chunks   = chunk_text(raw_text, source=txt_path.name)
                all_chunks.extend(chunks)
                log.success(
                    f"load_documents | {txt_path.name} → {len(chunks)} chunks"
                )
            except Exception as exc:
                log.error(
                    f"load_documents | failed to read {txt_path.name}: {exc}"
                )
    else:
        log.warning(
            f"load_documents | KB_DIR not found: {KB_DIR} — skipping markdown"
        )

    log.info(f"load_documents | total chunks ready: {len(all_chunks)}")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  3. BUILD VECTOR STORE  —  LLM used here (OpenAI embeddings)
# ═══════════════════════════════════════════════════════════════════════════════

def build_vector_store(chunks: list[dict]) -> None:
    """
    Embed all chunks using OpenAI text-embedding-ada-002 and persist
    a FAISS inner-product index to disk.

    LLM USAGE: YES
        One batch API call per EMBED_BATCH_SIZE chunks.
        Vectors are L2-normalised so inner product == cosine similarity.

    Saves:
        vector_store/index.faiss      — FAISS IndexFlatIP
        vector_store/chunks_meta.pkl  — parallel list of chunk dicts

    Args:
        chunks: Output of load_documents() — list of {text, section, source}.
    """
    if not chunks:
        log.error("build_vector_store | no chunks provided — aborting")
        return

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    log.info(
        f"build_vector_store | embedding {len(chunks)} chunks "
        f"via {EMBEDDING_MODEL} (batch_size={EMBED_BATCH_SIZE})"
    )

    # ── Embed chunks in batches to stay within API limits ────────────────────
    all_embeddings: list[list[float]] = []
    files_processed: set[str]         = set()

    for batch_num, batch_start in enumerate(
        range(0, len(chunks), EMBED_BATCH_SIZE), start=1
    ):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Track which source files were seen in this batch
        for c in batch:
            files_processed.add(c["source"])

        try:
            response         = client.embeddings.create(
                model=EMBEDDING_MODEL, input=texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            log.success(
                f"build_vector_store | batch {batch_num} — "
                f"embedded {len(batch)} chunks"
            )
        except Exception as exc:
            log.error(f"build_vector_store | batch {batch_num} failed: {exc}")
            raise   # re-raise so the caller knows the index is incomplete

    # ── Build FAISS index with L2-normalised vectors ──────────────────────────
    # Normalising so that IndexFlatIP (inner product) == cosine similarity.
    # Cosine similarity is preferred over L2 distance for text embeddings.
    vectors = np.array(all_embeddings, dtype=np.float32)
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)   # avoid div-by-zero

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    # ── Persist index + metadata to disk ─────────────────────────────────────
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    with open(CHUNKS_META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    log.success(
        f"build_vector_store | saved {index.ntotal} vectors ({EMBEDDING_DIM}D) "
        f"→ {FAISS_INDEX_FILE}"
    )
    log.info(
        f"build_vector_store | files embedded: {sorted(files_processed)}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  4. LOAD VECTOR STORE  —  pure Python, zero LLM tokens
# ═══════════════════════════════════════════════════════════════════════════════

def load_vector_store() -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously saved FAISS index and its chunk metadata from disk.
    No LLM. Pure Python FAISS + pickle I/O.

    Returns:
        (faiss_index, chunks_meta_list)
        where chunks_meta_list[i] corresponds to vector i in faiss_index.

    Raises:
        FileNotFoundError — if index files are missing (run build_vector_store first).
    """
    if not FAISS_INDEX_FILE.exists() or not CHUNKS_META_FILE.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_DIR}. "
            "Call build_vector_store() first."
        )

    index  = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(CHUNKS_META_FILE, "rb") as f:
        chunks = pickle.load(f)

    log.info(
        f"load_vector_store | loaded {index.ntotal} vectors from {FAISS_INDEX_FILE}"
    )
    return index, chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  5. SEARCH  —  LLM used here (1 embedding call for the query)
# ═══════════════════════════════════════════════════════════════════════════════

def search(
    query: str,
    top_k: int = 3,
) -> dict:
    """
    Find the most similar knowledge-base chunks for a given query.

    LLM USAGE: YES — exactly 1 API call to embed the query string.

    GUARDRAIL (hardcoded, not LLM-decided):
        If best similarity score < SIMILARITY_THRESHOLD (0.75):
        → returns {"found": False, "reason": "not in knowledge base"}
        → the caller MUST show this to the user as-is and NOT forward
          it to a language model for answer generation.
        This prevents the LLM from confidently hallucinating answers
        that have no grounding in the actual documents.

    Args:
        query:  Natural language question or keyword string.
        top_k:  Max number of chunks to return when above threshold.

    Returns:
        On success (score >= threshold):
            {
                "found": True,
                "results": [
                    {
                        "text":             str,
                        "section":          str,
                        "source":           str,
                        "similarity_score": float,   # 0.0 – 1.0
                    },
                    ...               # up to top_k items
                ]
            }

        On guardrail trigger (score < threshold):
            {
                "found":      False,
                "reason":     "not in knowledge base",
                "best_score": float,
                "threshold":  float,
            }
    """
    # ── Load the FAISS index from disk ────────────────────────────────────────
    try:
        index, chunks = load_vector_store()
    except FileNotFoundError as exc:
        log.error(f"search | {exc}")
        return {"found": False, "reason": str(exc)}

    # ── Embed the query (1 LLM call) ──────────────────────────────────────────
    client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    short_q   = query[:80] + "..." if len(query) > 80 else query
    log.info(f"search | embedding query: '{short_q}'")

    try:
        response  = client.embeddings.create(
            model=EMBEDDING_MODEL, input=[query]
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as exc:
        log.error(f"search | query embedding failed: {exc}")
        return {"found": False, "reason": f"Embedding error: {exc}"}

    # ── L2-normalise query vector to match how the index was built ────────────
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    query_vec = query_vec.reshape(1, -1)    # FAISS expects shape (1, dim)

    # ── Run FAISS inner-product search ────────────────────────────────────────
    # scores  — shape (1, top_k): cosine similarities, higher is better
    # indices — shape (1, top_k): positions in chunks list
    scores, indices = index.search(query_vec, top_k)

    best_score = float(scores[0][0])
    log.info(
        f"search | best_score={best_score:.4f} | "
        f"threshold={SIMILARITY_THRESHOLD}"
    )

    # ── GUARDRAIL — reject weak matches unconditionally ───────────────────────
    if best_score < SIMILARITY_THRESHOLD:
        log.warning(
            f"search | GUARDRAIL TRIGGERED — best_score={best_score:.4f} "
            f"< threshold={SIMILARITY_THRESHOLD} — returning 'not found'"
        )
        return {
            "found":      False,
            "reason":     "not in knowledge base",
            "best_score": round(best_score, 4),
            "threshold":  SIMILARITY_THRESHOLD,
        }

    # ── Build result list ─────────────────────────────────────────────────────
    # FAISS returns -1 for empty slots when fewer results than top_k exist
    results = [
        {
            "text":             chunks[idx]["text"],
            "section":          chunks[idx]["section"],
            "source":           chunks[idx]["source"],
            "similarity_score": round(float(score), 4),
        }
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]

    log.success(
        f"search | {len(results)} result(s) returned | "
        f"top_score={best_score:.4f}"
    )
    return {"found": True, "results": results}


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  —  run directly to verify chunking + loading logic
#  Does NOT call OpenAI — skips build/search to avoid token cost
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  services/memory.py — Self Test (no LLM calls)")
    print("=" * 55 + "\n")

    # ── Test chunk_text ───────────────────────────────────────────────────────
    sample = (
        "[SECTION: Cold Chain Policy] "
        "All temperature-sensitive products must be stored at 2-8°C. "
        "Deviation from this range must be logged within 15 minutes. "
        "[SECTION: SLA Terms] "
        "Standard delivery SLA is 5 business days. "
        "Expedited orders must be fulfilled within 48 hours. "
        "SLA breaches trigger an automatic penalty clause. " * 20
    )

    chunks = chunk_text(sample, chunk_size=50, overlap=10, source="test_doc.txt")
    print(f"  chunk_text test")
    print(f"  Input : {len(sample.split())} words")
    print(f"  Output: {len(chunks)} chunks")
    print(f"  Sample chunk 0:")
    print(f"    section : {chunks[0]['section']}")
    print(f"    text    : {chunks[0]['text'][:80]}...")
    if len(chunks) > 1:
        print(f"  Sample chunk 1:")
        print(f"    section : {chunks[1]['section']}")
        print(f"    text    : {chunks[1]['text'][:80]}...")

    print()

    # ── Test load_documents ───────────────────────────────────────────────────
    print("  load_documents test")
    docs = load_documents()
    print(f"  Total chunks loaded from disk: {len(docs)}")
    if docs:
        print(f"  First chunk source  : {docs[0]['source']}")
        print(f"  First chunk section : {docs[0]['section']}")
        print(f"  First chunk preview : {docs[0]['text'][:80]}...")

    print()

    # ── Test load_vector_store (if index exists) ──────────────────────────────
    print("  load_vector_store test")
    if FAISS_INDEX_FILE.exists():
        idx, meta = load_vector_store()
        print(f"  Index loaded — {idx.ntotal} vectors, dim={EMBEDDING_DIM}")
        print(f"  Metadata records: {len(meta)}")
    else:
        print(f"  No index at {FAISS_INDEX_FILE} — run build_vector_store() first")

    print()
    print("=" * 55)
    print("  Self-test complete")
    print("=" * 55 + "\n")
