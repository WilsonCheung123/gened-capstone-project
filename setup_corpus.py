"""
One-time corpus download and indexing script.
Downloads texts from Project Gutenberg, chunks them, embeds with
sentence-transformers/all-MiniLM-L6-v2, and stores in ChromaDB.
Idempotent: skips figures already indexed.
"""

import os
import re
import sys
import math
import time
import random
import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from figures import FIGURES

CORPUS_DIR = Path("corpus")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "figures"
CHUNK_WORDS = 400
OVERLAP_WORDS = 50
MAX_CHUNKS_PER_FIGURE = 250
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def download_text(url: str, dest_path: Path) -> str:
    """Download text from URL, with retry on failure."""
    if dest_path.exists():
        print(f"  [cache] {dest_path.name} already on disk, skipping download.")
        return dest_path.read_text(encoding="utf-8", errors="replace")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; CounselFromTheDead/1.0)"}
    for attempt in range(3):
        try:
            print(f"  [download] {url}")
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(resp.content)
                print(f"  [saved] {dest_path} ({len(resp.content):,} bytes)")
                return resp.text
            else:
                print(f"  [warn] HTTP {resp.status_code} for {url} (attempt {attempt+1})")
        except requests.RequestException as e:
            print(f"  [warn] Request error: {e} (attempt {attempt+1})")
        time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to download {url} after 3 attempts.")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove PG license header and footer."""
    start_pattern = re.compile(r"\*\*\* START OF", re.IGNORECASE)
    end_pattern = re.compile(r"\*\*\* END OF", re.IGNORECASE)

    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if start_pattern.search(line):
            start_idx = i + 1
            break

    for i, line in enumerate(lines):
        if end_pattern.search(line):
            end_idx = i
            # don't break — take the LAST occurrence

    if start_idx == 0 and end_idx == len(lines):
        # markers not found — return as-is
        return text

    return "\n".join(lines[start_idx:end_idx])


def chunk_text(text: str, target_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """
    Chunk text into ~target_words segments with ~overlap_words overlap,
    splitting on paragraph boundaries (double newlines) where possible.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current_words: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        # If adding this paragraph would exceed target, flush first
        if current_word_count + len(para_words) > target_words and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap: take last overlap_words words
            current_words = current_words[-overlap_words:] if overlap_words else []
            current_word_count = len(current_words)

        current_words.extend(para_words)
        current_word_count += len(para_words)

    # Flush remaining
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def sample_chunks_evenly(all_chunks_by_work: list[tuple[str, list[str]]], max_total: int) -> list[tuple[str, str, int]]:
    """
    Given a list of (work_name, [chunks]) tuples, sample evenly across works
    so total <= max_total. Returns list of (work_name, chunk_text, original_idx).
    """
    total = sum(len(chunks) for _, chunks in all_chunks_by_work)
    if total <= max_total:
        result = []
        for work_name, chunks in all_chunks_by_work:
            for idx, chunk in enumerate(chunks):
                result.append((work_name, chunk, idx))
        return result

    result = []
    n_works = len(all_chunks_by_work)
    per_work_budget = max_total // n_works

    for work_name, chunks in all_chunks_by_work:
        if not chunks:
            continue
        if len(chunks) <= per_work_budget:
            for idx, chunk in enumerate(chunks):
                result.append((work_name, chunk, idx))
        else:
            # Sample evenly
            step = len(chunks) / per_work_budget
            selected = [int(i * step) for i in range(per_work_budget)]
            for i in selected:
                result.append((work_name, chunks[i], i))

    return result[:max_total]


def main():
    print("=== Counsel from the Dead — Corpus Setup ===\n")

    CORPUS_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Check which (figure, work) pairs are already indexed — work-level idempotency
    existing_work_keys: set[tuple[str, str]] = set()
    try:
        existing = collection.get(include=["metadatas"])
        for meta in existing["metadatas"]:
            if meta and "figure" in meta and "work" in meta:
                existing_work_keys.add((meta["figure"], meta["work"]))
    except Exception:
        pass

    if existing_work_keys:
        print(f"Already indexed {len(existing_work_keys)} (figure, work) pairs — will skip those.")

    # Determine whether any figure needs work
    any_new = any(
        (fig_key, work["name"]) not in existing_work_keys
        for fig_key, fig_data in FIGURES.items()
        for work in fig_data["works"]
    )

    if not any_new:
        print("All works already indexed. Nothing to do.")
        print_summary(collection)
        return

    # Load embedding model once
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print("Model loaded.\n")

    chunk_counts = {}

    for fig_key, fig_data in FIGURES.items():
        fig_name = fig_data["name"]

        new_works = [
            w for w in fig_data["works"]
            if (fig_key, w["name"]) not in existing_work_keys
        ]
        if not new_works:
            print(f"--- {fig_name}: all works already indexed, skipping ---\n")
            continue

        print(f"--- {fig_name} ({fig_key}): {len(new_works)} new work(s) to index ---")

        fig_corpus_dir = CORPUS_DIR / fig_key
        fig_corpus_dir.mkdir(parents=True, exist_ok=True)

        all_chunks_by_work = []

        for work in new_works:
            work_name = work["name"]
            url = work["url"]
            safe_name = re.sub(r"[^\w\s-]", "", work_name).strip().replace(" ", "_")
            dest_path = fig_corpus_dir / f"{safe_name}.txt"

            try:
                raw_text = download_text(url, dest_path)
            except RuntimeError as e:
                print(f"  [error] {e}")
                print(f"  Skipping work: {work_name}")
                continue

            cleaned = strip_gutenberg_boilerplate(raw_text)
            chunks = chunk_text(cleaned)
            print(f"  [chunk] '{work_name}': {len(chunks)} chunks before cap")
            all_chunks_by_work.append((work_name, chunks))

        if not all_chunks_by_work:
            print(f"  [skip] No new works successfully downloaded for {fig_name}.\n")
            continue

        # Sample evenly across new works; budget is proportional to how many works are new
        per_work_budget = MAX_CHUNKS_PER_FIGURE // max(len(fig_data["works"]), 1)
        work_budget = per_work_budget * len(all_chunks_by_work)
        sampled = sample_chunks_evenly(all_chunks_by_work, work_budget)
        print(f"  [total] {len(sampled)} new chunks to index")

        # Embed and store
        texts = [s[1] for s in sampled]
        metas = [
            {
                "figure": fig_key,
                "figure_name": fig_name,
                "work": s[0],
                "chunk_index": s[2],
            }
            for s in sampled
        ]
        ids = [f"{fig_key}_{s[0].replace(' ', '_')}_{s[2]}" for s in sampled]

        print(f"  [embed] Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = model.encode(batch, show_progress_bar=False).tolist()
            all_embeddings.extend(embeddings)
            print(f"    batch {i//BATCH_SIZE + 1}/{math.ceil(len(texts)/BATCH_SIZE)} done")

        print(f"  [store] Writing to ChromaDB collection '{COLLECTION_NAME}'...")
        for i in range(0, len(texts), BATCH_SIZE):
            collection.upsert(
                ids=ids[i : i + BATCH_SIZE],
                documents=texts[i : i + BATCH_SIZE],
                embeddings=all_embeddings[i : i + BATCH_SIZE],
                metadatas=metas[i : i + BATCH_SIZE],
            )

        chunk_counts[fig_key] = len(sampled)
        print(f"  [done] {fig_name}: {len(sampled)} new chunks indexed.\n")

    print_summary(collection)


def print_summary(collection):
    print("\n=== Final Summary ===")
    try:
        all_data = collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in all_data["metadatas"]:
            if meta and "figure" in meta:
                fig = meta["figure"]
                counts[fig] = counts.get(fig, 0) + 1

        for fig_key in sorted(FIGURES.keys()):
            count = counts.get(fig_key, 0)
            name = FIGURES[fig_key]["name"]
            print(f"  {name:30s} ({fig_key:12s}): {count:4d} chunks")

        total = sum(counts.values())
        print(f"\n  TOTAL: {total} chunks across {len(counts)} figures")
    except Exception as e:
        print(f"  [error] Could not retrieve summary: {e}")


if __name__ == "__main__":
    main()
