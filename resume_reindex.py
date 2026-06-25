#!/usr/bin/env python3
"""
Resume an interrupted reindex from where it left off.

Connects to the existing ChromaDB (no wipe), finds which filenames are
already indexed, and only ingests the missing ones. Defers BM25 rebuild
to a single call at the end.

Usage:
    python3 resume_reindex.py
    python3 resume_reindex.py --dir ~/Desktop/geo-assist-eval/docs
"""
import argparse
import asyncio
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=str(Path.home() / "Desktop/geo-assist-eval/docs"))
parser.add_argument("--folder", default="General", help="Folder name shown in the document list")
args = parser.parse_args()

src_dir = Path(args.dir).expanduser()
if not src_dir.is_dir():
    print(f"ERROR: {src_dir} is not a directory")
    sys.exit(1)

import config  # noqa: E402
import bm25_index as _bm25  # noqa: E402
import ingest as _ingest  # noqa: E402
import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

client = chromadb.PersistentClient(
    path=config.CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False),
)
col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

# Find already-ingested filenames
all_meta = col.get(include=["metadatas"])["metadatas"] or []
already_done = {m["filename"] for m in all_meta}
print(f"Already indexed: {len(already_done)} unique filenames ({col.count()} chunks)")

ALLOWED = {".pdf", ".docx", ".pptx", ".txt", ".csv"}
all_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in ALLOWED])
remaining = [f for f in all_files if f.name not in already_done]
print(f"Remaining: {len(remaining)} files\n")

BATCH = 10
total_batches = (len(remaining) + BATCH - 1) // BATCH

# Skip BM25 rebuilds during the loop — one rebuild at the end
real_rebuild = _bm25.rebuild
_bm25.rebuild = lambda col: None

async def run():
    total_chunks = 0
    for i in range(0, len(remaining), BATCH):
        batch = remaining[i : i + BATCH]
        files = [(f.read_bytes(), f.name) for f in batch]
        results = await _ingest.ingest_many(files, folder=args.folder)
        for r in results:
            status = r.get("status", "?")
            chunks = r.get("chunks", 0)
            total_chunks += chunks
            print(f"  [{status}] {r['filename']}  ({chunks} chunks)")
        print(f"  — batch {i//BATCH + 1}/{total_batches} done, {total_chunks} new chunks so far")

    _bm25.rebuild = real_rebuild
    total_in_index = col.count()
    print(f"\nBuilding BM25 index over all {total_in_index} chunks…")
    real_rebuild(col)
    print(f"\nDone. {total_chunks} new chunks added. Total in index: {total_in_index}")

asyncio.run(run())
