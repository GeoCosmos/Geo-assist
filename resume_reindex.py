#!/usr/bin/env python3
"""
Resume an interrupted reindex from where it left off.

Connects to the existing store (no wipe), finds which filenames are already
indexed, and only ingests the missing ones.

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

import bm25_index as _bm25
import ingest as _ingest
import store

ALLOWED = {".pdf", ".docx", ".pptx", ".txt", ".csv"}
all_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in ALLOWED])

BATCH = 10


async def run():
    # Document-level listing rather than a full metadata scan of every chunk.
    docs = await store.doc_summaries()
    already_done = {d["filename"] for d in docs}
    print(f"Already indexed: {len(already_done)} unique filenames ({await store.count()} chunks)")

    remaining = [f for f in all_files if f.name not in already_done]
    total_batches = (len(remaining) + BATCH - 1) // BATCH
    print(f"Remaining: {len(remaining)} files\n")

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

    _bm25.commit()
    total_in_index = await store.count()
    print(f"\nDone. {total_chunks} new chunks added. Total in index: {total_in_index}")
    await store.aclose()

asyncio.run(run())
