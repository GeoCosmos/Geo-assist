#!/usr/bin/env python3
"""
Clear the vector store + BM25, then re-ingest a set of files through the updated pipeline.

Usage:
    python3 reindex.py                          # re-ingest files listed in --files (default: see below)
    python3 reindex.py --dir ~/Desktop/geo-assist-eval/docs --limit 20
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True,
                    help="Directory of source files to re-ingest")
parser.add_argument("--limit", type=int, default=0,
                    help="Max files to ingest (0 = all)")
parser.add_argument("--filter", nargs="*",
                    help="Only ingest filenames matching these names (exact)")
parser.add_argument("--folder", default="General",
                    help="Folder name shown in the document list")
args = parser.parse_args()

src_dir = Path(args.dir).expanduser()
if not src_dir.is_dir():
    print(f"ERROR: {src_dir} is not a directory")
    sys.exit(1)

# ── wipe existing index ───────────────────────────────────────────────────────
import config
import store

if os.path.exists(config.BM25_PATH):
    os.remove(config.BM25_PATH)
    print("  BM25 index deleted")

# ── collect files ─────────────────────────────────────────────────────────────
ALLOWED = {".pdf", ".docx", ".pptx", ".txt", ".csv"}
all_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in ALLOWED])

if args.filter:
    filter_set = set(args.filter)
    all_files = [f for f in all_files if f.name in filter_set]

if args.limit:
    all_files = all_files[: args.limit]

print(f"\nRe-ingesting {len(all_files)} files from {src_dir}")

# ── ingest ────────────────────────────────────────────────────────────────────
import bm25_index as _bm25
import ingest as _ingest

BATCH = 10

async def run():
    print("Clearing the document store…")
    await store.clear()

    total_chunks = 0
    # No BM25 monkey-patching needed any more: the index is incremental, so a
    # commit per batch only re-fits from cached tokens rather than re-reading and
    # re-stemming the entire corpus.
    for i in range(0, len(all_files), BATCH):
        batch = all_files[i : i + BATCH]
        files = [(f.read_bytes(), f.name) for f in batch]
        results = await _ingest.ingest_many(files, folder=args.folder)
        for r in results:
            status = r.get("status", "?")
            chunks = r.get("chunks", 0)
            total_chunks += chunks
            print(f"  [{status}] {r['filename']}  ({chunks} chunks)")
        total_batches = (len(all_files) + BATCH - 1) // BATCH
        print(f"  — batch {i//BATCH + 1}/{total_batches} done, {total_chunks} chunks total so far")

    _bm25.commit()
    print(f"\nDone. {total_chunks} chunks indexed.")
    await store.aclose()

asyncio.run(run())
