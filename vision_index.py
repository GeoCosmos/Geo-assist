#!/usr/bin/env python3
"""
Offline vision indexer — analyzes images in already-ingested PDFs, PPTXs, and
DOCXs and adds image chunks to ChromaDB. Run this while the main server is NOT
running, ideally overnight or whenever the machine is idle.

The server does not need to be running. Ollama does.

Usage:
    python3 vision_index.py --dir ~/Desktop/my-docs
    python3 vision_index.py --dir ~/Desktop/my-docs --model moondream
    python3 vision_index.py --dir ~/Desktop/my-docs --skip-existing
"""
import argparse
import asyncio
import hashlib
import sys
import time
from pathlib import Path

import config
import ingest
import llm

SUPPORTED = {".pdf", ".pptx", ".docx"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="Directory containing the original PDF/PPTX/DOCX files")
    parser.add_argument("--model", default="moondream",
                        help="Vision model to use (default: moondream)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have image chunks in ChromaDB")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max files to process (0 = all)")
    return parser.parse_args(argv)


async def process_file(path: Path, skip_existing: bool = False) -> dict:
    """Analyze images in one file and add image chunks to ChromaDB.

    Returns a dict with keys: file, status, images.
    Statuses: ok | not_ingested | no_images | skipped_existing | analysis_failed
    """
    data = path.read_bytes()
    doc_id = hashlib.sha256(data).hexdigest()[:16]

    col = ingest._db()
    existing = col.get(where={"doc_id": {"$eq": doc_id}})
    if not existing["ids"]:
        return {"file": path.name, "status": "not_ingested", "images": 0}

    if skip_existing:
        img_chunks = col.get(where={"$and": [{"doc_id": {"$eq": doc_id}},
                                              {"chunk_type": {"$eq": "image"}}]})
        if img_chunks["ids"]:
            return {"file": path.name, "status": "skipped_existing",
                    "images": len(img_chunks["ids"])}

    raw_images = ingest.extract_images(data, path.name)
    if not raw_images:
        return {"file": path.name, "status": "no_images", "images": 0}

    print(f"  analyzing {len(raw_images)} image(s)…", flush=True)
    image_descriptions = await ingest._analyze_images(raw_images, path.name)
    if not image_descriptions:
        return {"file": path.name, "status": "analysis_failed", "images": 0}

    metas  = existing["metadatas"]
    folder = metas[0].get("folder", "General") if metas else "General"
    access = metas[0].get("access", "public")  if metas else "public"
    owner  = metas[0].get("owner",  "")        if metas else ""
    filename = metas[0].get("filename", path.name) if metas else path.name

    pages = ingest.extract_pages(data, path.name)
    title = ingest._extract_doc_title(pages, path.name)

    image_chunks = [
        {
            "page": page_num,
            "chunk_index": ingest._IMAGE_CHUNK_IDX_BASE - img_idx,
            "text": f"[Figure on page {page_num}]: {desc}",
            "chunk_type": "image",
        }
        for page_num, img_idx, desc in image_descriptions
    ]
    image_texts      = [f"[{title}] {c['text']}" for c in image_chunks]
    image_embeddings = await llm.embed(image_texts)

    async with ingest._lock:
        col = ingest._db()
        stale = col.get(where={"$and": [{"doc_id": {"$eq": doc_id}},
                                         {"chunk_type": {"$eq": "image"}}]})
        if stale["ids"]:
            col.delete(ids=stale["ids"])
        ids = [f"{doc_id}_{c['page']}_{c['chunk_index']}" for c in image_chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "page": c["page"],
                "folder": folder,
                "access": access,
                "owner": owner,
                "chunk_type": "image",
            }
            for c in image_chunks
        ]
        col.add(ids=ids, embeddings=image_embeddings,
                documents=image_texts, metadatas=metadatas)
        ingest._invalidate_cache()

    return {"file": path.name, "status": "ok", "images": len(image_chunks)}


async def run(src_dir: Path, model: str, skip_existing: bool, limit: int) -> list[dict]:
    config.VISION_MODEL = model

    files = sorted(
        p for p in src_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED and p.is_file()
    )
    if limit:
        files = files[:limit]

    if not files:
        print(f"No PDF/PPTX/DOCX files found in {src_dir}")
        return []

    print(f"Vision model : {model}")
    print(f"Files found  : {len(files)}")
    print(f"Skip existing: {skip_existing}")
    print()

    results = []
    for i, path in enumerate(files, 1):
        t0 = time.time()
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        try:
            result = await process_file(path, skip_existing=skip_existing)
        except Exception as e:
            result = {"file": path.name, "status": f"error: {e}", "images": 0}
        elapsed = time.time() - t0
        print(f"  → {result['status']} | {result['images']} image chunk(s) | {elapsed:.1f}s")
        results.append(result)

    total = sum(r["images"] for r in results)
    print(f"\nDone. {total} image chunk(s) added across {len(files)} file(s).")
    return results


if __name__ == "__main__":
    args = parse_args()
    src_dir = Path(args.dir).expanduser()
    if not src_dir.is_dir():
        print(f"ERROR: {src_dir} is not a directory")
        sys.exit(1)
    asyncio.run(run(src_dir, args.model, args.skip_existing, args.limit))
