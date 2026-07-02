"""One-off backfill: regenerate each document's summary chunk using the fixed
front-matter extraction (ingest.py now scans the first 3 pages instead of just
page 1). Not part of the app — reconstructs page text from chunks already in
ChromaDB rather than re-parsing source files (which aren't retained on disk).

Usage: python3 backfill_summaries.py [--dry-run]
"""
import asyncio
import sys

import bm25_index
import ingest
import llm


async def main():
    dry_run = "--dry-run" in sys.argv
    col = ingest._db()

    result = col.get(include=["documents", "metadatas"])
    by_doc: dict[str, dict] = {}
    for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        doc_id = meta["doc_id"]
        entry = by_doc.setdefault(doc_id, {"filename": meta["filename"], "chunks": []})
        entry["chunks"].append((cid, doc, meta))

    print(f"{len(by_doc)} documents found\n")

    changed = 0
    for doc_id, entry in sorted(by_doc.items(), key=lambda kv: kv[1]["filename"]):
        filename = entry["filename"]
        chunks = entry["chunks"]

        old_summary_text = None
        for cid, doc, meta in chunks:
            if cid.endswith("_1_-1"):
                old_summary_text = doc
                break

        # Reconstruct first-3-pages text from existing text chunks (page 1-3,
        # chunk_index >= 0 — excludes the old summary/image chunks themselves).
        page_chunks = [
            (meta["page"], meta.get("chunk_index", 0), doc)
            for cid, doc, meta in chunks
            if meta.get("page") in (1, 2, 3) and meta.get("chunk_type") != "image"
            and not cid.endswith("_1_-1")
        ]
        page_chunks.sort(key=lambda t: (t[0], t[1]))
        pages_by_num: dict[int, list[str]] = {}
        for page_num, _, text in page_chunks:
            pages_by_num.setdefault(page_num, []).append(text)
        pages = [(p, "\n".join(texts)) for p, texts in sorted(pages_by_num.items())]

        title = ingest._extract_doc_title(pages, filename)
        full_text = "\n".join(t for _, t in pages)
        summary = ingest._summary_chunk(title, full_text)
        if not summary:
            front_matter_text = "\n".join(t for _, t in pages[:3])
            llm_sum = await ingest._llm_summary(title, front_matter_text)
            if llm_sum:
                summary = f"{title} — {llm_sum}"

        if not summary:
            print(f"[{filename}] no summary generated (unchanged)")
            continue

        if summary == old_summary_text:
            print(f"[{filename}] summary unchanged")
            continue

        changed += 1
        print(f"[{filename}] SUMMARY CHANGED")
        print(f"  OLD: {old_summary_text!r}")
        print(f"  NEW: {summary!r}\n")

        if not dry_run:
            summary_id = f"{doc_id}_1_-1"
            embeddings = await llm.embed([summary])
            existing_meta = next((m for _, _, m in chunks if _.endswith("_1_-1")), None)
            meta = existing_meta or {"doc_id": doc_id, "filename": filename, "page": 1, "folder": "General"}
            col.upsert(ids=[summary_id], embeddings=embeddings, documents=[summary], metadatas=[meta])

    print(f"\n{changed}/{len(by_doc)} documents updated")

    if not dry_run and changed:
        print("Rebuilding BM25 index...")
        bm25_index.rebuild(col)
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
