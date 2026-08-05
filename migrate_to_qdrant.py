"""One-off migration: ChromaDB → Qdrant.

Copies every chunk out of the legacy `data/chroma_db` collection and into Qdrant,
**reusing the stored embedding vectors** — nothing is re-embedded, so a 148k-chunk
corpus migrates in minutes rather than the hours a full re-ingest would take.

Usage
─────
    # Qdrant must already be running (start.ps1 / start_mac.sh launch it)
    python3 migrate_to_qdrant.py                 # migrate everything
    python3 migrate_to_qdrant.py --dry-run       # report what would happen
    python3 migrate_to_qdrant.py --batch 500     # tune batch size
    python3 migrate_to_qdrant.py --verify-only   # just compare counts

Restartability
──────────────
Chunk IDs are deterministic (`{doc_id}_{page}_{chunk_index}`) and writes use an
overwrite policy, so re-running after an interruption is safe and idempotent — it
simply re-writes chunks it already wrote. Progress is reported per batch so an
interrupted run can be resumed by re-running the whole command.

What this does NOT backfill
───────────────────────────
* `chunk_index` metadata — the legacy pipeline never stored it. It is recovered
  here by parsing the chunk ID, which is where the value came from originally.
* `doc_number` / `revision` — these are new, produced by an LLM at ingest time.
  Migrated documents get "—" until they are re-ingested. Run with
  `--backfill-catalog` to fill them in via the LLM after migrating (slow: one
  generation per document).
* Extracted figures — the old pipeline discarded image bytes, so there is nothing
  on disk to migrate. Image *chunks* (OCR text) carry over fine; re-ingest a file
  if you want its figures persisted.

The legacy Chroma database is only ever read. Nothing is deleted — keep it until
you have verified the migration, then remove `data/chroma_db` by hand.
"""
import argparse
import asyncio
import logging
import os
import sys

import config
import store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("migrate")


def _open_chroma():
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        sys.exit(
            "chromadb is no longer in requirements.txt. Install it just for this "
            'migration:  pip install "chromadb>=0.6,<1.0"'
        )
    if not os.path.isdir(config.CHROMA_PATH):
        sys.exit(f"No legacy database found at {config.CHROMA_PATH} — nothing to migrate.")
    client = chromadb.PersistentClient(
        path=config.CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def _chunk_index_from_id(chunk_id: str, fallback: int = 0) -> int:
    """Recover chunk_index from the ID. Legacy metadata never stored it."""
    parts = chunk_id.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return fallback


def _iter_batches(col, batch_size: int):
    """Page through the Chroma collection.

    Chroma's `get` pulls everything into memory when unbounded, which on a 700 MB
    database is exactly the pattern that triggers the 1.x pool-timeout hang. Paging
    with limit/offset keeps peak memory to one batch.
    """
    offset = 0
    while True:
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = result.get("ids") or []
        if not ids:
            return
        yield result
        offset += len(ids)


async def migrate(batch_size: int, dry_run: bool) -> int:
    col = _open_chroma()
    total = col.count()
    log.info("legacy collection holds %d chunks", total)
    if dry_run:
        log.info("dry run — nothing written")
        return total

    written = 0
    skipped_no_vector = 0
    for result in _iter_batches(col, batch_size):
        documents = []
        embeddings = result.get("embeddings")
        for i, cid in enumerate(result["ids"]):
            meta = dict(result["metadatas"][i] or {})
            text = result["documents"][i] or ""
            vector = embeddings[i] if embeddings is not None else None
            if vector is None:
                skipped_no_vector += 1
                continue
            meta.setdefault("chunk_index", _chunk_index_from_id(cid))
            meta.setdefault("folder", "General")
            meta.setdefault("doc_number", "—")
            meta.setdefault("revision", "—")
            documents.append(store.to_document(cid, text, meta, list(vector)))

        if documents:
            await store.add(documents)
            written += len(documents)
        log.info("migrated %d / %d", written, total)

    if skipped_no_vector:
        log.warning("%d chunks had no stored embedding and were skipped", skipped_no_vector)
    return written


async def backfill_catalog() -> None:
    """Fill doc_number / revision for migrated documents via the LLM.

    Separate from the main migration because it costs one LLM generation per
    document and is safe to run later, or not at all.
    """
    import ingest

    docs = await store.doc_summaries()
    stale = [d for d in docs if d.get("doc_number", "—") == "—" or d.get("revision", "—") == "—"]
    log.info("%d of %d documents need catalog fields", len(stale), len(docs))
    for n, doc in enumerate(stale, start=1):
        hits = await store.get_by_filter(
            store.and_(store.eq("doc_id", doc["doc_id"]), store.eq("page", 1))
        )
        front_matter = "\n".join(
            text for text, meta, _ in sorted(hits.values(), key=lambda h: h[1].get("chunk_index", 0))
        )
        doc_no, rev = await ingest._extract_catalog_fields(doc["filename"], front_matter)
        await store.get_store().update_by_filter_async(
            filters=store.eq("doc_id", doc["doc_id"]),
            meta={"doc_number": doc_no, "revision": rev},
        )
        log.info("[%d/%d] %s → %s / %s", n, len(stale), doc["filename"], doc_no, rev)


async def verify() -> None:
    col = _open_chroma()
    legacy = col.count()
    migrated = await store.count()
    log.info("legacy=%d  qdrant=%d  delta=%d", legacy, migrated, legacy - migrated)
    if migrated < legacy:
        log.warning("Qdrant has fewer chunks than Chroma — re-run the migration.")
    else:
        log.info("counts match or exceed — migration looks complete")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate ChromaDB → Qdrant")
    ap.add_argument("--batch", type=int, default=500, help="chunks per batch (default 500)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--verify-only", action="store_true", help="compare counts and exit")
    ap.add_argument("--backfill-catalog", action="store_true",
                    help="after migrating, fill doc_number/revision via the LLM (slow)")
    ap.add_argument("--skip-bm25", action="store_true", help="do not rebuild the keyword index")
    args = ap.parse_args()

    if args.verify_only:
        await verify()
        return

    written = await migrate(args.batch, args.dry_run)
    if args.dry_run:
        return
    log.info("migrated %d chunks", written)

    if args.backfill_catalog:
        await backfill_catalog()

    if not args.skip_bm25:
        import bm25_index
        log.info("rebuilding keyword index from the new store")
        await bm25_index.rebuild_from_store()

    await verify()
    log.info("done. The legacy database at %s was not modified — "
             "delete it once you are satisfied.", config.CHROMA_PATH)
    await store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
