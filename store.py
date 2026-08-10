"""Qdrant-backed document store — the single seam between the app and the vector DB.

Replaces the ChromaDB collection that `ingest` and `retriever` used to talk to
directly. Everything that touches persistence goes through this module.

Why Qdrant over ChromaDB
────────────────────────
* ChromaDB 1.x's Rust connection pool hangs indefinitely on databases ≥ 700 MB
  (see CLAUDE.md "Known issues"). The workaround was pinning to the 0.6 Python
  backend, which is slower and unmaintained.
* Chroma's Python API is synchronous. Every `col.get` / `col.query` blocked the
  asyncio event loop, so the API could not serve `/ingest/status` polls while a
  query was running. Qdrant's client has a real async API, used throughout here.
* Qdrant has payload indexes, so folder- and doc_id-filtered queries stay fast
  instead of scanning every payload.

Server, not embedded
────────────────────
`QdrantDocumentStore` also supports an embedded mode (`path=...`). It is not used:
that mode is brute-force only, documented as suitable for under ~20k points, and
raises RuntimeError on concurrent access to the same path. The production corpus
was ~35k chunks when last measured (34,809 across 16 documents, Aug 2026 —
earlier notes claimed 148k, which the database did not bear out). Still well past
the local-mode ceiling. The start scripts launch a loopback-bound `qdrant` server.

Chunk IDs
─────────
Chunk IDs keep the existing `{doc_id}_{page}_{chunk_index}` scheme, because the
retrieval pipeline parses meaning out of them (summary chunks end in `_1_-1`,
neighbour expansion reconstructs sibling IDs arithmetically). Qdrant only accepts
UUID or unsigned-int point IDs, but the Haystack integration transparently maps
each string ID through a deterministic uuid5 and preserves the original in the
payload, so the scheme survives the move unchanged.

Distances
─────────
Chroma returned cosine *distance* (lower is better, 0..2). Qdrant returns cosine
*similarity* (higher is better, -1..1). This module converts back to distance via
`1.0 - score` so `DISTANCE_THRESHOLD` and the pipeline's sentinel values (991.0
through 999.0, which mark force-injected chunks) keep their existing meaning.
"""
import asyncio
import logging
import os
from typing import Any

# ── air-gap: this must execute BEFORE haystack is imported ────────────────────
# Haystack reads HAYSTACK_TELEMETRY_ENABLED at import time and, when unset, posts
# usage events to a deepset endpoint. config sets this too, but relying on import
# *order* to enforce it is fragile: an import sorter will happily move
# `import config` below the haystack imports and silently re-enable telemetry.
# This assignment sits between the two import blocks, where a sorter cannot move
# an import across it, so the ordering is structural rather than conventional.
# The test in tests/test_airgap.py pins this.
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.telemetry import _telemetry
from haystack_integrations.document_stores.qdrant import (
    QdrantDocumentStore,
)

import config

# Belt and braces: even if the env var were somehow missed, a null telemetry
# object cannot send anything.
_telemetry.send_event = lambda *a, **kw: None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# (content, metadata, distance) — the shape the retrieval pipeline works in.
Hit = tuple[str, dict, float]

_store: QdrantDocumentStore | None = None
_lock = asyncio.Lock()


def get_store() -> QdrantDocumentStore:
    """Lazily construct the document store. Safe to call from any thread."""
    global _store
    if _store is None:
        _store = QdrantDocumentStore(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            index=config.QDRANT_INDEX,
            embedding_dim=config.EMBED_DIM,
            similarity="cosine",
            # Keep payloads on disk; the 16 GB target machine should not hold tens
            # of thousands of chunk texts in RAM alongside Ollama's resident models.
            on_disk_payload=True,
            hnsw_config={"m": 16, "ef_construct": 100, "on_disk": False},
            payload_fields_to_index=config.QDRANT_INDEXED_FIELDS,
            write_batch_size=config.QDRANT_WRITE_BATCH,
            progress_bar=False,
            return_embedding=False,
        )
    return _store


def reset_store(store: QdrantDocumentStore | None) -> None:
    """Swap the module singleton. Used by tests to inject an isolated store."""
    global _store
    _store = store


# ── filter helpers ────────────────────────────────────────────────────────────
# Haystack filter syntax, converted to Qdrant conditions by the integration.

def eq(field: str, value: Any) -> dict:
    return {"operator": "==", "field": f"meta.{field}", "value": value}


def in_(field: str, values: list) -> dict:
    return {"operator": "in", "field": f"meta.{field}", "value": values}


def and_(*conditions: dict | None) -> dict | None:
    live = [c for c in conditions if c]
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    return {"operator": "AND", "conditions": live}


def with_folder(where: dict | None, folder: str | None) -> dict | None:
    """Intersect a filter with an optional folder constraint."""
    return and_(where, eq("folder", folder) if folder else None)


# ── conversion ────────────────────────────────────────────────────────────────

def to_document(chunk_id: str, text: str, meta: dict, embedding: list[float] | None = None) -> Document:
    return Document(id=chunk_id, content=text, meta=meta, embedding=embedding)


def _as_hit(doc: Document, *, distance: float | None = None) -> Hit:
    """Convert a Haystack Document to the (content, meta, distance) tuple.

    Qdrant cosine score is a similarity in -1..1; the pipeline expects a distance
    in 0..2 where lower is better. When a document comes back from a filter fetch
    rather than a similarity search it has no score, and the caller supplies the
    sentinel distance that marks how it was injected.
    """
    if distance is None:
        score = doc.score
        distance = (1.0 - score) if score is not None else 999.0
    return (doc.content or "", dict(doc.meta), distance)


# ── reads ─────────────────────────────────────────────────────────────────────

async def count() -> int:
    return await get_store().count_documents_async()


async def query_embedding(
    vector: list[float],
    top_k: int,
    filters: dict | None = None,
) -> dict[str, Hit]:
    """Dense similarity search → {chunk_id: (content, meta, distance)}."""
    docs = await get_store()._query_by_embedding_async(
        query_embedding=vector,
        filters=filters,
        top_k=top_k,
    )
    return {d.id: _as_hit(d) for d in docs}


async def get_by_ids(ids: list[str], *, distance: float = 999.0) -> dict[str, Hit]:
    """Fetch specific chunks by ID. Missing IDs are simply absent from the result."""
    if not ids:
        return {}
    docs = await get_store().get_documents_by_id_async(ids=ids)
    return {d.id: _as_hit(d, distance=distance) for d in docs}


async def get_by_filter(filters: dict | None, *, distance: float = 999.0) -> dict[str, Hit]:
    """Fetch every chunk matching a filter. Scrolls internally — safe on large sets,
    but the caller is responsible for not asking for the whole corpus."""
    docs = await get_store().filter_documents_async(filters=filters)
    return {d.id: _as_hit(d, distance=distance) for d in docs}


async def get_doc_chunks(doc_id: str, folder: str | None = None) -> dict[str, Hit]:
    return await get_by_filter(with_folder(eq("doc_id", doc_id), folder))


# ── writes ────────────────────────────────────────────────────────────────────

async def add(documents: list[Document]) -> int:
    """Upsert documents. OVERWRITE means re-ingesting the same file is idempotent."""
    if not documents:
        return 0
    return await get_store().write_documents_async(documents, policy=DuplicatePolicy.OVERWRITE)


async def delete_ids(ids: list[str]) -> None:
    if ids:
        await get_store().delete_documents_async(document_ids=ids)


async def delete_doc(doc_id: str) -> int:
    """Delete every chunk of a document. Returns the number removed."""
    store = get_store()
    filters = eq("doc_id", doc_id)
    n = await store.count_documents_by_filter_async(filters=filters)
    if n:
        await store.delete_by_filter_async(filters=filters)
    return n


async def delete_doc_text_chunks(doc_id: str) -> None:
    """Delete a document's text chunks while preserving OCR-derived image chunks.

    Re-ingesting a file replaces its text but must not discard image chunks, which
    are written later by a background OCR task and would otherwise be orphaned.
    """
    await get_store().delete_by_filter_async(
        filters=and_(
            eq("doc_id", doc_id),
            {"operator": "!=", "field": "meta.chunk_type", "value": "image"},
        )
    )


async def move_doc(doc_id: str, new_folder: str) -> int:
    """Reassign a document's folder in place — no re-embedding, no rewrite."""
    store = get_store()
    filters = eq("doc_id", doc_id)
    n = await store.count_documents_by_filter_async(filters=filters)
    if n:
        await store.update_by_filter_async(filters=filters, meta={"folder": new_folder})
    return n


async def clear() -> int:
    """Drop and recreate the collection. Returns the number of chunks removed."""
    store = get_store()
    n = await store.count_documents_async()
    await store.delete_all_documents_async(recreate_index=True)
    return n


# ── document-level views ──────────────────────────────────────────────────────

async def doc_summaries() -> list[dict]:
    """One record per ingested document: doc_id, filename, folder, chunk count.

    Built from the summary chunk (`{doc_id}_1_-1`) plus a per-doc chunk count,
    rather than by scanning every chunk's metadata as the Chroma version did. On
    a corpus this size that scan pulled the entire payload set into memory on
    every cache miss.
    """
    store = get_store()
    doc_ids, _ = await store.get_metadata_field_unique_values_async(
        metadata_field="meta.doc_id", size=100_000
    )
    if not doc_ids:
        return []
    counts = await store.count_unique_metadata_by_filter_async(
        filters=in_("doc_id", list(doc_ids)), metadata_fields=["meta.doc_id"]
    )
    heads = await get_by_filter(in_("chunk_index", [-1, 0]))

    by_doc: dict[str, dict] = {}
    for _, meta, _dist in heads.values():
        did = meta.get("doc_id")
        if did and did not in by_doc:
            by_doc[did] = {
                "doc_id": did,
                "filename": meta.get("filename", ""),
                "folder": meta.get("folder", "General"),
                "doc_number": meta.get("doc_number", "—"),
                "revision": meta.get("revision", "—"),
                "chunks": counts.get(did, 0),
            }
    return list(by_doc.values())


async def folders() -> list[str]:
    values, _ = await get_store().get_metadata_field_unique_values_async(
        metadata_field="meta.folder", size=1000
    )
    return sorted(v for v in values if v)


async def health() -> dict:
    """Liveness + size, for /health. Never raises — reports the failure instead."""
    try:
        return {"reachable": True, "chunks_stored": await count()}
    except Exception as exc:  # surfaced to the user, not swallowed
        log.warning("qdrant health check failed: %s", exc)
        return {"reachable": False, "chunks_stored": 0, "error": str(exc)}


async def aclose() -> None:
    global _store
    if _store is not None:
        await _store.close_async()
        _store = None
