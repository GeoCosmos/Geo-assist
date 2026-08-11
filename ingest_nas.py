"""
Drives ingestion from a mounted NAS share.

Composes three pieces that each stay ignorant of the others: `nas` decides what
is a document, `nas_manifest` remembers what has been seen, and
`ingest._run_batch` does the work. Kept out of ingest.py, which is long enough.

Runs in the server process by necessity, not preference: the BM25 index is
in-memory state persisted to a pickle, so a second process writing it while the
server holds a stale copy is last-writer-wins data loss.
"""
import asyncio
import logging
import os

import config
import ingest
import nas
import store
from nas_manifest import Manifest

log = logging.getLogger(__name__)

_manifest_singleton: Manifest | None = None


class NasUnreachable(Exception):
    """The share is missing, unreadable, or dropped out mid-scan."""


def _manifest() -> Manifest:
    global _manifest_singleton
    path = config.NAS_MANIFEST_PATH
    if _manifest_singleton is None or getattr(_manifest_singleton, "_path", None) != path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        _manifest_singleton = Manifest(path)
    return _manifest_singleton


async def _walk(subpath: str) -> tuple[list, dict]:
    """Resolve, contain, and walk — off the event loop."""
    target = nas.resolve_subpath(subpath)  # raises ValueError on escape
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, nas.scan, target)
    except FileNotFoundError as exc:
        raise NasUnreachable(str(exc)) from exc
    except OSError as exc:
        raise NasUnreachable(f"NAS unreachable: {exc}") from exc


def _split_new(files: list, reingest: bool) -> tuple[list, int]:
    """Partition into (to_ingest, unchanged_count) using the manifest."""
    if reingest:
        return list(files), 0
    manifest = _manifest()
    fresh, unchanged = [], 0
    for f in files:
        if manifest.known(f.relpath, f.size, f.mtime):
            unchanged += 1
        else:
            fresh.append(f)
    return fresh, unchanged


async def preview(subpath: str, reingest: bool = False) -> dict:
    """Dry run: what a scan would do, without reading or ingesting anything.

    `new` means "not in the manifest". Some of those may turn out to be content
    the store already holds (the same document filed under two NAS paths); those
    surface as `skipped` in the job results, because detecting them requires
    hashing the file, which is exactly the cost the manifest exists to avoid.
    """
    files, counts = await _walk(subpath)
    fresh, unchanged = _split_new(files, reingest)
    return {
        "subpath": subpath,
        "new": len(fresh),
        "unchanged": unchanged,
        "unsupported": counts["unsupported"],
        "oversized": counts["oversized"],
        "excluded": counts["excluded"],
        "total_bytes": sum(f.size for f in fresh),
    }


def _append_capped(target: list[str], new: list[str]) -> None:
    """Add errors up to the cap, then a single summary line.

    A share holding thousands of unreadable files must not produce a status
    response with one line per file.
    """
    room = max(config.NAS_MAX_ERRORS - len(target), 0)
    target.extend(new[:room])
    hidden = len(new) - room
    if hidden > 0:
        target.append(f"… and {hidden} more error(s) not shown")


async def scan_and_ingest(subpath: str, job, reingest: bool = False) -> None:
    """Walk the share and ingest everything the manifest has not already seen."""
    try:
        files, counts = await _walk(subpath)
    except (NasUnreachable, ValueError) as exc:
        job.status = "failed"
        job.errors.append(str(exc))
        return

    fresh, _unchanged = _split_new(files, reingest)
    job.total = len(fresh)
    if counts["unsupported"]:
        job.errors.append(f"{counts['unsupported']} file(s) skipped: unsupported type")
    if counts["oversized"]:
        job.errors.append(f"{counts['oversized']} file(s) skipped: over 100 MB")

    if not fresh:
        job.status = "done"
        return

    by_key = {f.relpath: f for f in fresh}
    manifest = _manifest()

    # Snapshot the doc_ids already in the store so duplicate content is skipped
    # before it is parsed or embedded, rather than being rewritten under a
    # NAS-derived folder over whatever the user filed it under by hand.
    try:
        known_doc_ids = {d["doc_id"] for d in await store.doc_summaries()}
    except Exception:
        log.warning("could not snapshot existing doc_ids; duplicates may be rewritten",
                    exc_info=True)
        known_doc_ids = set()

    io_errors = 0

    def _loader_for(path: str):
        def load() -> bytes:
            nonlocal io_errors
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                # The CIFS mounts use `soft`, so a vanished share fails fast with
                # EIO on every read rather than hanging. Without a circuit
                # breaker the job would grind through thousands of guaranteed
                # failures before reporting anything.
                #
                # Loaders run in executor threads, so this is a rough failure
                # density rather than a strict consecutive count. That is enough
                # for the case it exists for: when the mount is gone every read
                # fails and nothing ever resets it.
                io_errors += 1
                if io_errors >= config.NAS_IO_ERROR_LIMIT:
                    raise NasUnreachable(
                        f"NAS unreachable — {io_errors} read failures, mount likely gone"
                    )
                raise
            io_errors = 0
            return data
        return load

    # One batch across every folder: BatchItem carries its own folder, so there is
    # no positional correlation between two parallel lists to get wrong.
    items = [
        ingest.BatchItem(
            loader=_loader_for(f.abs_path),
            filename=os.path.basename(f.relpath),
            key=f.relpath,
            folder=f.folder,
        )
        for f in fresh
    ]

    def _tick() -> None:
        job.prepared += 1

    def _record(written: list[dict]) -> None:
        rows = []
        for result in written:
            src = by_key.get(result.get("key"))
            if src is None:
                continue
            rows.append((src.relpath, src.size, src.mtime,
                         result.get("doc_id"), result.get("status", "ok")))
            if result.get("doc_id"):
                known_doc_ids.add(result["doc_id"])
        if rows:
            manifest.record_many(rows)

    try:
        results, errors = await ingest._run_batch(
            items,
            on_prepared=_tick,
            on_flush=_record,
            skip_doc_ids=known_doc_ids,
            # Without this the breaker above is dead code: _run_batch's per-file
            # handler would swallow NasUnreachable as just another failed file
            # and the job would report "done" against a dead mount.
            fatal_exceptions=(NasUnreachable,),
        )
        job.results.extend(results)
        _append_capped(job.errors, errors)
        job.status = "done"
    except NasUnreachable as exc:
        job.status = "failed"
        job.errors.append(str(exc))
    except Exception as exc:
        log.exception("NAS scan job %s failed", job.id)
        job.status = "failed"
        job.errors.append(str(exc))
