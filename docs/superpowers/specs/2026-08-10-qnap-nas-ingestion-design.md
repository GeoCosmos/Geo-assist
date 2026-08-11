# QNAP NAS Document Ingestion — Design

**Date:** 2026-08-10
**Status:** Approved, pending two deployment facts (see Open Questions)

## Problem

Geo-Assist ingests documents one way: a user drags files into the browser UI. The
company's real document repository lives on a QNAP NAS with thousands of files.
Getting that corpus into the index currently means manually copying batches into
`data/to_ingest/` and dragging them into the browser — which does not scale, gives
no record of what has already been loaded, and has to be redone from scratch every
time the NAS gains new documents.

We want to point Geo-Assist at a NAS folder and have it ingest what it has not
already seen, repeatably, without re-reading the whole share each time.

## Constraints

1. **The NAS is read-only.** It is the company's live document repository. Nothing
   is written to it — no `processed/` folders, no sidecar state files, no moves.
   All state stays local under `data/`. This is enforced structurally by mounting
   the share `:ro` into the container, not merely by convention in code.
2. **Air-gap holds.** LAN access to the QNAP introduces no internet egress. All
   network I/O to the NAS happens at the OS mount layer, outside the Python
   process, so no SMB client library or stored credentials enter the app.
3. **The server stays up during ingestion.** BM25 lives in-process
   (`bm25_index.py`), loaded at startup and persisted to `data/bm25_index.pkl` via
   atomic replace. A second process writing that pickle while the server holds a
   stale in-memory copy is last-writer-wins data loss. Therefore NAS ingestion runs
   **inside the server process**, as a route — never as a standalone script.
4. **Thousands of files.** Memory, incremental re-scan, and resumability are
   load-bearing, not nice-to-have.

## Approach

A directory walker plus a local manifest, driven by a new route that reuses the
existing background-job machinery (`jobs.py`, `ingest_many_tracked`,
`/ingest/status/{job_id}`) and the progress UI already built for bulk upload.

Two approaches were rejected:

- **A client script POSTing to `/ingest/bulk`.** Zero new server code, but it reads
  every byte of every file on every run and pushes it through HTTP multipart, with
  nowhere natural to keep the manifest.
- **A standalone script importing `ingest` directly.** Fastest to write, but only
  correct with the server stopped, per constraint 3. "Remember to shut down
  Geo-Assist first" is a rule that gets forgotten once and silently corrupts the
  keyword index.

## Architecture

### `nas.py` (new)

The walker and its policy. No knowledge of ingestion or HTTP.

- `resolve_subpath(subpath: str) -> Path` — joins a **relative** subpath onto
  `config.NAS_ROOT`, resolves it, and raises unless the result is contained under
  the root. The route never accepts absolute paths; the mount point is fixed by
  the container, so a relative subpath (empty string = whole share) is the entire
  input surface. Walks use `followlinks=False`.
- `scan(root: Path) -> Iterator[NasFile]` — recursive walk yielding
  `NasFile(abs_path, relpath, size, mtime, folder, ext)`.
- `folder_for(relpath)` — maps the containing directory to the existing `folder`
  metadata field. `Manuals/Avionics/x.pdf` → folder `Manuals/Avionics`; a file at
  the scan root → `General`.
- Exclusion policy, applied to directory and file names before `stat`:
  `@eaDir` and `@Recycle` (QNAP scatters these through every share), `.DS_Store`,
  `Thumbs.db`, `~$*` Office lock files, and any dotfile or dot-directory.
- Extension allowlist: the five `ingest.extract_pages` supports — `.pdf`, `.docx`,
  `.pptx`, `.txt`, `.csv`. Everything else (`.xlsx`, `.msg`, `.dwg`, …) is counted
  and reported, never raised.

### `nas_manifest.py` (new)

SQLite at `data/nas_manifest.db` via stdlib `sqlite3` — no new dependency.

```sql
CREATE TABLE IF NOT EXISTS seen (
  relpath    TEXT PRIMARY KEY,
  size       INTEGER NOT NULL,
  mtime      REAL    NOT NULL,
  doc_id     TEXT,
  status     TEXT    NOT NULL,   -- ok | empty | unsupported | error
  scanned_at REAL    NOT NULL
);
```

A file whose `(size, mtime)` match its row is skipped **without being read**. That
is what makes a re-scan cheap over SMB — the alternative is hashing every byte of
every file on every run, since `doc_id` is a content hash. Rows are committed as
each write batch flushes, so an interrupted job resumes by simply re-running.

The manifest is a cache, not a source of truth: deleting the DB causes a full
re-scan that re-derives the same state (files already in the store are skipped by
`doc_id`), it does not cause duplication.

### `ingest.py` (modified)

`_run_batch` currently takes `list[tuple[bytes, str]]` — every file's bytes
resident at once. That is fine for a browser upload and fatal for thousands of NAS
files. It is generalized to take `list[tuple[loader, filename]]` where `loader` is
a zero-argument callable returning bytes, invoked **inside** the existing
`PREPARE_CONCURRENCY` semaphore so at most that many file bodies are ever in
memory. Existing callers pass a trivial loader closing over their bytes; behaviour
is unchanged for them.

New `ingest_paths_tracked(items, job, ...)` is the NAS-side entry point, mirroring
`ingest_many_tracked`.

**Skip-if-known.** `doc_id` is `sha256(content)[:16]`, so a file already uploaded
through the UI into folder `Avionics` that also lives on the NAS hashes identically.
Without a guard, `_write` calls `delete_doc_text_chunks` and rewrites it with the
NAS-derived folder — silent reclassification of a document the user filed by hand.
NAS ingestion skips any `doc_id` already present in the store unless the request
sets `reingest: true`.

### `main.py` (modified)

- `GET /ingest/nas/health` — root exists, is a mountpoint, is readable, is
  non-empty. Distinguishes "permission denied" from "no files found", which are
  otherwise indistinguishable and are the two most likely deploy-day failures
  (see Deployment). Also returns the root's immediate subdirectories, which is what
  populates the UI's subfolder picker — this is a live check, so a mount that
  appears after server startup is picked up without a restart.
- `POST /ingest/nas/preview` — `{subpath, reingest}` → counts only, no ingestion:
  `{new, unchanged, unsupported, oversized, total_bytes}`. At this scale a dry run
  is essential; the user should see "1,247 new, 3,010 unchanged, 88 unsupported"
  before committing to an embed run measured in hours.
- `POST /ingest/nas/scan` — same body → `{job_id, total, skipped}`, polled through
  the existing `/ingest/status/{job_id}`. No new status endpoint.

Both scanning endpoints run the walk in a thread executor: `stat` over SMB blocks,
and blocking the event loop stalls every concurrent chat request.

### `config.py` (modified)

```python
NAS_ROOT          = os.environ.get("GEO_NAS_ROOT", "/mnt/nas")
NAS_MANIFEST_PATH = os.path.join(DATA_DIR, "nas_manifest.db")
```

Whether a NAS is actually present is deliberately **not** a config constant. An
`os.path.isdir()` evaluated at import time goes stale the moment the mount drops or
appears, and a container that starts before the host mount is ready would disable
the feature until someone restarted it. `GET /ingest/nas/health` performs the check
live on each call, and the UI shows or hides the control based on that response.

### `static/index.html` (modified)

A "Scan NAS folder" control beside the existing upload: a subfolder picker
populated from `/ingest/nas/health` (not a free-text path box — the root is fixed),
a Preview button showing the dry-run counts, and a Start button that hands off to
the progress polling already wired up for bulk upload. The control is hidden when
the health check reports no usable NAS root.

## Data flow

```
POST /ingest/nas/scan {subpath}
  → resolve_subpath()               reject anything escaping NAS_ROOT
  → nas.scan() in thread executor   walk, exclude junk, filter extensions, stat
  → manifest diff                   drop files whose (size, mtime) are unchanged
  → jobs.create(total=len(new))
  → ingest_paths_tracked()          background task
      per file, under semaphore:
        read bytes
        skip if doc_id already in store (unless reingest)
        _prepare()  →  parse, chunk, embed
      flush to Qdrant + BM25 every 25 files
      record manifest rows for the flushed batch
  → client polls /ingest/status/{job_id}
```

## Error handling

Per-file failures land in `job.errors` and the run continues, matching what
`_run_batch` already does. Three additions:

- **Capped error list.** ~200 entries plus summary counts by category. A share with
  4,000 `.xlsx` files must not produce a 4,000-line status response.
- **Mount-loss detection.** Consecutive I/O errors (`ENOENT`, `ESTALE`, `EACCES`)
  past a small threshold abort the job with "NAS unreachable" rather than grinding
  through thousands of failures after the mount drops mid-scan.
- **Oversized files** (>100 MB, matching the existing upload cap) are skipped and
  reported, not attempted.

## Deployment

Three services on the VM under `/opt/geo-assist/docker-compose.yml`, project
`geo-assist`: `geo-assist-ollama`, `geo-assist-qdrant`, `geo-assist-app`. The app is
built from source (`build: ./app`), so **shipping `nas.py` is an image rebuild**
(`docker compose build app && docker compose up -d app`), not a file copy.

The host has two CIFS mounts, both `rw`, both `soft` (so reads fail with `EIO`
rather than hanging when the NAS goes away — which is what the mount-loss detection
in Error Handling relies on):

| | Mount 1 | Mount 2 |
|---|---|---|
| Share | `primary document share` | `secondary share` |
| Host path | `$NAS_MOUNT_A` | `$NAS_MOUNT_B` |
| Ownership | `uid=0,gid=0` (not forced) | `uid=1000,gid=1000` forced |
| SMB | 3.0 | 3.1.1 |

Compose changes required:

- **Fix `data/` persistence first — this is a pre-existing bug, not a new
  requirement.** `config.DATA_DIR` resolves to `/app/data`, but the app service
  mounts only `./data/uploads:/app/uploads`, a path no code references. `/app/data`
  is therefore ephemeral container filesystem: every rebuild destroys
  `bm25_index.pkl` (self-heals via `load_or_rebuild()`, at the cost of a full re-fit
  each boot), `originals/` (**unrecoverable** — citation file links 404 forever),
  and `images/`. Qdrant survives only because it is a separate container with its
  own bind. Add `- ./data:/app/data` to the app service. Without it, the NAS
  manifest is wiped on every deploy and each scan re-reads the entire share.
- **Bind the document share read-only** at `/app/documents`, with
  `GEO_NAS_ROOT=/app/documents`. An unused `a single-project bind under /app/documents/`
  bind already exists — a half-finished attempt at this feature, since no code reads
  that path. It is replaced by a read-only bind of the share root. `:ro` is what
  makes constraint 1 structural; the underlying CIFS mounts are `rw`.
- **`bind-propagation=rslave`** on that mount. Docker bind mounts are point-in-time:
  if the host CIFS mount drops and remounts, the container keeps pointing at the
  stale mountpoint and the scan sees an empty directory — reporting "0 new,
  everything unchanged" rather than an error. `rslave` propagates the remount.
- **Container UID must be able to read the mount.** Mount 2 forces `uid=1000` with
  `file_mode=0755`; mount 1 is `uid=0`. The app service declares no `user:`, so it
  runs as whatever the Dockerfile sets — root unless stated otherwise, which reads
  both. If a `USER` directive is added later, it must match. `EACCES` on every read
  is the most likely deploy-day failure, and `/ingest/nas/health` exists to name it
  precisely rather than presenting it as an empty share.

## Testing

`tests/test_nas.py`, against a fake NAS tree in `tmp_path`, using the existing
mocked-embedding fixtures:

- Junk exclusion: `@eaDir`, `@Recycle`, `Thumbs.db`, `.DS_Store`, `~$doc.docx`,
  dotfiles and dot-directories.
- Folder derivation from relpath, including files at the scan root → `General`.
- Containment: `../` traversal, absolute paths, and symlinks pointing outside the
  root are all rejected.
- Manifest: unchanged `(size, mtime)` is skipped without the file being read
  (assert via a loader that raises if called); changed mtime is re-ingested.
- Unsupported extensions are counted and reported, never raised.
- Skip-if-known: a doc_id already in the store is not rewritten, and its existing
  `folder` metadata survives the scan; `reingest: true` overrides.
- Memory: `_run_batch` invokes loaders lazily — assert at most
  `PREPARE_CONCURRENCY` loaders have been called concurrently.

`tests/test_api.py` additions: out-of-root subpath → 403; preview returns counts
without ingesting; health distinguishes missing root, unreadable root, and empty
root.

**One field validation before trusting the manifest.** CIFS mtime is usually stable
but not universally. On the VM, `stat` the same unmodified file twice and across two
scans; if mtime is unstable there, the manifest silently re-ingests the entire
corpus every run and the only symptom is wall-clock hours. Cheap to check once,
expensive to miss.

## Out of scope

No filesystem watcher, no scheduling, no deletion propagation. Scans are manually
triggered. If recurring scans are wanted later, a cron-invoked call to the existing
route beats a watcher over SMB.

## Open questions

1. **Which share holds the document repository.** `$NAS_MOUNT_B`
   (mount 2) is 50 MB and contains exactly one file of a supported type, so it is
   not the corpus despite being the compose-adjacent mount. Mount 1 (`$NAS_MOUNT_A`)
   is the likely repository and is not currently bound into any container. Blocks
   implementation: the wrong answer means ingesting the wrong company share.
Resolved:

- **Originals storage — keep the copy.** The VM has 74 GB free of 193 GB and the
  share that could be sized is 50 MB, so duplicating originals into `data/` is not
  a meaningful cost. This is the existing `_prepare` behaviour and needs no code.
- **Deployment is an image rebuild.** The app's code is baked in (`build: ./app`).
  Recorded in Deployment.
- **`data/` persistence is a pre-existing bug**, not a consequence of this feature,
  and is fixed first (plan Task 0) so the rebuild at the end of the work is safe.
