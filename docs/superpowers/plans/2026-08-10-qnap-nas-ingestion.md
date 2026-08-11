# QNAP NAS Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Geo-Assist ingest documents from a mounted QNAP share on demand, skipping files it has already seen, without stopping the server.

**Architecture:** A directory walker (`nas.py`) plus a SQLite manifest (`nas_manifest.py`) feed the existing background-job machinery through new routes on `main.py`. `ingest._run_batch` is generalised from eager `bytes` to lazy loader callables so thousands of files never sit in memory at once. Everything runs inside the server process, because BM25 is in-process state and a second writer would silently clobber it.

**Tech Stack:** Python 3.10+, FastAPI, Haystack + Qdrant, stdlib `sqlite3` (no new dependency), pytest + pytest-asyncio.

## Global Constraints

- **No new pip dependencies.** The manifest uses stdlib `sqlite3`. `requirements.txt` is unchanged.
- **The NAS is never written to.** No moves, no sidecar files, no `processed/` dirs. Enforced structurally by a `:ro` bind mount; code must not rely on that alone.
- **Air-gap holds.** No new network calls. All NAS I/O goes through the OS mount layer.
- **The route never accepts an absolute path.** Only a relative subpath under `config.NAS_ROOT`, resolved and containment-checked.
- **Supported extensions** are exactly `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv` — matching `main.ALLOWED_EXTENSIONS` and `ingest.extract_pages`.
- **Max file size** is `main.MAX_UPLOAD_BYTES` (100 MB), matching the upload path.
- **Blocking filesystem work runs in an executor.** `stat` and `read` over CIFS block; the event loop serves concurrent chat requests.
- Tests run fully offline. Use the existing `fresh_store` and `mock_ollama` fixtures in `tests/conftest.py`.

---

### Task 0: Fix `data/` persistence on the VM (do this first)

A pre-existing data-loss bug, independent of this feature and currently active. Every image rebuild destroys `data/originals/`, which is not recoverable. Doing it first means the rebuild at the end of this plan is safe.

**Files:** `/opt/geo-assist/docker-compose.yml` on the VM (not in this repo yet — Task 7 adds a tracked reference copy).

- [ ] **Step 1: See what is currently unprotected**

```bash
sudo docker exec geo-assist-app sh -c 'ls -la /app/data; du -sh /app/data/* 2>/dev/null'
ls -la /opt/geo-assist/data
```

Anything present inside the container but absent on the host is living on ephemeral container filesystem right now.

- [ ] **Step 2: Rescue it, copying named paths only**

Do **not** recursively copy `/app/data/.` onto the host — `/opt/geo-assist/data/qdrant` is bind-mounted into the live Qdrant container, and a recursive copy would write into a running database.

```bash
sudo docker cp geo-assist-app:/app/data/originals     /opt/geo-assist/data/ 2>/dev/null || true
sudo docker cp geo-assist-app:/app/data/images        /opt/geo-assist/data/ 2>/dev/null || true
sudo docker cp geo-assist-app:/app/data/bm25_index.pkl /opt/geo-assist/data/ 2>/dev/null || true
sudo tar czf /opt/geo-assist/data-backup-$(date +%F).tar.gz -C /opt/geo-assist data
```

- [ ] **Step 3: Replace the useless bind with the real one**

In `/opt/geo-assist/docker-compose.yml`, under `app.volumes`, replace:

```yaml
      - ./data/uploads:/app/uploads
```

with:

```yaml
      - ./data:/app/data
```

`config.DATA_DIR` is `/app/data`; nothing in the codebase references `/app/uploads`.

- [ ] **Step 4: Restart and verify persistence**

```bash
cd /opt/geo-assist
sudo docker compose up -d
sudo docker exec geo-assist-app touch /app/data/persistence-probe
ls /opt/geo-assist/data/persistence-probe && echo "PERSISTED"
sudo docker compose up -d --force-recreate app
sudo docker exec geo-assist-app ls /app/data/persistence-probe && echo "SURVIVED RECREATE"
sudo docker exec geo-assist-app rm /app/data/persistence-probe
```

Expected: both `PERSISTED` and `SURVIVED RECREATE` print.

---

### Task 1: NAS walker (`nas.py`)

Pure filesystem policy — no ingestion, no HTTP, no manifest.

**Files:**
- Create: `nas.py`
- Modify: `config.py` (append NAS settings after the Qdrant block, before `API_PORT`)
- Test: `tests/test_nas.py`

**Interfaces:**
- Consumes: `config.NAS_ROOT`
- Produces:
  - `NasFile` dataclass with fields `abs_path: str`, `relpath: str`, `size: int`, `mtime: float`, `folder: str`, `ext: str`
  - `resolve_subpath(subpath: str) -> Path` — raises `ValueError` on escape
  - `scan(root: Path, base: Path | None = None) -> tuple[list[NasFile], dict[str, int]]` — returns files and a counts dict with keys `unsupported`, `oversized`, `excluded`. `base` is what relpaths are measured against and defaults to `root`; callers scanning a subfolder must pass the share root, or a file's manifest key and folder change with the scan scope.
  - `folder_for(relpath: str) -> str`
  - `SUPPORTED_EXTS: set[str]`

- [ ] **Step 1: Add config settings**

Append to `config.py` immediately after the `QDRANT_WRITE_BATCH` line:

```python
# ── NAS ingestion ─────────────────────────────────────────────────────────────
# Root of the mounted document share, as seen *inside* this container. The host
# CIFS mount is bind-mounted here read-only (see docker-compose.yml), so the app
# never has a write path to the NAS regardless of what the code does.
#
# Whether a NAS is actually present is deliberately not a constant here: an
# os.path.isdir() evaluated at import time goes stale the moment the mount drops
# or appears, and a container that starts before the host mount is ready would
# disable the feature until someone restarted it. GET /ingest/nas/health checks
# live on each call.
NAS_ROOT = os.environ.get("GEO_NAS_ROOT", "/app/documents")
NAS_MANIFEST_PATH = os.path.join(DATA_DIR, "nas_manifest.db")
# Consecutive I/O errors during a scan before the job aborts with "NAS
# unreachable". The CIFS mounts use `soft`, so a dropped mount surfaces as EIO on
# every read rather than hanging — without this the job would grind through
# thousands of guaranteed failures.
NAS_IO_ERROR_LIMIT = int(os.environ.get("GEO_NAS_IO_ERROR_LIMIT", "10"))
# Cap on entries in a job's error list. A share with 4,000 unsupported files must
# not produce a 4,000-line status response; counts still reflect the true total.
NAS_MAX_ERRORS = 200
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_nas.py`:

```python
import os
from pathlib import Path

import pytest

import config
import nas


@pytest.fixture
def fake_nas(tmp_path, monkeypatch):
    """A NAS tree with real documents, QNAP junk, and unsupported files."""
    root = tmp_path / "documents"
    (root / "Manuals" / "Avionics").mkdir(parents=True)
    (root / "@eaDir").mkdir()
    (root / "@Recycle").mkdir()
    (root / "Manuals" / "@eaDir").mkdir()
    (root / ".hidden").mkdir()

    (root / "top.pdf").write_bytes(b"top")
    (root / "Manuals" / "guide.docx").write_bytes(b"guide")
    (root / "Manuals" / "Avionics" / "spec.pdf").write_bytes(b"spec")
    (root / "Manuals" / "sheet.xlsx").write_bytes(b"nope")
    (root / "@eaDir" / "thumb.pdf").write_bytes(b"junk")
    (root / "Manuals" / "@eaDir" / "thumb2.pdf").write_bytes(b"junk")
    (root / "@Recycle" / "deleted.pdf").write_bytes(b"junk")
    (root / "Thumbs.db").write_bytes(b"junk")
    (root / ".DS_Store").write_bytes(b"junk")
    (root / "~$draft.docx").write_bytes(b"lock")
    (root / ".hidden" / "secret.pdf").write_bytes(b"junk")

    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    return root


def test_scan_finds_only_supported_documents(fake_nas):
    files, counts = nas.scan(fake_nas)
    assert sorted(f.relpath for f in files) == [
        "Manuals/Avionics/spec.pdf",
        "Manuals/guide.docx",
        "top.pdf",
    ]


def test_scan_counts_unsupported_without_raising(fake_nas):
    files, counts = nas.scan(fake_nas)
    assert counts["unsupported"] == 1  # sheet.xlsx; junk is excluded, not counted


def test_scan_skips_oversized_files(fake_nas, monkeypatch):
    monkeypatch.setattr(nas, "MAX_FILE_BYTES", 3)
    files, counts = nas.scan(fake_nas)
    # "guide" and "spec" are 5 and 4 bytes; "top" is 3 and survives.
    assert [f.relpath for f in files] == ["top.pdf"]
    assert counts["oversized"] == 2


def test_folder_derives_from_directory(fake_nas):
    files, _ = nas.scan(fake_nas)
    by_path = {f.relpath: f.folder for f in files}
    assert by_path["top.pdf"] == "General"
    assert by_path["Manuals/guide.docx"] == "Manuals"
    assert by_path["Manuals/Avionics/spec.pdf"] == "Manuals/Avionics"


def test_scan_records_size_and_mtime(fake_nas):
    files, _ = nas.scan(fake_nas)
    top = next(f for f in files if f.relpath == "top.pdf")
    assert top.size == 3
    assert top.mtime > 0


def test_resolve_subpath_accepts_empty_and_subdir(fake_nas):
    assert nas.resolve_subpath("") == Path(fake_nas).resolve()
    assert nas.resolve_subpath("Manuals") == (Path(fake_nas) / "Manuals").resolve()


@pytest.mark.parametrize("bad", ["..", "../etc", "Manuals/../..", "/etc", "/etc/passwd"])
def test_resolve_subpath_rejects_escapes(fake_nas, bad):
    with pytest.raises(ValueError):
        nas.resolve_subpath(bad)


def test_resolve_subpath_rejects_symlink_escape(fake_nas, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (fake_nas / "escape").symlink_to(outside)
    with pytest.raises(ValueError):
        nas.resolve_subpath("escape")


def test_scan_does_not_follow_symlinked_dirs(fake_nas, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leaked.pdf").write_bytes(b"leaked")
    (fake_nas / "link").symlink_to(outside)
    files, _ = nas.scan(fake_nas)
    assert not any("leaked" in f.relpath for f in files)


def test_scan_missing_root_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        nas.scan(tmp_path / "nope")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_nas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nas'`

- [ ] **Step 4: Implement `nas.py`**

```python
"""
Walker for a mounted NAS document share.

Filesystem policy only — what counts as a document, what counts as junk, and
what counts as inside the share. Knows nothing about ingestion, HTTP, or the
manifest, so it can be tested against a plain tmp_path tree.
"""
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

import config

SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".csv"}
MAX_FILE_BYTES = 100 * 1024 * 1024  # mirrors main.MAX_UPLOAD_BYTES

# QNAP scatters @eaDir (thumbnail/metadata sidecars) through every share, one per
# directory containing media, each holding copies of the originals' names. Left
# in, they would double the corpus with unreadable stubs. @Recycle is the NAS
# trash. The rest is Windows/macOS/Office debris.
EXCLUDED_DIRS = {"@eaDir", "@Recycle", "#recycle", "$RECYCLE.BIN", "System Volume Information"}
EXCLUDED_FILE_PATTERNS = ("~$*", "Thumbs.db", "desktop.ini", ".DS_Store")


@dataclass(frozen=True)
class NasFile:
    abs_path: str
    relpath: str      # POSIX-style, relative to the scan root
    size: int
    mtime: float
    folder: str       # derived from relpath's parent; "General" at the root
    ext: str


def resolve_subpath(subpath: str) -> Path:
    """Resolve a *relative* subpath under NAS_ROOT.

    Absolute paths are rejected outright rather than silently reinterpreted: the
    mount point is fixed by the container, so the only legitimate input is a
    subfolder name. Resolution happens before the containment check so symlinks
    pointing outside the share are caught too.
    """
    root = Path(config.NAS_ROOT).resolve()
    cleaned = (subpath or "").strip().lstrip("/")
    if os.path.isabs(subpath or ""):
        raise ValueError("absolute paths are not accepted")
    target = (root / cleaned).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes the NAS root: {subpath!r}")
    return target


def folder_for(relpath: str) -> str:
    """Map a file's directory onto the existing `folder` metadata field."""
    parent = str(Path(relpath).parent).replace(os.sep, "/")
    if parent in (".", "", "/"):
        return "General"
    return parent[:64]


def _excluded_file(name: str) -> bool:
    if name.startswith("."):
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDED_FILE_PATTERNS)


def scan(root: Path) -> tuple[list[NasFile], dict[str, int]]:
    """Walk `root`, returning ingestable files and counts of what was passed over.

    Blocking: `stat` over CIFS is a network round trip. Callers must run this in
    an executor, never on the event loop.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"NAS path is not a directory: {root}")

    files: list[NasFile] = []
    counts = {"unsupported": 0, "oversized": 0, "excluded": 0}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune in place so os.walk never descends into junk directories.
        keep = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        counts["excluded"] += len(dirnames) - len(keep)
        dirnames[:] = keep

        for name in filenames:
            if _excluded_file(name):
                counts["excluded"] += 1
                continue
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                counts["unsupported"] += 1
                continue
            abs_path = os.path.join(dirpath, name)
            try:
                st = os.stat(abs_path)
            except OSError:
                # A file that vanished or is unreadable is not fatal to the walk.
                counts["excluded"] += 1
                continue
            if st.st_size > MAX_FILE_BYTES:
                counts["oversized"] += 1
                continue
            relpath = os.path.relpath(abs_path, root).replace(os.sep, "/")
            files.append(NasFile(
                abs_path=abs_path,
                relpath=relpath,
                size=st.st_size,
                mtime=st.st_mtime,
                folder=folder_for(relpath),
                ext=ext,
            ))

    return files, counts
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_nas.py -v`
Expected: PASS, 10 tests

- [ ] **Step 6: Commit**

The working tree has a pre-existing uncommitted change to `config.py` (`OLLAMA_BASE`
now reads `GEO_OLLAMA_BASE`, which is what lets the container reach the `ollama`
service). It belongs to the Docker migration, not to this feature — commit it
separately first so this commit does not silently absorb it:

```bash
git add -p config.py    # stage only the OLLAMA_BASE hunk
git commit -m "fix: read Ollama base URL from GEO_OLLAMA_BASE for container deploys"

git add nas.py tests/test_nas.py config.py
git commit -m "feat: add NAS share walker with QNAP junk exclusion and path containment"
```

---

### Task 2: Scan manifest (`nas_manifest.py`)

**Files:**
- Create: `nas_manifest.py`
- Test: `tests/test_nas_manifest.py`

**Interfaces:**
- Consumes: `config.NAS_MANIFEST_PATH`, `nas.NasFile`
- Produces:
  - `Manifest(path: str)` with methods `known(relpath, size, mtime) -> bool`, `record(relpath, size, mtime, doc_id, status) -> None`, `record_many(rows) -> None`, `forget(relpath) -> None`, `count() -> int`, `close() -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nas_manifest.py`:

```python
import pytest

from nas_manifest import Manifest


@pytest.fixture
def manifest(tmp_path):
    m = Manifest(str(tmp_path / "manifest.db"))
    yield m
    m.close()


def test_unknown_file_is_not_known(manifest):
    assert manifest.known("a.pdf", 100, 1000.0) is False


def test_recorded_file_is_known(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    assert manifest.known("a.pdf", 100, 1000.0) is True


def test_changed_mtime_is_not_known(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    assert manifest.known("a.pdf", 100, 2000.0) is False


def test_changed_size_is_not_known(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    assert manifest.known("a.pdf", 200, 1000.0) is False


def test_record_replaces_previous_row(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    manifest.record("a.pdf", 200, 2000.0, "def456", "ok")
    assert manifest.count() == 1
    assert manifest.known("a.pdf", 200, 2000.0) is True


def test_errored_file_is_known_so_it_is_not_retried_forever(manifest):
    manifest.record("bad.pdf", 100, 1000.0, None, "error")
    assert manifest.known("bad.pdf", 100, 1000.0) is True


def test_forget_makes_file_unknown_again(manifest):
    manifest.record("a.pdf", 100, 1000.0, "abc123", "ok")
    manifest.forget("a.pdf")
    assert manifest.known("a.pdf", 100, 1000.0) is False


def test_record_many_is_atomic_batch(manifest):
    manifest.record_many([
        ("a.pdf", 1, 1.0, "id1", "ok"),
        ("b.pdf", 2, 2.0, "id2", "ok"),
    ])
    assert manifest.count() == 2


def test_manifest_persists_across_instances(tmp_path):
    path = str(tmp_path / "m.db")
    m1 = Manifest(path)
    m1.record("a.pdf", 100, 1000.0, "abc123", "ok")
    m1.close()
    m2 = Manifest(path)
    assert m2.known("a.pdf", 100, 1000.0) is True
    m2.close()


def test_mtime_comparison_tolerates_float_noise(manifest):
    """CIFS can return mtimes differing in the sub-microsecond digits between
    stats of an unmodified file. Exact float equality would re-ingest the whole
    corpus every scan, with no symptom except hours of wasted embedding."""
    manifest.record("a.pdf", 100, 1000.0000001, "abc123", "ok")
    assert manifest.known("a.pdf", 100, 1000.0000002) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_nas_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nas_manifest'`

- [ ] **Step 3: Implement `nas_manifest.py`**

```python
"""
Local record of which NAS files have already been ingested.

Purely a cache, never a source of truth. Deleting the database causes a full
re-scan that re-derives the same state — files already in the store are skipped
by content hash — it does not cause duplication.

Its job is to make a re-scan cheap. `doc_id` is a hash of file *content*, so
without this the only way to know whether a file has been seen is to read every
byte of it. Over CIFS, on thousands of files, that is the difference between a
scan measured in seconds and one measured in hours.
"""
import sqlite3
import time

# CIFS timestamps can jitter in the low-order digits between stats of an
# unmodified file. Comparing mtimes for exact equality would mark the entire
# corpus as changed on every scan, whose only symptom is a very slow re-ingest.
MTIME_TOLERANCE = 1e-3


class Manifest:
    def __init__(self, path: str):
        # check_same_thread=False: the walk and the ingest loop touch this from
        # different executor threads, serialised by the ingest lock upstream.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS seen (
              relpath    TEXT PRIMARY KEY,
              size       INTEGER NOT NULL,
              mtime      REAL    NOT NULL,
              doc_id     TEXT,
              status     TEXT    NOT NULL,
              scanned_at REAL    NOT NULL
            )
        """)
        self._conn.commit()

    def known(self, relpath: str, size: int, mtime: float) -> bool:
        """True if this exact file (path, size, mtime) has been processed before.

        Files recorded with status="error" count as known: a permanently corrupt
        PDF should not be re-attempted on every scan. Editing the file changes
        its mtime and makes it unknown again.
        """
        row = self._conn.execute(
            "SELECT size, mtime FROM seen WHERE relpath = ?", (relpath,)
        ).fetchone()
        if row is None:
            return False
        return row[0] == size and abs(row[1] - mtime) <= MTIME_TOLERANCE

    def record(self, relpath: str, size: int, mtime: float, doc_id: str | None, status: str) -> None:
        self.record_many([(relpath, size, mtime, doc_id, status)])

    def record_many(self, rows: list[tuple]) -> None:
        """Commit a batch of rows in one transaction.

        Called after each write flush, which is what makes an interrupted job
        resumable: re-running the scan skips whatever already committed.
        """
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO seen "
            "(relpath, size, mtime, doc_id, status, scanned_at) VALUES (?,?,?,?,?,?)",
            [(r[0], r[1], r[2], r[3], r[4], now) for r in rows],
        )
        self._conn.commit()

    def forget(self, relpath: str) -> None:
        self._conn.execute("DELETE FROM seen WHERE relpath = ?", (relpath,))
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_nas_manifest.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add nas_manifest.py tests/test_nas_manifest.py
git commit -m "feat: add SQLite manifest for incremental NAS scans"
```

---

### Task 3: Lazy-loader refactor of `ingest._run_batch`

The existing signature holds every file's bytes in memory simultaneously. Fine for a browser upload, fatal for thousands of NAS files.

**Files:**
- Modify: `ingest.py:908-962` (`_run_batch`, `ingest_many`, `ingest_many_tracked`)
- Test: `tests/test_ingest.py` (append)

**Interfaces:**
- Produces:
  - `BatchItem` dataclass: `loader: Callable[[], bytes]`, `filename: str`, `key: Any`, `folder: str`. Carrying `folder` per item lets one `_run_batch` call span many NAS subfolders instead of one call per folder correlated by a positional `zip`.
  - `_run_batch(items: list[BatchItem], on_prepared=None, on_flush=None, skip_doc_ids: set[str] | None = None, fatal_exceptions: tuple = ()) -> tuple[list[dict], list[str]]`. `key` is opaque and echoed back on each result as `result["key"]`. Exceptions matching `fatal_exceptions` propagate instead of being converted to per-file errors.
  - `_bytes_loader(data: bytes) -> Callable[[], bytes]`
  - `_prepare(..., skip_doc_ids: set[str] | None = None)` returns `{"status": "skipped", ...}` before parsing when the content hash is already present

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
import asyncio

import pytest

import config
import ingest


@pytest.mark.asyncio
async def test_run_batch_invokes_loaders_lazily(mock_ollama):
    """Loaders must not all be called up front — that is the whole point of the
    refactor. With a concurrency of 2, no more than 2 may be in flight at once."""
    in_flight = 0
    peak = 0

    def make_loader(payload: bytes):
        def load():
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                return payload
            finally:
                in_flight -= 1
        return load

    items = [
        ingest.BatchItem(make_loader(f"doc {i}".encode()), f"f{i}.txt", f"key{i}", "General")
        for i in range(8)
    ]
    old = config.PREPARE_CONCURRENCY
    config.PREPARE_CONCURRENCY = 2
    try:
        results, errors = await ingest._run_batch(items)
    finally:
        config.PREPARE_CONCURRENCY = old

    assert errors == []
    assert len(results) == 8
    assert peak <= 2


@pytest.mark.asyncio
async def test_run_batch_echoes_correlation_key(mock_ollama):
    items = [ingest.BatchItem(
        ingest._bytes_loader(b"alpha content"), "a.txt", "Manuals/a.txt", "General")]
    results, errors = await ingest._run_batch(items)
    assert errors == []
    assert results[0]["key"] == "Manuals/a.txt"


@pytest.mark.asyncio
async def test_run_batch_honours_per_item_folder(mock_ollama):
    """One batch must be able to span several folders — the NAS scan maps each
    subdirectory onto its own folder and ingests them together."""
    items = [
        ingest.BatchItem(ingest._bytes_loader(b"alpha body"), "a.txt", "a", "Manuals"),
        ingest.BatchItem(ingest._bytes_loader(b"beta body"), "b.txt", "b", "Avionics"),
    ]
    await ingest._run_batch(items)
    docs = {d["filename"]: d["folder"] for d in await ingest.list_documents()}
    assert docs["a.txt"] == "Manuals"
    assert docs["b.txt"] == "Avionics"


@pytest.mark.asyncio
async def test_run_batch_reports_loader_failure_without_killing_batch(mock_ollama):
    def boom():
        raise OSError("stale file handle")

    items = [
        ingest.BatchItem(boom, "bad.txt", "bad", "General"),
        ingest.BatchItem(ingest._bytes_loader(b"good content"), "good.txt", "good", "General"),
    ]
    results, errors = await ingest._run_batch(items)
    assert len(errors) == 1
    assert "bad.txt" in errors[0]
    assert [r["key"] for r in results] == ["good"]


@pytest.mark.asyncio
async def test_fatal_exceptions_propagate_instead_of_becoming_per_file_errors(mock_ollama):
    """Without this, a circuit breaker raised inside a loader is caught by the
    per-file handler and the job reports success on a dead mount."""
    class Unreachable(Exception):
        pass

    def boom():
        raise Unreachable("mount gone")

    items = [ingest.BatchItem(boom, "x.txt", "x", "General")]
    with pytest.raises(Unreachable):
        await ingest._run_batch(items, fatal_exceptions=(Unreachable,))


@pytest.mark.asyncio
async def test_ingest_many_still_accepts_bytes(mock_ollama):
    """The upload path must be unaffected by the refactor."""
    results = await ingest.ingest_many([(b"hello world", "h.txt")], folder="General")
    assert results[0]["status"] == "ok"
    assert results[0]["filename"] == "h.txt"


@pytest.mark.asyncio
async def test_skip_doc_ids_avoids_reingesting_known_content(mock_ollama):
    first = await ingest.ingest(b"unique payload", "orig.txt", folder="Avionics")
    doc_id = first["doc_id"]

    items = [ingest.BatchItem(
        ingest._bytes_loader(b"unique payload"), "orig.txt", "nas/orig.txt", "NasFolder")]
    results, errors = await ingest._run_batch(items, skip_doc_ids={doc_id})
    assert errors == []
    assert results[0]["status"] == "skipped"

    # The hand-filed folder must survive — this is the silent-reclassification bug.
    docs = await ingest.list_documents()
    assert next(d for d in docs if d["doc_id"] == doc_id)["folder"] == "Avionics"


@pytest.mark.asyncio
async def test_on_flush_receives_written_results(mock_ollama):
    seen = []
    items = [ingest.BatchItem(
        ingest._bytes_loader(f"payload {i}".encode()), f"f{i}.txt", f"k{i}", "General")
        for i in range(3)]
    await ingest._run_batch(items, on_flush=lambda batch: seen.extend(batch))
    assert sorted(r["key"] for r in seen) == ["k0", "k1", "k2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ingest.py -k "run_batch or skip_doc_ids or on_flush or still_accepts" -v`
Expected: FAIL — `AttributeError: module 'ingest' has no attribute '_bytes_loader'`

- [ ] **Step 3: Add the skip check to `_prepare`**

In `ingest.py`, change the `_prepare` signature and add the guard immediately after `doc_id` is computed, before the original file is written:

```python
async def _prepare(
    data: bytes,
    filename: str,
    folder: str = "General",
    skip_doc_ids: set[str] | None = None,
) -> dict:
    """Parse, chunk, and embed one file. The slow part — safe to run concurrently."""
    doc_id = hashlib.sha256(data).hexdigest()[:16]

    # doc_id is a content hash, so a file already ingested through the UI hashes
    # identically when the same bytes turn up on the NAS. Without this guard
    # _write would delete its chunks and rewrite them with the NAS-derived
    # folder, silently reclassifying a document the user filed by hand. Checked
    # before the original is written and before parsing, so a skip costs nothing.
    if skip_doc_ids and doc_id in skip_doc_ids:
        return {"doc_id": doc_id, "filename": filename, "status": "skipped"}

    loop = asyncio.get_running_loop()
```

Leave the rest of `_prepare` unchanged.

- [ ] **Step 4: Rewrite `_run_batch`, `ingest_many`, and `ingest_many_tracked`**

Replace `ingest.py` lines 908-988 (from `async def _run_batch` through the end of `ingest_many_tracked`) with:

```python
Loader = Callable[[], bytes]


@dataclass
class BatchItem:
    """One file's worth of work for _run_batch.

    `folder` rides along per item rather than being a batch-wide argument so a
    single batch can span many NAS subdirectories, each mapping onto its own
    document folder. `key` is opaque to this module and echoed back on the
    result — the NAS scan uses it to carry the file's path relative to the share
    root, which is the only stable identifier: basenames collide across
    directories, and doc_id is not known until the content has been read.
    """
    loader: Loader
    filename: str
    key: Any
    folder: str = "General"


def _bytes_loader(data: bytes) -> Loader:
    """Wrap already-resident bytes as a loader, for the upload path."""
    return lambda: data


async def _run_batch(
    items: list[BatchItem],
    on_prepared=None,
    on_flush=None,
    skip_doc_ids: set[str] | None = None,
    fatal_exceptions: tuple = (),
) -> tuple[list[dict], list[str]]:
    """Prepare files concurrently and flush to the store in batches.

    The loader is called inside the concurrency semaphore rather than by the
    caller, so at most PREPARE_CONCURRENCY file bodies are ever resident. The
    previous signature took `bytes` and required the caller to have read
    everything up front, which on a share of thousands of files means gigabytes
    resident before any work begins.

    Returns (results, errors). A file that fails to load or parse is recorded as
    an error and the rest of the batch still completes — except for exceptions
    listed in `fatal_exceptions`, which propagate. That escape hatch exists
    because a caller's circuit breaker (e.g. "the mount has gone away") would
    otherwise be caught by the per-file handler and reported as one more failed
    file, letting the job finish "successfully" against a dead mount.
    """
    sem = asyncio.Semaphore(config.PREPARE_CONCURRENCY)
    results: list[dict] = []
    errors: list[str] = []
    pending: list[dict] = []

    async def _bounded(item: BatchItem) -> dict:
        async with sem:
            try:
                # Blocking read — a NAS file is a network round trip, and doing
                # it on the event loop stalls every concurrent chat request.
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, item.loader)
                payload = await _prepare(
                    data, item.filename, folder=item.folder, skip_doc_ids=skip_doc_ids
                )
            except fatal_exceptions:
                raise
            except Exception as exc:  # per-file isolation is the point
                log.warning("ingest failed for %s", item.filename, exc_info=True)
                payload = {"filename": item.filename, "status": "error", "error": str(exc)}
            payload["key"] = item.key
            if on_prepared:
                on_prepared()
            return payload

    async def _flush() -> None:
        if not pending:
            return
        written: list[dict] = []
        async with _lock:
            for payload in pending:
                if payload["status"] == "skipped":
                    written.append({
                        "doc_id": payload["doc_id"], "filename": payload["filename"],
                        "key": payload["key"], "chunks": 0, "status": "skipped",
                    })
                    continue
                result = await _write(payload)
                result["key"] = payload["key"]
                written.append(result)
            bm25_index.commit()
        results.extend(written)
        if on_flush:
            on_flush(written)
        pending.clear()

    tasks = [asyncio.create_task(_bounded(item)) for item in items]
    try:
        for coro in asyncio.as_completed(tasks):
            payload = await coro
            if payload.get("status") == "error":
                errors.append(f"{payload['filename']}: {payload['error']}")
                continue
            pending.append(payload)
            if len(pending) >= _WRITE_FLUSH_EVERY:
                await _flush()
    except fatal_exceptions:
        # Commit what already succeeded before surfacing the fatal condition, so
        # an aborted scan still resumes from where it stopped.
        for task in tasks:
            task.cancel()
        await _flush()
        raise
    await _flush()
    return results, errors


async def ingest_many(files: list[tuple[bytes, str]], folder: str = "General") -> list[dict]:
    """Ingest multiple files with concurrent parse+embed and batched writes."""
    items = [BatchItem(_bytes_loader(data), fn, fn, folder) for data, fn in files]
    results, _errors = await _run_batch(items)
    return results


async def ingest_many_tracked(files: list[tuple[bytes, str]], job, folder: str = "General") -> None:
    """Background variant of ingest_many. Updates job.prepared as each file
    clears the embed phase, then flips job.status to done at the end.

    job.prepared is safe to increment without a lock because asyncio is
    single-threaded — only one coroutine runs at a time, and increments happen
    between awaits, so there are no races.

    The job only fails outright on an error that is not attributable to a single
    file; per-file failures are collected into job.errors and the rest proceed.
    """
    def _tick() -> None:
        job.prepared += 1

    items = [BatchItem(_bytes_loader(data), fn, fn, folder) for data, fn in files]
    try:
        results, errors = await _run_batch(items, on_prepared=_tick)
        job.results = results
        job.errors.extend(errors)
        job.status = "done"
    except Exception as exc:
        log.exception("bulk ingest job %s failed", job.id)
        job.status = "failed"
        job.errors.append(str(exc))
```

Add to `ingest.py`'s imports at the top:

```python
from dataclasses import dataclass
from typing import Any, Callable
```

- [ ] **Step 5: Run the full ingest suite**

Run: `python3 -m pytest tests/test_ingest.py -v`
Expected: PASS — the six new tests plus every pre-existing test, unchanged.

- [ ] **Step 6: Run the whole suite to catch upload-path regressions**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, no regressions in `test_api.py` or `test_features.py`

- [ ] **Step 7: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "refactor: make _run_batch take lazy loaders so bulk ingest is memory-bounded"
```

---

### Task 4: NAS ingestion driver (`ingest_nas.py`)

Ties walker, manifest, and batch runner together. Kept out of `ingest.py`, which is already ~1050 lines.

**Files:**
- Create: `ingest_nas.py`
- Test: `tests/test_ingest_nas.py`

**Interfaces:**
- Consumes: `nas.scan`, `nas.resolve_subpath`, `nas_manifest.Manifest`, `ingest._run_batch`, `ingest._bytes_loader`, `store.doc_summaries`
- Produces:
  - `async preview(subpath: str, reingest: bool = False) -> dict` with keys `new`, `unchanged`, `unsupported`, `oversized`, `excluded`, `total_bytes`, `subpath`
  - `async scan_and_ingest(subpath: str, job, reingest: bool = False) -> None`
  - `NasUnreachable(Exception)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingest_nas.py`:

```python
import os

import pytest

import config
import ingest
import ingest_nas
import jobs


@pytest.fixture
def nas_tree(tmp_path, monkeypatch):
    root = tmp_path / "documents"
    (root / "Manuals").mkdir(parents=True)
    (root / "Manuals" / "alpha.txt").write_bytes(b"alpha document body")
    (root / "Manuals" / "beta.txt").write_bytes(b"beta document body")
    (root / "gamma.txt").write_bytes(b"gamma document body")
    (root / "sheet.xlsx").write_bytes(b"unsupported")
    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    monkeypatch.setattr(config, "NAS_MANIFEST_PATH", str(tmp_path / "manifest.db"))
    return root


@pytest.mark.asyncio
async def test_preview_counts_new_files_without_ingesting(nas_tree, mock_ollama):
    result = await ingest_nas.preview("")
    assert result["new"] == 3
    assert result["unchanged"] == 0
    assert result["unsupported"] == 1
    assert await ingest.list_documents() == []


@pytest.mark.asyncio
async def test_preview_scoped_to_subpath(nas_tree, mock_ollama):
    result = await ingest_nas.preview("Manuals")
    assert result["new"] == 2


@pytest.mark.asyncio
async def test_preview_rejects_escape(nas_tree, mock_ollama):
    with pytest.raises(ValueError):
        await ingest_nas.preview("../..")


@pytest.mark.asyncio
async def test_scan_ingests_all_files(nas_tree, mock_ollama):
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert job.status == "done"
    docs = await ingest.list_documents()
    assert sorted(d["filename"] for d in docs) == ["alpha.txt", "beta.txt", "gamma.txt"]


@pytest.mark.asyncio
async def test_scan_assigns_folder_from_directory(nas_tree, mock_ollama):
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    docs = {d["filename"]: d["folder"] for d in await ingest.list_documents()}
    assert docs["alpha.txt"] == "Manuals"
    assert docs["gamma.txt"] == "General"


@pytest.mark.asyncio
async def test_second_scan_skips_everything(nas_tree, mock_ollama):
    job1 = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job1)

    preview = await ingest_nas.preview("")
    assert preview["new"] == 0
    assert preview["unchanged"] == 3

    job2 = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job2)
    assert job2.total == 0
    assert len(await ingest.list_documents()) == 3


@pytest.mark.asyncio
async def test_modified_file_is_reingested(nas_tree, mock_ollama):
    job1 = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job1)

    path = nas_tree / "gamma.txt"
    path.write_bytes(b"gamma document body, revised")
    os.utime(path, (9_000_000, 9_000_000))

    preview = await ingest_nas.preview("")
    assert preview["new"] == 1


@pytest.mark.asyncio
async def test_existing_document_keeps_its_folder(nas_tree, mock_ollama):
    """A file uploaded by hand into one folder must not be reclassified when the
    identical bytes are found on the NAS."""
    await ingest.ingest(b"gamma document body", "gamma.txt", folder="HandFiled")
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    docs = {d["filename"]: d["folder"] for d in await ingest.list_documents()}
    assert docs["gamma.txt"] == "HandFiled"


@pytest.mark.asyncio
async def test_manifest_records_survive_for_resume(nas_tree, mock_ollama):
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    m = ingest_nas._manifest()
    assert m.count() == 3


@pytest.mark.asyncio
async def test_missing_root_fails_job_cleanly(tmp_path, monkeypatch, mock_ollama):
    monkeypatch.setattr(config, "NAS_ROOT", str(tmp_path / "gone"))
    monkeypatch.setattr(config, "NAS_MANIFEST_PATH", str(tmp_path / "m.db"))
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert job.status == "failed"
    assert any("unreachable" in e.lower() or "not a directory" in e.lower() for e in job.errors)


@pytest.mark.asyncio
async def test_mount_loss_aborts_the_job(nas_tree, mock_ollama, monkeypatch):
    """A mount that disappears mid-scan must fail the job, not report success
    after quietly logging one error per unreadable file."""
    monkeypatch.setattr(config, "NAS_IO_ERROR_LIMIT", 1)

    def gone(*a, **kw):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("builtins.open", gone)
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert job.status == "failed"
    assert any("unreachable" in e.lower() for e in job.errors)


@pytest.mark.asyncio
async def test_errors_are_capped(nas_tree, mock_ollama, monkeypatch):
    monkeypatch.setattr(config, "NAS_MAX_ERRORS", 2)
    for i in range(5):
        (nas_tree / f"bad{i}.txt").write_bytes(b"x")

    async def boom(*a, **kw):
        raise RuntimeError("parse exploded")

    monkeypatch.setattr(ingest, "_prepare", boom)
    job = jobs.create(total=0)
    await ingest_nas.scan_and_ingest("", job)
    assert len(job.errors) <= 3  # cap plus the summary line
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ingest_nas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingest_nas'`

- [ ] **Step 3: Implement `ingest_nas.py`**

```python
"""
Drives ingestion from a mounted NAS share.

Composes three pieces that each stay ignorant of the others: `nas` decides what
is a document, `nas_manifest` remembers what has been seen, and
`ingest._run_batch` does the work. Lives outside ingest.py, which is already
long enough.

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
    if _manifest_singleton is None or _manifest_singleton_path() != config.NAS_MANIFEST_PATH:
        os.makedirs(os.path.dirname(config.NAS_MANIFEST_PATH), exist_ok=True)
        _manifest_singleton = Manifest(config.NAS_MANIFEST_PATH)
    return _manifest_singleton


def _manifest_singleton_path() -> str | None:
    return getattr(_manifest_singleton, "_path", None)


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


async def scan_and_ingest(subpath: str, job, reingest: bool = False) -> None:
    """Walk the share and ingest everything the manifest has not already seen."""
    try:
        files, counts = await _walk(subpath)
    except (NasUnreachable, ValueError) as exc:
        job.status = "failed"
        job.errors.append(str(exc))
        return

    fresh, unchanged = _split_new(files, reingest)
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
    # before it is parsed or embedded, rather than being rewritten with a
    # NAS-derived folder over whatever the user filed it under by hand.
    try:
        known_doc_ids = {d["doc_id"] for d in await store.doc_summaries()}
    except Exception:
        log.warning("could not snapshot existing doc_ids; duplicates may be rewritten", exc_info=True)
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
                # Loaders run in executor threads, so this counter is not a
                # strict consecutive count — it is a rough failure density. That
                # is enough for the case it exists for: when the mount is gone
                # every read fails and nothing ever resets it.
                io_errors += 1
                if io_errors >= config.NAS_IO_ERROR_LIMIT:
                    raise NasUnreachable(
                        f"NAS unreachable — {io_errors} read failures, mount likely gone"
                    )
                raise
            io_errors = 0
            return data
        return load

    # One batch across every folder: BatchItem carries its own folder, so there
    # is no positional correlation between two parallel lists to get wrong.
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


def _append_capped(target: list[str], new: list[str]) -> None:
    """Add errors up to the cap, then a single summary line.

    A share holding thousands of unreadable files must not produce a status
    response with one line per file.
    """
    room = config.NAS_MAX_ERRORS - len(target)
    if room > 0:
        target.extend(new[:room])
    hidden = len(new) - max(room, 0)
    if hidden > 0:
        target.append(f"… and {hidden} more error(s) not shown")
```

Add `_path` tracking to `Manifest.__init__` in `nas_manifest.py` so `_manifest()` can detect a monkeypatched path between tests:

```python
        self._path = path
```

(as the first line of `__init__`, before `sqlite3.connect`)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ingest_nas.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add ingest_nas.py nas_manifest.py tests/test_ingest_nas.py
git commit -m "feat: add NAS scan driver with manifest-based incremental ingest"
```

---

### Task 5: Routes

**Files:**
- Modify: `main.py` (add after `ingest_status`, around line 150)
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `ingest_nas.preview`, `ingest_nas.scan_and_ingest`, `nas.resolve_subpath`, `nas.SUPPORTED_EXTS`
- Produces: `GET /ingest/nas/health`, `POST /ingest/nas/preview`, `POST /ingest/nas/scan`

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py` uses a module-level `client = TestClient(main.app)` — not a
fixture — and does not run the app's lifespan. Match that pattern: reference
`client` directly and never add it to a test signature.

Add to the imports at the top of `tests/test_api.py`:

```python
import os

import pytest

import config
import ingest_nas
```

Append to `tests/test_api.py`:

```python
@pytest.fixture
def nas_api_tree(tmp_path, monkeypatch):
    root = tmp_path / "documents"
    (root / "Manuals").mkdir(parents=True)
    (root / "Manuals" / "a.txt").write_bytes(b"alpha body")
    (root / "b.txt").write_bytes(b"beta body")
    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    monkeypatch.setattr(config, "NAS_MANIFEST_PATH", str(tmp_path / "m.db"))
    # The manifest is a module-level singleton; drop it so it reopens against
    # this test's monkeypatched path rather than a previous test's database.
    ingest_nas._manifest_singleton = None
    yield root
    ingest_nas._manifest_singleton = None


def test_nas_health_reports_available_root(nas_api_tree):
    r = client.get("/ingest/nas/health")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["subfolders"] == ["Manuals"]


def test_nas_health_reports_missing_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NAS_ROOT", str(tmp_path / "absent"))
    body = client.get("/ingest/nas/health").json()
    assert body["available"] is False
    assert body["reason"] == "missing"


def test_nas_health_distinguishes_permission_denied(tmp_path, monkeypatch):
    """EACCES and an empty share look identical without this — and a UID mismatch
    between the container and the CIFS mount is the likeliest deploy failure."""
    root = tmp_path / "locked"
    root.mkdir()
    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    monkeypatch.setattr(os, "access", lambda p, m: False)
    body = client.get("/ingest/nas/health").json()
    assert body["available"] is False
    assert body["reason"] == "unreadable"


def test_nas_preview_returns_counts(nas_api_tree, mock_ollama):
    r = client.post("/ingest/nas/preview", json={"subpath": ""})
    assert r.status_code == 200
    assert r.json()["new"] == 2


def test_nas_preview_rejects_escape(nas_api_tree):
    r = client.post("/ingest/nas/preview", json={"subpath": "../.."})
    assert r.status_code == 403


def test_nas_preview_rejects_absolute_path(nas_api_tree):
    r = client.post("/ingest/nas/preview", json={"subpath": "/etc"})
    assert r.status_code == 403


def test_nas_scan_returns_job_id(nas_api_tree, mock_ollama):
    r = client.post("/ingest/nas/scan", json={"subpath": ""})
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_nas_scan_rejects_escape(nas_api_tree):
    r = client.post("/ingest/nas/scan", json={"subpath": "../../etc"})
    assert r.status_code == 403
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_api.py -k nas -v`
Expected: FAIL — 404 on every NAS route

- [ ] **Step 3: Implement the routes**

Add to `main.py` imports:

```python
import ingest_nas
import nas
```

Add after the `ingest_status` route:

```python
class NasScanRequest(BaseModel):
    subpath: str = ""
    reingest: bool = False


@app.get("/ingest/nas/health")
async def nas_health():
    """Liveness of the NAS mount, checked on every call rather than at startup.

    Separates "missing", "unreadable", and "empty" because they are
    indistinguishable from the outside and have completely different fixes: a
    missing bind mount, a UID mismatch against the CIFS mount, and an actually
    empty share.
    """
    root = config.NAS_ROOT
    if not root or not os.path.isdir(root):
        return {"available": False, "reason": "missing", "root": root, "subfolders": []}
    if not os.access(root, os.R_OK | os.X_OK):
        return {"available": False, "reason": "unreadable", "root": root, "subfolders": []}
    try:
        entries = await asyncio.get_running_loop().run_in_executor(
            None, lambda: sorted(
                e.name for e in os.scandir(root)
                if e.is_dir(follow_symlinks=False)
                and e.name not in nas.EXCLUDED_DIRS
                and not e.name.startswith(".")
            )
        )
    except OSError as exc:
        return {"available": False, "reason": "unreadable", "root": root,
                "subfolders": [], "detail": str(exc)}
    return {"available": True, "reason": "ok", "root": root, "subfolders": entries}


@app.post("/ingest/nas/preview")
async def nas_preview(req: NasScanRequest):
    try:
        return await ingest_nas.preview(req.subpath, reingest=req.reingest)
    except ValueError as exc:
        raise HTTPException(403, str(exc))
    except ingest_nas.NasUnreachable as exc:
        raise HTTPException(503, str(exc))


@app.post("/ingest/nas/scan")
async def nas_scan(req: NasScanRequest):
    # Validate containment before creating a job, so a rejected path never shows
    # up as a failed background job the user has to go and read.
    try:
        nas.resolve_subpath(req.subpath)
    except ValueError as exc:
        raise HTTPException(403, str(exc))

    job = jobs.create(total=0)
    task = asyncio.create_task(
        ingest_nas.scan_and_ingest(req.subpath, job, reingest=req.reingest)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job.id, "total": job.total}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_api.py -k nas -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, no regressions

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: add NAS health, preview, and scan routes"
```

---

### Task 6: Frontend control

**Files:**
- Modify: `static/index.html` (markup near line 613; JS near line 1104)

- [ ] **Step 1: Add the markup**

Insert directly after the closing `</div>` of `#folder-input-row` (around line 620), before `#progress-bar`:

```html
      <div id="nas-row" style="display:none">
        <button id="nas-btn" onclick="toggleNasPanel()">🗄 Scan NAS folder</button>
        <div id="nas-panel" style="display:none">
          <select id="nas-subpath"></select>
          <div id="nas-actions">
            <button onclick="previewNas()">Preview</button>
            <button id="nas-start-btn" onclick="startNasScan()" disabled>Start</button>
          </div>
          <div id="nas-summary"></div>
        </div>
      </div>
```

- [ ] **Step 2: Add the JS**

Insert after the `pollJob` function (around line 1140):

```javascript
// ── NAS scan ──────────────────────────────────────────────────────────────────
// The share root is fixed by the container's bind mount, so the user picks a
// subfolder rather than typing a path — the server rejects anything else anyway.
async function initNas() {
  try {
    const r = await _apiFetch(`${API}/ingest/nas/health`);
    const h = await r.json();
    if (!h.available) return;                       // no mount: hide entirely
    const sel = document.getElementById('nas-subpath');
    sel.innerHTML = '<option value="">(entire share)</option>' +
      h.subfolders.map(f => `<option value="${f}">${f}</option>`).join('');
    document.getElementById('nas-row').style.display = 'block';
  } catch { /* backend down; the upload path already reports that */ }
}

function toggleNasPanel() {
  const p = document.getElementById('nas-panel');
  p.style.display = p.style.display === 'none' ? 'block' : 'none';
}

async function previewNas() {
  const subpath = document.getElementById('nas-subpath').value;
  const out = document.getElementById('nas-summary');
  out.textContent = 'Scanning…';
  try {
    const r = await _apiFetch(`${API}/ingest/nas/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subpath }),
    });
    const p = await r.json();
    if (!r.ok) { out.textContent = p.detail || 'Preview failed'; return; }
    const mb = (p.total_bytes / 1048576).toFixed(1);
    out.textContent =
      `${p.new} new (${mb} MB) · ${p.unchanged} unchanged · ${p.unsupported} unsupported`;
    document.getElementById('nas-start-btn').disabled = p.new === 0;
  } catch {
    out.textContent = 'Preview failed — backend unreachable';
  }
}

async function startNasScan() {
  if (_ingestLocked) { showToast('Ingest already in progress — please wait', true); return; }
  const subpath = document.getElementById('nas-subpath').value;

  setIngestLock(true);
  bar.style.display = 'block';
  fill.style.width = '0%';
  setIngestLabel('Scanning NAS…');

  let job;
  try {
    const r = await _apiFetch(`${API}/ingest/nas/scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subpath }),
    });
    job = await r.json();
    if (!r.ok) {
      showToast(job.detail || 'NAS scan failed', true);
      setIngestLock(false);
      bar.style.display = 'none'; setIngestLabel('');
      return;
    }
  } catch {
    showToast('NAS scan failed — backend unreachable', true);
    setIngestLock(false);
    bar.style.display = 'none'; setIngestLabel('');
    return;
  }

  // The walk happens server-side after the response, so the total is not known
  // yet — pass null and let pollJob read the denominator from the job status.
  pollJob(job.job_id, null);
}

```

**Step 2b: Generalise the existing `pollJob` rather than copying it.**

An upload knows its file count up front; a NAS scan discovers it during the
server-side walk. Rather than a near-duplicate poller, make the denominator
optional. Replace the first three lines of the existing `pollJob` body and its
completion toast:

```javascript
function pollJob(jobId, total) {
  const iv = setInterval(async () => {
    try {
      const r   = await _apiFetch(`${API}/ingest/status/${jobId}`);
      if (r.status === 404) {
        clearInterval(iv);
        bar.style.display = 'none'; fill.style.width = '0%'; setIngestLock(false);
        setIngestLabel('');
        showToast('Ingest job lost — server was restarted. Please re-upload your files.', true);
        return;
      }
      const job = await r.json();

      // A NAS scan passes null: its total is only known once the walk finishes,
      // so the denominator comes from the job status instead of the caller.
      const denom = total ?? job.total ?? 0;
      const pct = denom > 0 ? Math.round((job.prepared / denom) * 100) : 0;
      fill.style.width = pct + '%';
      setIngestLabel(denom ? `Processing ${job.prepared} / ${denom}…` : 'Scanning…');

      if (job.status === 'done' || job.status === 'failed') {
        clearInterval(iv);
        fill.style.width = '100%';
        setTimeout(() => { bar.style.display = 'none'; fill.style.width = '0%'; setIngestLabel(''); }, 600);
        setIngestLock(false);
        loadDocs();
        if (job.status === 'done') {
          const n = job.results.filter(x => x.status === 'ok').length;
          const skipped = job.results.filter(x => x.status === 'skipped').length;
          showToast(`${n} document${n !== 1 ? 's' : ''} ready to query` +
                    (skipped ? `, ${skipped} already indexed` : ''));
        } else {
          showToast(job.errors[0] || 'Ingest failed — see console for details', true);
          console.error('Ingest job failed', job);
        }
      }
    } catch (e) { console.error('Status poll error', e); }
  }, 1000);
}
```

- [ ] **Step 3: Call `initNas()` at startup**

Find the existing startup sequence that calls `loadDocs()` on page load and add `initNas();` alongside it.

- [ ] **Step 4: Verify manually**

Run: `GEO_NAS_ROOT=$(pwd)/data/to_ingest python3 -m uvicorn main:app --host 127.0.0.1 --port 8743`
Open http://localhost:8743. Expected: the "Scan NAS folder" button appears; Preview reports the file counts in `data/to_ingest`; Start ingests them with a live progress bar.

Then run with `GEO_NAS_ROOT=/nonexistent` and confirm the button is hidden entirely.

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: add NAS scan control to the sidebar"
```

---

### Task 7: Compose fixes and deployment notes

Two changes to `/opt/geo-assist/docker-compose.yml` on the VM. The first is a pre-existing data-loss bug, not part of this feature.

**Files:**
- Create: `deploy/docker-compose.yml` (version-controlled reference copy — the live file is on the VM and currently tracked nowhere)
- Modify: `README.md` (NAS section)

- [ ] **Step 1: Write the reference compose file**

Create `deploy/docker-compose.yml`:

```yaml
# Reference copy of the VM deployment at /opt/geo-assist/docker-compose.yml.
# Kept in version control because the live file was previously untracked.
# Apply changes on the VM, then mirror them here.
services:
  ollama:
    image: ollama/ollama:latest
    container_name: geo-assist-ollama
    restart: unless-stopped
    volumes:
      - ./ollama-models:/root/.ollama
    networks:
      - geo-assist-net

  qdrant:
    image: qdrant/qdrant:latest
    container_name: geo-assist-qdrant
    restart: unless-stopped
    volumes:
      - ./data/qdrant:/qdrant/storage
    networks:
      - geo-assist-net

  app:
    build: ./app
    container_name: geo-assist-app
    restart: unless-stopped
    depends_on:
      - ollama
      - qdrant
    environment:
      - GEO_CHAT_MODEL=qwen3:14b
      - GEO_EMBED_MODEL=nomic-embed-text
      - GEO_OLLAMA_BASE=http://ollama:11434
      - GEO_QDRANT_HOST=qdrant
      - GEO_QDRANT_PORT=6333
      - GEO_NAS_ROOT=/app/documents
    ports:
      - "8743:8743"
    volumes:
      # config.DATA_DIR is /app/data. Without this bind it is ephemeral
      # container filesystem, and every image rebuild destroys the BM25 index
      # (rebuildable, but a full re-fit each boot), data/originals (NOT
      # recoverable — every citation's source link 404s afterwards), and
      # data/images. The previous ./data/uploads:/app/uploads bind pointed at a
      # path no code references and protected nothing.
      - ./data:/app/data
      # Document share, read-only. The underlying CIFS mount is rw, so :ro here
      # is the only thing keeping the app off the NAS write path.
      # bind-propagation=rslave: without it, a host CIFS remount leaves the
      # container pointing at a stale mountpoint, and the scan reports "0 new,
      # everything unchanged" instead of failing.
      - type: bind
        source: /REPLACE/WITH/HOST/SHARE/PATH   # the CIFS mount holding the documents
        target: /app/documents
        read_only: true
        bind:
          propagation: rslave
    networks:
      - geo-assist-net

networks:
  geo-assist-net:
    driver: bridge
```

- [ ] **Step 2: Confirm Task 0 already landed**

The `data/` persistence fix is Task 0 and should be in place before this point.
Verify rather than redo:

```bash
grep -A2 'volumes:' /opt/geo-assist/docker-compose.yml | grep '/app/data'
```

Expected: `- ./data:/app/data` is present.

- [ ] **Step 3: Apply the remaining compose changes on the VM**

Edit `/opt/geo-assist/docker-compose.yml` to match `deploy/docker-compose.yml`, substituting the confirmed share path for `$NAS_MOUNT_A`.

- [ ] **Step 4: Rebuild and verify**

```bash
cd /opt/geo-assist
sudo docker compose build app
sudo docker compose up -d
sudo docker exec geo-assist-app ls /app/documents | head
curl -s localhost:8743/ingest/nas/health
```

Expected: `ls` lists the share's top-level folders; health returns `{"available": true, ...}`.

If health returns `{"available": false, "reason": "unreadable"}`, the container UID cannot read the CIFS mount — compare `id` inside the container against the `uid=` option in `mount | grep cifs`.

- [ ] **Step 5: Validate mtime stability before trusting the manifest**

On the VM, against the real share:

`/app/documents` exists only inside the container, so the whole check runs there:

```bash
sudo docker exec geo-assist-app sh -c '
  F=$(find /app/documents -type f -name "*.pdf" | head -1)
  echo "checking: $F"
  stat -c "%s %.9Y" "$F"
  sleep 2
  stat -c "%s %.9Y" "$F"
'
```

Expected: identical output both times. If the mtime differs on an unmodified file, CIFS is reporting unstable timestamps and the manifest will re-ingest the whole corpus every scan — raise `MTIME_TOLERANCE` in `nas_manifest.py` to cover the observed jitter, or key the manifest on size alone.

- [ ] **Step 6: Document it**

Add to `README.md` after the "Bulk re-indexing" section:

```markdown
## Ingesting from a NAS share

Mount the share read-only into the container and point `GEO_NAS_ROOT` at it (see
`deploy/docker-compose.yml`). The sidebar then shows **Scan NAS folder**: pick a
subfolder, hit Preview for a count of what would be ingested, then Start.

Re-running a scan only ingests what is new or changed. A local manifest at
`data/nas_manifest.db` records each file's size and mtime, so unchanged files are
skipped without being read — deleting that file forces a full re-scan, which is
safe but slow. NAS subfolders become document folders. The share is never
written to.
```

- [ ] **Step 7: Commit**

```bash
git add deploy/docker-compose.yml README.md
git commit -m "docs: add reference compose with data/ persistence fix and NAS mount"
```

---

## Self-Review

**Spec coverage:** walker → Task 1; manifest → Task 2; memory-bounded ingest → Task 3; skip-if-known and folder preservation → Tasks 3 and 4; error capping and mount-loss detection → Task 4; health/preview/scan routes and containment → Task 5; UI → Task 6; compose, `data/` fix, UID and mtime validation → Task 7. No spec section is unimplemented.

**Deferred deliberately:** the originals-storage decision (spec open question 2) is unresolved pending corpus size; the plan implements the default (keep the copy), which is the current `_prepare` behaviour and needs no code. If reference mode is chosen later it is a change to `_prepare` and `_remove_files` only.

**Blocked:** Task 7 Step 3 needs the confirmed share path. Every other task is implementable now.
