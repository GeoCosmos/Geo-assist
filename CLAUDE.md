# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What this project is

Geo-Assist is a fully local RAG-based document Q&A assistant. Users upload engineering documents (PDF, DOCX, PPTX, TXT, CSV) and ask plain-English questions. Answers are grounded strictly in the ingested documents — the model can cite, calculate, and infer from retrieved content, but will not fabricate.

**Nothing leaves the machine.** No cloud APIs, no telemetry. Designed for air-gapped or sensitive environments.

## Requirements

These are non-negotiable constraints that must drive every design and implementation decision:

1. **High-volume ingestion** — must handle large numbers of files (hundreds to thousands) without requiring the user to babysit the process or hit memory/timeout walls. Batch processing, streaming, and incremental ingestion are preferred over all-at-once approaches.
2. **Speed** — ingestion and query must complete in seconds to minutes, not hours. Bottlenecks (embedding, chunking, I/O) must be profiled and optimized before shipping a feature.
3. **Fully local, air-gapped** — no data may leave the machine under any circumstances. No cloud API calls, no telemetry, no external model endpoints. All models (embeddings + chat) must run via Ollama locally.
4. **Inference from documents** — the system must reason across ingested content to surface best practices, patterns, and recommendations that aren't stated verbatim. Retrieval alone is not enough; the LLM prompt must be designed to synthesize and infer, not just quote.

## Deployment (primary, as of 2026-08)

Docker Compose on a Linux VM, compose file at `/opt/geo-assist/docker-compose.yml`,
project `geo-assist`. Three containers: `geo-assist-app`, `geo-assist-ollama`,
`geo-assist-qdrant`. A version-controlled reference copy lives at
`deploy/docker-compose.yml` — apply changes on the VM, then mirror them there.

The app image **bakes in the source** (`build: ./app`), so shipping a code change
is `docker compose build app && docker compose up -d app`, not a file copy. The VM
user is not in the `docker` group, so every docker command needs `sudo`.

`config.DATA_DIR` is `/app/data` and **must** be bind-mounted (`./data:/app/data`).
Without it, every rebuild destroys `data/originals/` — which is unrecoverable, and
makes every citation's source link 404 permanently.

The Windows `start.bat` / `start.ps1` path below still works and is what the
release zips ship, but it is no longer the primary deployment.

## Target hardware (Windows release path)

**Primary:** Dell Precision 3260 (Windows 10/11 Pro)
- CPU: Intel 12th-Gen Core i5-12500 / i7-12700 / i9-12900
- GPU: none
- RAM: 16 GB DDR5

**Active configuration:** CPU-only, 16 GB RAM. `config.py` defaults are tuned for this profile (`EMBED_CONCURRENCY=2`, `EMBED_BATCH=32`). `start.ps1` may override these per detected hardware but the defaults are safe to run directly.

| Hardware | Chat model | OLLAMA_NUM_GPU | EMBED_CONCURRENCY | Expected speed |
|---|---|---|---|---|
| T400 / T600 GPU | `qwen3.5:4b` | 999 (hybrid) | 4 | not yet benchmarked on this hardware |
| CPU only (current) | `qwen3.5:4b` | — | 2 | not yet benchmarked on this hardware |

`qwen3.5:4b` replaced `llama3.1:8b`/`llama3.2:latest` as the default after real
grounded-accuracy testing showed it beat both (the 8B model self-contradicted
with multiple wrong values in one answer; the 3B model hallucinated wrong
values). It requires `"think": false` on every Ollama call (see `llm.py`) or it
silently burns 20-30s+ per response on hidden reasoning tokens before streaming
any visible output — this is already applied, but matters if you add new Ollama
call sites.

## Stack

- **Backend**: Python FastAPI on port 8743
- **Vector store**: Qdrant, running as a local loopback-bound server process (`./qdrant/qdrant[.exe]`, storage in `./data/qdrant/`), accessed through Haystack's `QdrantDocumentStore`. All persistence goes through `store.py` — no other module talks to the database.
- **Orchestration**: Haystack (`haystack-ai`) for the document store, `Document` type, and retrieval plumbing. The domain-specific pipeline logic (table completeness, parent-chunk expansion, named-doc injection, procedure mode) stays as plain Python in `retriever.py` rather than as custom Haystack components.
- **Keyword index**: BM25 via `rank-bm25` (in-memory, persisted to `./data/bm25_index.pkl`, updated **incrementally**)
- **Embeddings**: Ollama `nomic-embed-text` with `search_document:` / `search_query:` task prefixes
- **Chat model**: Ollama `qwen3.5:4b` (override with `GEO_CHAT_MODEL`)
- **Frontend**: Single static HTML file (`static/index.html`) — no build step, no Node.js, no CDN dependencies (air-gapped safe)
- **Static assets**: `static/geocosmos_icon.png` — GeoCosmos company logo served at `/static/geocosmos_icon.png`

## Running

### Windows (primary — Dell Precision 3260)

```
# Double-click start.bat
```

`start.bat` calls `start.ps1` which:
1. Detects CPU cores and NVIDIA GPU
2. Checks for Ollama and Visual Studio Build Tools (C++ workload); offers to install either via winget if missing (see "Prerequisite auto-install pattern" below)
3. Sets `OLLAMA_NUM_THREADS`, `OLLAMA_NUM_GPU`, concurrency tuning per hardware
4. Sets `OLLAMA_MAX_LOADED_MODELS=2` — keeps both chat and embed models resident simultaneously (prevents 30s cold-load latency on first query)
5. Verifies / pulls required Ollama models
6. Starts the Qdrant server on `127.0.0.1:6333` (skipped if one is already running) and waits for `/readyz`
7. Installs Python deps
8. Starts uvicorn with `--loop asyncio --workers 1`
9. Opens browser when ready

On exit it stops uvicorn, and stops Qdrant **only if it started it** — a pre-existing instance is left alone.

Override any setting before running:
```powershell
$env:GEO_CHAT_MODEL = "qwen3.5:4b"
.\start.ps1
```

### macOS

```bash
./start_mac.sh
```

Mirrors `start.ps1`'s behavior: detects hardware, checks for Ollama (offers `brew install ollama`) and Xcode Command Line Tools (offers `xcode-select --install`), pulls models, installs Python deps, starts uvicorn, opens the browser.

### Linux

```bash
./start_linux.sh
```

Mirrors the same behavior: detects hardware and GPU (`nvidia-smi`), checks for Ollama (offers the official install script) and a C compiler (offers an `apt`/`dnf` build-tools install), pulls models, installs Python deps, starts uvicorn, opens the browser via `xdg-open`.

### Prerequisite auto-install pattern

All three start scripts (`start.ps1`, `start_mac.sh`, `start_linux.sh`) follow the same shape when a prerequisite is missing: print what's missing and why it's needed, ask the user to confirm (`y/N`) before installing anything, then use the platform's native package manager (winget / brew / apt-dnf, or Ollama's own installer script). If the user declines, the script prints manual install instructions and either exits (Ollama — required) or continues with a warning (C compiler — optional; `pip install` may fail later without it). Follow this pattern if extending these scripts further — don't reintroduce silent or unconditional installs.

### Manual run (any OS, dev only)

```bash
ollama pull qwen3.5:4b && ollama pull nomic-embed-text
pip3 install -r requirements.txt
GEO_CHAT_MODEL=qwen3.5:4b python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

## Releases

Pushing a tag matching `v*` (e.g. `git tag v1.1.0 && git push origin v1.1.0`) triggers `.github/workflows/release.yml`, which zips the repo three times via `git archive` and publishes them to a GitHub Release. Each zip is trimmed to runtime-only files via `git archive` pathspec excludes (`:(exclude)path`):

- `COMMON_EXCLUDES` drops dev-only paths from all three zips — `tests/`, `docs/`, `.github/`, `CLAUDE.md`, `pytest.ini`, `.gitignore`, `migrate_to_qdrant.py` (a one-off migration script, not part of the app), and `handoff.md` (internal notes — gitignored, but excluded explicitly so it cannot ship if ever committed).
- The Qdrant binary is **not** in the repo or the zips. Ship `qdrant/qdrant[.exe]` alongside the release archive, or have the target machine download it once.
- Per-OS excludes drop the other OSes' start scripts — `geo-assist-windows-*.zip` ships only `start.bat`/`start.ps1`, `geo-assist-macos-*.zip` only `start_mac.sh`, `geo-assist-linux-*.zip` only `start_linux.sh`.

Anyone needing the full source (including tests/docs) should use GitHub's automatic per-tag "Source code (zip/tar.gz)" links, which always ship everything — no workflow change needed for that. Add new dev-only or OS-specific files to the relevant exclude list in the workflow to keep them out of the runtime zips.

### OCR support (optional)

OCR extracts text from images embedded in documents (screenshots, UI captures, text-heavy figures) in sub-second time. Images with sparse OCR output (diagrams, schematics) are **kept but left uncaptioned** (`caption_status="pending"`) — there is no vision-model fallback yet, but the image bytes are on disk so one can be added without re-ingesting.

To enable:
```bash
pip install -r requirements-ocr.txt   # ~200 MB — easyocr + PyTorch CPU
GEO_OCR=true python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

No system binary required — pure pip install. The easyocr model downloads once to `~/.EasyOCR/` on first use (needs internet once; then fully offline). On air-gapped machines, pre-download on a connected machine and copy the cache directory.

## Testing

```bash
python3 -m pytest tests/ -v
```

All tests run without a real Ollama — embeddings and chat are mocked with deterministic fakes. Each test gets an isolated Qdrant store in `tmp_path` (the client's local/embedded mode, which needs no running server) and a reset BM25 index.

```bash
# Single file
python3 -m pytest tests/test_retriever.py -v
```

**Every new feature or bug fix must include tests.** Add them to the appropriate file in `tests/` — `test_api.py` for endpoint behaviour, `test_ingest.py` for parsing/chunking, `test_retriever.py` for RAG pipeline logic. Use the `mock_ollama` fixture for anything that touches embeddings or chat.

## Architecture

```
geo-assist/
├── main.py          FastAPI app + routes + static serving
├── config.py        All tunable constants + air-gap env kill-switches
├── llm.py           Async Ollama wrappers (embed + chat), trust_env=False
├── store.py         Qdrant/Haystack document store — the only DB seam
├── ingest.py        File parsing, chunking, embedding, storage
├── retriever.py     Hybrid RAG pipeline (semantic + BM25 + RRF)
├── bm25_index.py    Incremental BM25 index, persisted between restarts
├── nas.py           NAS share walker — junk exclusion, path containment
├── nas_manifest.py  SQLite record of which NAS files have been ingested
├── ingest_nas.py    NAS scan driver (walk → manifest diff → batch ingest)
├── migrate_to_qdrant.py  One-off ChromaDB → Qdrant migration
├── deploy/
│   └── docker-compose.yml  Reference copy of the VM deployment
├── static/
│   └── index.html   Single-file frontend (vanilla JS, no build step)
├── tests/
│   ├── conftest.py          Fixtures: isolated Qdrant (local mode), mock Ollama
│   ├── test_ingest.py       Parsing + chunking + pipeline tests
│   ├── test_retriever.py    RAG accuracy tests (incl. hybrid BM25 rescue)
│   ├── test_nas.py          Walker: exclusion, folder mapping, containment
│   ├── test_nas_manifest.py Manifest: change detection, resume
│   ├── test_ingest_nas.py   Scan driver: incremental ingest, mount loss
│   └── test_api.py          FastAPI endpoint tests
└── data/
    ├── qdrant/           Persisted vector store (gitignored)
    ├── originals/        Source files, for citation links (gitignored)
    └── nas_manifest.db   NAS scan state (gitignored)
```

## RAG pipeline

**Ingestion:** file → **parse in a worker process** (text extraction per page/slide with headings prepended, recursive character chunking, table detection, image extraction) → summary chunk synthesised (KV-pattern regex → LLM fallback) → document number + revision extracted once by the LLM → batch embed with `search_document:` prefix → store with `{doc_id, filename, page, chunk_index, folder, doc_number, revision, chunk_type, [table_id]}` metadata. `doc_id = sha256(bytes)[:16]`; re-ingesting the same file is idempotent.

Parsing runs in a `ProcessPoolExecutor` (`PREPARE_CONCURRENCY`, default `min(8, cpu_count)`). It used to run inline in an async function, which meant the old `_PREPARE_CONCURRENCY = 16` bought no parallelism at all — every file was parsed sequentially on the event loop, stalling `/ingest/status` polls and any in-flight query.

Bulk ingest flushes to the store every `_WRITE_FLUSH_EVERY` (25) files instead of gathering every payload first, and uses per-file error isolation: one corrupt PDF among hundreds is recorded in `job.errors` and the rest of the batch still lands. Previously it raised out of `asyncio.gather` and discarded the whole job's work.

**Images (optional):** extracted from PDF/PPTX/DOCX in the parse worker, downscaled, and **written to `./data/images/{doc_id}/`**. OCR runs if `GEO_OCR=true` (easyocr, sub-second). Images whose OCR is too sparse (diagrams, schematics) are **no longer dropped** — they are stored with `caption_status="pending"` so a vision pass can caption them later by filter, without re-parsing the source. Retrieval runs over the text caption only: joint image/text embedding models are strongly biased toward text and retrieve images poorly. Stored as `chunk_type=image` chunks with `chunk_index < -100`, written by a background task that does not block text ingest.

**Query:**
1. Last user turn from session history prepended to retrieval query (context-aware retrieval for follow-ups)
2. Multi-query expansion — **off by default** (`QUERY_EXPANSION=false`); it costs a full LLM round-trip (~20-25s on CPU) before retrieval even starts. When enabled the LLM generates up to 3 variants
3. Query (plus any variants) embedded with `search_query:` prefix, searched concurrently
4. BM25 keyword search — value queries (pressure, voltage, diameter, etc.) get 2× candidate pool. Top-k is taken with a heap, not by sorting the full corpus
5. Semantic hits + BM25 hits merged via Reciprocal Rank Fusion (RRF, k=60); value queries get 2× BM25 weight
6. Named-doc injection, system-name injection, summary boost, table-chunk boost (1.5× for value queries)
7. Cross-encoder reranking — scores top 20 (query, chunk) pairs; falls back to RRF order if not installed
8. If any top chunk has a `table_id`, all chunks sharing that ID are fetched in one batch and prepended (table completeness)
9. For each selected chunk, ±2 neighbours on the same page fetched and concatenated (parent-chunk expansion)
10. Top chunks passed as context to the LLM; system prompt includes explicit source-file list to prevent hallucinated citations
11. LLM answers with cite/calculate/infer rules enforced by system prompt

**Why hybrid:** semantic search misses exact tokens (part numbers, acronyms, numeric units). BM25 rescues these. RRF combines both rankings without needing score normalisation.

**Why table completeness:** PDF/DOCX tables split across chunk boundaries lose column alignment. When any chunk from a table is retrieved, injecting all sibling chunks ensures the model sees the full table.

**Why parent-chunk expansion:** 512-char chunks can start or end mid-sentence. Fetching ±2 neighbours gives the LLM full paragraphs with proper context boundaries.

## Key constants (`config.py`)

| Constant | Default | Purpose |
|---|---|---|
| `API_PORT` | 8743 | Backend port |
| `CHAT_MODEL` | `qwen3.5:4b` | Ollama chat model; requires `"think": false` on every call (see `llm.py`) |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHUNK_SIZE` | 512 | Characters per chunk (~128 tokens) |
| `CHUNK_OVERLAP` | 128 | Character overlap between chunks |
| `RETRIEVAL_K` | 15 | Candidates fetched per query before RRF — do not raise above ~20; value queries double this automatically |
| `DISTANCE_THRESHOLD` | 1.3 | Cosine distance cutoff for semantic gate (distance = `1 - qdrant_similarity`) |
| `QUERY_EXPANSION` | `false` | Multi-query expansion; adds a full LLM round-trip before retrieval |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking; requires `pip install -r requirements-reranker.txt` (falls back gracefully if not installed). Runs in a thread — never call it inline on the event loop |
| `RERANK_TOP_N` | 20 | Candidates scored by the cross-encoder |
| `OCR_ENABLED` | `false` | easyocr fast-path for image text extraction; requires `pip install -r requirements-ocr.txt` |
| `OCR_MIN_WORDS` | 10 | Minimum word count from OCR to accept it as a caption |
| `KEEP_UNCAPTIONED_IMAGES` | `true` | Persist images whose OCR was too sparse, flagged `caption_status="pending"` for a later vision pass |
| `CHAT_CONCURRENCY` | 1 | Bound on concurrent **non-interactive** LLM generations. `chat_stream` (the interactive answer path) is deliberately ungated so a user question is never queued behind a bulk ingest |
| `PREPARE_CONCURRENCY` | `min(8, cpu_count)` | Files parsed concurrently in the process pool |
| `QDRANT_HOST` / `QDRANT_PORT` | `127.0.0.1` / 6333 | Local Qdrant server |
| `EMBED_DIM` | 768 | Must match the embedding model; changing it requires a re-index |

## Bulk ingest / reindex scripts

`reindex.py` and `resume_reindex.py` both accept `--folder <name>` to assign all ingested documents to a named folder (default: `"General"`):

```bash
python3 reindex.py --dir ~/my-docs --folder "Project Alpha"
python3 resume_reindex.py --dir ~/my-docs --folder "Project Alpha"
```

Both scripts defer `bm25_index.commit()` to a single call at the end (not after every batch) — do not change this.

**Both are only safe with the server stopped.** BM25 is in-memory state persisted
to a pickle by atomic replace. A second process that loads the pickle, adds to it,
and writes it back while the server holds a stale in-memory copy loses whichever
write lands first. This is why NAS ingestion is a route rather than a script — see
below.

## NAS ingestion

`nas.py` (walk + policy) → `nas_manifest.py` (what's been seen) → `ingest_nas.py`
(driver) → `POST /ingest/nas/{health,preview,scan}` in `main.py`. Runs **inside the
server process** for the BM25 reason above.

The share is bind-mounted read-only at `GEO_NAS_ROOT` (default `/app/documents`),
so the app has no write path to the NAS regardless of code. Routes accept a
*relative* subpath only — never an absolute path — resolved and containment-checked
by `nas.resolve_subpath()`.

Re-scans are cheap because `data/nas_manifest.db` keys on `(relpath, size, mtime)`:
unchanged files are skipped without being read. `doc_id` is a content hash, so
without the manifest the only way to know a file has been seen is to read every
byte — the difference between a re-scan in seconds and one in hours over CIFS.

Two non-obvious invariants:

- **Skip, don't rewrite.** A file already in the store is skipped rather than
  re-ingested. `doc_id` is a content hash, so a document filed by hand into one
  folder would otherwise be silently reclassified into its NAS folder.
- **`fatal_exceptions` in `_run_batch`.** The mount-loss circuit breaker raises
  `NasUnreachable` from inside a loader. Without that parameter, `_run_batch`'s
  per-file handler swallows it as one more failed file and the job reports "done"
  against a dead mount. `tests/test_ingest_nas.py::test_mount_loss_aborts_the_job`
  pins this.

**BM25 is now incremental.** `bm25_index` keeps the tokenised corpus in memory and persists it alongside the index, so adding or deleting a document only tokenises the chunks that changed. `add()` / `remove_doc()` mark the index dirty; `commit()` re-fits and saves. The old `rebuild(col)` re-read the entire corpus from the database and re-ran the Snowball stemmer over every chunk on *every single ingest and delete* — ingesting 500 files one at a time meant 500 full rebuilds. `rebuild_from_store()` still exists but is only the cold-start fallback; keep it off the hot path.

## Benchmarking

`benchmark_models.py` compares chat models on **latency and stability** — no labelled
answers required. Retrieval runs once per question and the identical context is
replayed to every model, so the numbers reflect the model, not retrieval jitter.

```bash
python3 benchmark_models.py --models qwen3.5:4b llama3.1:8b --runs 3
python3 benchmark_models.py --questions my_questions.txt --out results.md
```

The column that matters is **numeric drift**: every number+unit in each answer is
extracted and compared across repeat runs of the same question. A model that
answers "28 V" once and "24 V" the next time is unusable for spec lookups no
matter how fast it is — which is exactly how `llama3.1:8b` and `llama3.2` were
disqualified. Treat near-zero drift as the entry requirement and speed as the
tiebreak, not the other way round.

## Linting

```powershell
# Windows
pip install ruff
ruff check .
ruff check --fix .
```

## Known non-issues

- `asyncio.iscoroutinefunction` deprecation warnings from FastAPI/Starlette on Python 3.14 — third-party issue, not ours.
- SWIG `DeprecationWarning`s during tests come from PyMuPDF — harmless.

## Qdrant

Qdrant runs as a **server process**, started by the platform start script and bound to `127.0.0.1`. The binary is not committed (~30 MB) — download `qdrant-x86_64-pc-windows-msvc.zip` (or the matching platform build) from https://github.com/qdrant/qdrant/releases and unpack it to `./qdrant/`. On an air-gapped machine, copy it across alongside the release zip.

**Embedded mode is deliberately not used.** `QdrantDocumentStore(path=...)` runs in the client's local mode, which is brute-force only (no HNSW), documented as suitable for under ~20k points, and raises `RuntimeError` on concurrent access to the same path. The production corpus is well past that ceiling (34,809 chunks across 16 documents when last measured; earlier notes claimed 148k, which the database did not bear out). The **test suite does** use local mode — a handful of documents per test, no server needed, same client API.

Chunk IDs keep the `{doc_id}_{page}_{chunk_index}` scheme. Qdrant only accepts UUID/int point IDs, but the Haystack integration maps each string ID through a deterministic uuid5 and keeps the original in the payload, so ID-derived logic (`endswith("_1_-1")`, neighbour reconstruction) is unaffected.

Qdrant returns cosine **similarity**; Chroma returned cosine **distance**. `store.py` converts via `1.0 - score` so `DISTANCE_THRESHOLD` and the injection sentinels (991.0–999.0) keep their meaning. Don't remove that conversion without auditing every comparison in `retriever.py`.

### Migrating an existing ChromaDB

```bash
pip install "chromadb>=0.6,<1.0"        # only needed for the migration
python3 migrate_to_qdrant.py --dry-run  # check what would happen
python3 migrate_to_qdrant.py            # reuses stored vectors, no re-embedding
python3 migrate_to_qdrant.py --verify-only
```

Stored embeddings are copied as-is, so a 148k-chunk corpus migrates in minutes rather than the hours a re-ingest would take. `doc_number`/`revision` are new fields and stay `—` until you run `--backfill-catalog` (one LLM call per document) or re-ingest. Figures are not backfillable — the old pipeline discarded image bytes; re-ingest a file to persist its figures. The legacy database is only read, never modified.

## Air-gap enforcement

Requirement 3 ("no data leaves the machine") is enforced in code and pinned by `tests/test_airgap.py`, not left to deployment discipline:

- `config.py` sets `HAYSTACK_TELEMETRY_ENABLED=False` (Haystack telemetry is **on by default** and posts to a deepset endpoint), plus `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`.
- `store.py` re-sets the telemetry flag in a bare statement *between* its two import blocks, because an import sorter will move `import config` below third-party imports and silently re-enable it. It also replaces `_telemetry.send_event` with a no-op. Don't "tidy" either.
- `llm.py` builds its httpx client with `trust_env=False`. Without it, `HTTP_PROXY`/`ALL_PROXY` in the environment would route every prompt and document excerpt bound for localhost Ollama through a corporate proxy.
- Start scripts bind Qdrant to `127.0.0.1` explicitly and set `QDRANT__TELEMETRY_DISABLED=true`; the default bind is `0.0.0.0`, which would expose the whole corpus to the local network.

## Known issues

- ~~**ChromaDB 1.x Rust backend hangs on large databases.**~~ — resolved by the move to Qdrant. ChromaDB is no longer a runtime dependency; it is only installed temporarily to run `migrate_to_qdrant.py`.


## Frontend design system

The UI uses a Space/Geo theme. Key design tokens (all in `:root` CSS variables):

| Variable | Value | Purpose |
|---|---|---|
| `--accent` | `#0ea5e9` | Teal/cyan — primary interactive colour |
| `--accent-hover` | `#0284c7` | Darker teal on hover |
| `--sidebar-bg` | gradient `#0c1424 → #070b14` | Deep space navy |
| `--sidebar-text` | `#b0bfd4` | Default sidebar text |
| `--sidebar-muted` | `#48607e` | Dim sidebar labels |
| `--bubble-user` | gradient `#0ea5e9 → #0284c7` | User message bubble |
| `--bubble-assistant` | `#ffffff` (light) / `#111d30` (dark) | Assistant bubble |

The `#send-btn` is `position: absolute` inside `#input-wrap` — do not add `display: flex; gap` to `#input-bar` or the positioning breaks.

`static/geocosmos_icon.png` is shown in `.empty-state` at `max-width: 260px`. In dark mode a CSS filter inverts it. Do not delete the PNG — it is referenced directly by the HTML and served from `/static/`.

## Phase 2 ideas (not yet implemented)

- ~~Sync conversation history to backend on switch~~ — fixed.
- ~~DOCX/PPTX table extraction~~ — fixed. Tables rendered as markdown; PPTX grouped shapes recursed.
- ~~CSV telemetry filter false positives~~ — fixed. CSV routed to dedicated `_extract_csv`.
- ~~Armenian and Russian~~ — implemented. System prompt responds in user's language.
- ~~Folder management~~ — implemented. Documents stored with folder metadata, moveable, filterable.
- ~~Folder drag-drop in UI~~ — implemented. Dropping a folder auto-fills the folder name field and recursively collects all files via `webkitGetAsEntry()` (handles the 100-item `readEntries` pagination limit).
- ~~Comparison/information tables~~ — implemented. LLM formats comparison queries as markdown tables.
- ~~Table-aware extraction (Phase 2)~~ — implemented. `_extract_pdf` uses `page.find_tables()` (PyMuPDF ≥1.25) and renders tables as markdown in reading order (y0-sorted interleaving of text blocks and table markdown). Documents uploaded before this was added need re-ingestion to benefit.
- ~~Query-time revision extraction regex~~ — replaced. `doc_number` and `revision` are extracted once at ingest by an LLM call and stored in metadata; catalog queries are now a metadata read. This deleted ~280 lines of regex tuned to one customer's "Is. 3 - Rev. 2" / "E3R10" / French "Éd./Rév." conventions, which ran on every catalog query for every document and silently produced an em dash for any convention it had not been taught.
- ~~Named-doc matching hijacking the context window~~ — fixed. Matching is word-bounded with a 6-character stem minimum. Under the old bare-substring, 4-character rule a file called `data.pdf` matched any question containing "data" — and a matched document is granted *every* context slot by `_diverse_top`.
- **Vision captioning for figures** — not yet wired. Images are now persisted at ingest and flagged `caption_status="pending"`, so this can be added without re-ingesting. Approach (per the Haystack multimodal tutorial): caption with a local Ollama VLM at index time for retrieval, and pass the original image to a vision model at answer time. Model choice is unbenchmarked — candidates should be measured on real figures for caption quality and per-image CPU latency before committing.
- ~~Re-ranking with a cross-encoder after RRF~~ — implemented. Optional; enable with `GEO_RERANK=true`. Requires `pip install -r requirements-reranker.txt` and pre-downloading the model (needs internet once). See `reranker.py` and `config.py`.
- ~~Delete all / clear collection endpoint~~ — implemented. `DELETE /documents`. "Clear all" button in sidebar UI.
- ~~Procedure agent mode~~ — implemented. Each doc has a ▶ button (hover to reveal) that starts procedure mode. LLM synthesizes steps from the document content (works on specs, manuals, and descriptions — not just docs with explicit numbered lists). Current step is injected into the system prompt; model is instructed to flag conflicts against retrieved reference docs with `⚠️ CONFLICT:` warnings. See `retriever._generate_procedure_steps()`, `retriever._PROCEDURE_SYSTEM`, `retriever._build_system()`, and the `/procedure/session/*` endpoints in `main.py`.
- ~~OCR for image ingestion~~ — implemented. Set `GEO_OCR=true` to enable. Install: `pip install -r requirements-ocr.txt`. OCR (easyocr, CPU-only) runs on every image extracted from PDF/PPTX/DOCX; if ≥10 words are found the OCR text is stored as an image chunk (sub-second per image). Sparse results (diagrams, schematics) are dropped — there is no vision-model fallback.

## Procedure agent

Step-by-step procedure walkthrough is implemented within the main agent (no separate service). See handoff.md for full implementation notes.

**Key files:**
- `retriever._generate_procedure_steps(chunks)` — async; calls the LLM with `_PROCEDURE_EXTRACT_SYSTEM` to extract or synthesize numbered steps; falls back to raw chunks on failure
- `retriever._PROCEDURE_SYSTEM` — system prompt for procedure Q&A; injects current step, instructs conflict detection
- `retriever._build_system(context_parts, procedure)` — picks between `_SYSTEM` and `_PROCEDURE_SYSTEM`
- `/procedure/session/{id}/start`, `/navigate`, `DELETE` in `main.py`; state stored in `_proc_sessions`
- `#proc-bar` in `static/index.html`; JS state in `_procState`

## Model residency

`config.KEEP_ALIVE` (env `GEO_KEEP_ALIVE`, default `"5m"`) controls how long Ollama
keeps a model in RAM after its last use. It is passed on every Ollama call in
`llm.py`.

This was previously hard-coded to `-1` — never unload — at all three call sites, to
dodge the ~30s cold load on first query. On a 16 GB machine that is too aggressive:
the chat model and the embedding model each pin themselves permanently, and any
second model (a benchmark sweep, or changing `GEO_CHAT_MODEL`) stacks on top rather
than replacing, until the machine swaps.

`"5m"` keeps an active session warm and lets an idle machine reclaim the memory.
Set `GEO_KEEP_ALIVE=-1` to restore always-resident behaviour where RAM allows, or
`0` to unload after every call.

`llm.unload(model)` evicts a model immediately; `benchmark_models.py` calls it
between models so a sweep does not accumulate all of them.
