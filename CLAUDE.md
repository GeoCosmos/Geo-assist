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

## Target hardware

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
- **Vector store**: ChromaDB (local persistent, `./data/chroma_db/`)
- **Keyword index**: BM25 via `rank-bm25` (in-memory, rebuilt on every ingest/delete)
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
6. Installs Python deps
7. Starts uvicorn with `--loop asyncio --workers 1`
8. Opens browser when ready

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

- `COMMON_EXCLUDES` drops dev-only paths from all three zips — `tests/`, `docs/`, `.github/`, `CLAUDE.md`, `pytest.ini`, `.gitignore`, `backfill_summaries.py` (a one-off migration script, not part of the app).
- Per-OS excludes drop the other OSes' start scripts — `geo-assist-windows-*.zip` ships only `start.bat`/`start.ps1`, `geo-assist-macos-*.zip` only `start_mac.sh`, `geo-assist-linux-*.zip` only `start_linux.sh`.

Anyone needing the full source (including tests/docs) should use GitHub's automatic per-tag "Source code (zip/tar.gz)" links, which always ship everything — no workflow change needed for that. Add new dev-only or OS-specific files to the relevant exclude list in the workflow to keep them out of the runtime zips.

### OCR support (optional)

OCR extracts text from images embedded in documents (screenshots, UI captures, text-heavy figures) in sub-second time. Images with sparse OCR output (diagrams, schematics) are dropped — there is no vision-model fallback.

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

All tests run without a real Ollama — embeddings and chat are mocked with deterministic fakes. Each test gets an isolated ChromaDB via `tmp_path` and a reset BM25 index.

```bash
# Single file
python3 -m pytest tests/test_retriever.py -v
```

**Every new feature or bug fix must include tests.** Add them to the appropriate file in `tests/` — `test_api.py` for endpoint behaviour, `test_ingest.py` for parsing/chunking, `test_retriever.py` for RAG pipeline logic. Use the `mock_ollama` fixture for anything that touches embeddings or chat.

## Architecture

```
geo-assist/
├── main.py          FastAPI app + routes + static serving
├── config.py        All tunable constants
├── llm.py           Async Ollama wrappers (embed + chat)
├── ingest.py        File parsing, chunking, embedding, ChromaDB storage
├── retriever.py     Hybrid RAG pipeline (semantic + BM25 + RRF)
├── bm25_index.py    In-memory BM25 index, rebuilt after every write
├── static/
│   └── index.html   Single-file frontend (vanilla JS, no build step)
├── tests/
│   ├── conftest.py          Fixtures: isolated ChromaDB, mock Ollama
│   ├── test_ingest.py       Parsing + chunking + pipeline tests
│   ├── test_retriever.py    RAG accuracy tests (incl. hybrid BM25 rescue)
│   └── test_api.py          FastAPI endpoint tests
└── data/
    └── chroma_db/   Persisted vector store (gitignored)
```

## RAG pipeline

**Ingestion:** file → text extraction (per page/slide, with headings detected and prepended) → recursive character chunking (512 chars, 80 overlap) → table detection (`Table N` regions tagged with `table_id` in metadata; applies to PDF and DOCX) → summary chunk synthesised (KV-pattern regex → LLM fallback for manuals/reports) → batch embed with `search_document:` prefix → store in ChromaDB with `{doc_id, filename, page, folder, chunk_type, [table_id]}` metadata. `doc_id = sha256(bytes)[:16]`; re-ingesting the same file is idempotent. Up to 16 files prepared concurrently (semaphore-bounded to cap RAM).

**Images (optional):** extracted from PDF/PPTX/DOCX → OCR if `GEO_OCR=true` (easyocr, sub-second; results with fewer than `OCR_MIN_WORDS` words are dropped). Stored as `chunk_type=image` chunks with `chunk_index < -100`. Run as background task — does not block text ingest.

**Query:**
1. Last user turn from session history prepended to retrieval query (context-aware retrieval for follow-ups)
2. LLM generates 2 query variants (multi-query expansion)
3. All 3 queries embedded with `search_query:` prefix, searched concurrently
4. BM25 keyword search — value queries (pressure, voltage, diameter, etc.) get 2× candidate pool
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
| `CHUNK_OVERLAP` | 80 | Character overlap between chunks |
| `RETRIEVAL_K` | 15 | Candidates fetched per query before RRF — do not raise above ~20; value queries double this automatically |
| `DISTANCE_THRESHOLD` | 1.3 | Cosine distance cutoff for semantic gate |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking; requires `pip install -r requirements-reranker.txt` (falls back gracefully if not installed) |
| `RERANK_TOP_N` | 20 | Candidates scored by the cross-encoder |
| `OCR_ENABLED` | `false` | easyocr fast-path for image text extraction; requires `pip install -r requirements-ocr.txt` |
| `OCR_MIN_WORDS` | 10 | Minimum word count from OCR to accept the result (below this the image is dropped) |

## Bulk ingest / reindex scripts

`reindex.py` and `resume_reindex.py` both accept `--folder <name>` to assign all ingested documents to a named folder (default: `"General"`):

```bash
python3 reindex.py --dir ~/my-docs --folder "Project Alpha"
python3 resume_reindex.py --dir ~/my-docs --folder "Project Alpha"
```

Both scripts defer BM25 rebuild to a single call at the end (not after every batch) — do not change this; O(n×batches) rebuilds would make large ingests take hours.

## Linting

```powershell
# Windows
pip install ruff
ruff check .
ruff check --fix .
```

## Known non-issues

- ChromaDB prints `"capture() takes 1 positional argument"` telemetry warnings — harmless.
- `asyncio.iscoroutinefunction` deprecation warnings from FastAPI/Starlette on Python 3.14 — third-party issue, not ours.

## Known issues

- **ChromaDB 1.x Rust backend hangs on large databases.** `chromadb>=1.0` uses a Rust-based connection pool. On a production database ≥ 700 MB (e.g. 148k chunks), `col.count()` blocks indefinitely with `pool timed out while waiting for an open connection`. Tests pass because they use empty `tmp_path` databases. If you hit this on Windows, downgrade: `pip install "chromadb>=0.6,<1.0"` to use the Python backend.


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

**Do not** set `keep_alive: -1` unconditionally if targeting machines where RAM is tight. The current value keeps models loaded permanently; consider `keep_alive: 300` (5-minute idle unload) as an alternative.
