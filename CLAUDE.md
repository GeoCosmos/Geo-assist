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
| T400 / T600 GPU | `llama3.1:8b` | 999 (hybrid) | 4 | ~15 tok/s |
| CPU only (current) | `llama3.2:latest` | — | 2 | ~4–5 tok/s |

## Stack

- **Backend**: Python FastAPI on port 8743
- **Vector store**: ChromaDB (local persistent, `./data/chroma_db/`)
- **Keyword index**: BM25 via `rank-bm25` (in-memory, rebuilt on every ingest/delete)
- **Embeddings**: Ollama `nomic-embed-text` with `search_document:` / `search_query:` task prefixes
- **Chat model**: Ollama `llama3.1:8b` with GPU, `llama3.2:latest` CPU-only (auto-selected; override with `GEO_CHAT_MODEL`)
- **Frontend**: Single static HTML file (`static/index.html`) — no build step, no Node.js, no CDN dependencies (air-gapped safe)
- **Static assets**: `static/geocosmos_icon.png` — GeoCosmos company logo served at `/static/geocosmos_icon.png`

## Running

### Windows (primary — Dell Precision 3260)

```
# Double-click start.bat
```

`start.bat` calls `start.ps1` which:
1. Detects CPU cores and NVIDIA GPU
2. Sets `OLLAMA_NUM_THREADS`, `OLLAMA_NUM_GPU`, concurrency tuning per hardware
3. Sets `OLLAMA_MAX_LOADED_MODELS=2` — keeps both chat and embed models resident simultaneously (prevents 30s cold-load latency on first query)
4. Verifies / pulls required Ollama models
5. Installs Python deps
6. Starts uvicorn with `--loop asyncio --workers 1`
7. Opens browser when ready

Override any setting before running:
```powershell
$env:GEO_CHAT_MODEL = "llama3.1:8b"
$env:GEO_VISION_MODEL = "llava:7b"   # optional — enables image analysis
.\start.ps1
```

### macOS / Linux (dev only)

```bash
ollama pull llama3.1:8b && ollama pull nomic-embed-text
pip3 install -r requirements.txt
GEO_CHAT_MODEL=llama3.1:8b python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

### Audio/video support (optional)

Audio/video transcription requires `faster-whisper`, which pulls in `torch` and `ctranslate2` (~1–2 GB). It is intentionally excluded from `requirements.txt` to avoid a heavyweight install for users who don't need it.

To enable:
```bash
pip install -r requirements-audio.txt
```

On CPU-only hardware (no GPU), transcription runs on the `tiny` Whisper model (~39 MB) which fits comfortably in RAM but is slower and less accurate than GPU inference. Image analysis via `GEO_VISION_MODEL` is also possible on CPU — `llava:7b` (~4 GB) fits within 16 GB RAM alongside the chat and embed models, but expect 2–5 minutes per image. Use only for documents where figures are essential to the answers.

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

**Ingestion:** file → text extraction (per page/slide) → recursive character chunking (512 chars, 80 overlap) → table detection (`Table N` regions tagged with `table_id` in metadata) → batch embed with `search_document:` prefix → store in ChromaDB with `{doc_id, filename, page, [table_id]}` metadata. `doc_id = sha256(bytes)[:16]`; re-ingesting the same file is idempotent.

**Query:**
1. Last user turn from session history prepended to retrieval query (context-aware retrieval for follow-ups)
2. LLM generates 2 query variants (multi-query expansion)
3. All 3 queries embedded with `search_query:` prefix, searched concurrently
4. BM25 keyword search run on original question
5. Semantic hits + BM25 hits merged via Reciprocal Rank Fusion (RRF, k=60)
6. If any top chunk has a `table_id`, all chunks sharing that ID are fetched and prepended to context (table completeness)
7. Top chunks passed as context to the LLM (with session history as prior messages)
8. LLM answers with cite/calculate/infer rules enforced by system prompt

**Why hybrid:** semantic search misses exact tokens (part numbers, acronyms, numeric units). BM25 rescues these. RRF combines both rankings without needing score normalisation.

**Why table completeness:** PDF tables split across chunk boundaries lose column alignment. When any chunk from a table is retrieved, injecting all sibling chunks ensures the model sees the full table.

## Key constants (`config.py`)

| Constant | Value | Purpose |
|---|---|---|
| `API_PORT` | 8743 | Backend port |
| `CHAT_MODEL` | `llama3.2:latest` | Ollama chat model (3B; fits in 8GB alongside embed model) |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHUNK_SIZE` | 512 | Characters per chunk (~128 tokens) |
| `CHUNK_OVERLAP` | 80 | Character overlap between chunks |
| `RETRIEVAL_K` | 15 | Candidates fetched per query before RRF — do not raise above ~20; only 8 chunks reach the LLM and higher values waste ChromaDB + BM25 time |
| `DISTANCE_THRESHOLD` | 1.3 | Cosine distance cutoff for semantic gate |

## Security

- **All document-management endpoints require authentication** when `AUTH_ENABLED=True` (the default). This covers: `GET /documents`, `GET /folders`, `POST /documents/{doc_id}/folder`, `DELETE /documents/{doc_id}`, `GET /ingest/status/{job_id}`. Adding a new document-touching endpoint without `Depends(get_current_user)` is a silent auth bypass.
- `auth.py` uses atomic writes (`os.replace()`) for both `auth.json` and `revocations.json` — never truncate-then-write these files or a crash mid-write will corrupt auth state and lock out all users.
- `_load()` in `auth.py` raises on JSON corruption rather than returning `{}` (which would re-open the admin setup endpoint). If you see a startup error about corrupt `auth.json`, restore from backup — do not silently swallow the exception.

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
- ~~Image extraction from PDF/PPTX/DOCX~~ — done. Set `GEO_VISION_MODEL=moondream` (or any vision-capable Ollama model) to enable. Use `vision_index.py` to back-fill image chunks for already-ingested docs (run with server stopped to avoid RAM competition).
- ~~DOCX/PPTX table extraction~~ — fixed. Tables rendered as markdown; PPTX grouped shapes recursed.
- ~~CSV telemetry filter false positives~~ — fixed. CSV routed to dedicated `_extract_csv`.
- ~~Audio/video transcription~~ — implemented. Optional install via `requirements-audio.txt`.
- ~~Armenian and Russian~~ — implemented. System prompt responds in user's language.
- ~~Folder management~~ — implemented. Documents stored with folder metadata, moveable, filterable.
- ~~Folder drag-drop in UI~~ — implemented. Dropping a folder auto-fills the folder name field and recursively collects all files via `webkitGetAsEntry()` (handles the 100-item `readEntries` pagination limit).
- ~~Comparison/information tables~~ — implemented. LLM formats comparison queries as markdown tables.
- ~~Table-aware extraction (Phase 2)~~ — implemented. `_extract_pdf` uses `page.find_tables()` (PyMuPDF ≥1.25) and renders tables as markdown in reading order (y0-sorted interleaving of text blocks and table markdown). Documents uploaded before this was added need re-ingestion to benefit.
- ~~Re-ranking with a cross-encoder after RRF~~ — implemented. Optional; enable with `GEO_RERANK=true`. Requires `pip install -r requirements-reranker.txt` and pre-downloading the model (needs internet once). See `reranker.py` and `config.py`.
- ~~Delete all / clear collection endpoint~~ — implemented. `DELETE /documents` (requires auth when enabled). "Clear all" button in sidebar UI.
- ~~Procedure agent mode~~ — implemented. Each doc has a ▶ button (hover to reveal) that starts procedure mode. LLM synthesizes steps from the document content (works on specs, manuals, and descriptions — not just docs with explicit numbered lists). Current step is injected into the system prompt; model is instructed to flag conflicts against retrieved reference docs with `⚠️ CONFLICT:` warnings. See `retriever._generate_procedure_steps()`, `retriever._PROCEDURE_SYSTEM`, `retriever._build_system()`, and the `/procedure/session/*` endpoints in `main.py`.
- ~~OCR fast-path for image ingestion~~ — implemented. Set `GEO_OCR=true` to enable. Install: `pip install -r requirements-ocr.txt`. OCR (easyocr, CPU-only) runs first on every extracted image; if ≥10 words are found the OCR text is stored directly (sub-second per image). Sparse results fall through to `GEO_VISION_MODEL` if set — so screenshots go through OCR and schematics/wiring diagrams still go to Moondream. The `vision_index.py` backfill script automatically uses the same two-tier logic.

## Procedure agent

Step-by-step procedure walkthrough is implemented within the main agent (no separate service). See handoff.md for full implementation notes.

**Key files:**
- `retriever._generate_procedure_steps(chunks)` — async; calls the LLM with `_PROCEDURE_EXTRACT_SYSTEM` to extract or synthesize numbered steps; falls back to raw chunks on failure
- `retriever._PROCEDURE_SYSTEM` — system prompt for procedure Q&A; injects current step, instructs conflict detection
- `retriever._build_system(context_parts, procedure)` — picks between `_SYSTEM` and `_PROCEDURE_SYSTEM`
- `/procedure/session/{id}/start`, `/navigate`, `DELETE` in `main.py`; state stored in `_proc_sessions`
- `#proc-bar` in `static/index.html`; JS state in `_procState`

**Do not** set `keep_alive: -1` unconditionally if targeting machines where RAM is tight. The current value keeps models loaded permanently; consider `keep_alive: 300` (5-minute idle unload) as an alternative.
