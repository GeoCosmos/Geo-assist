# Geo-Assist

A fully local, air-gapped document Q&A assistant. Upload engineering documents and ask plain-English questions. All inference runs on your machine via Ollama — no cloud, no telemetry, nothing leaves the device.

## Download

Grab the latest release for your OS from the [Releases page](../../releases):

| OS | What to do |
|---|---|
| Windows | Download `geo-assist-windows-*.zip`, unzip, double-click `start.bat` |
| macOS | Download `geo-assist-macos-*.zip`, unzip, run `./start_mac.sh` |
| Linux | Download `geo-assist-linux-*.zip`, unzip, run `./start_linux.sh` |

Each script checks for Ollama and a C compiler (needed to build a few Python packages), offers to install anything missing, pulls the required models, installs Python dependencies, and opens the app in your browser. The rest of this README explains what's happening under the hood, and how to do any of it by hand.

## Prerequisites

The start script for your OS checks for these and offers to install them — you normally don't need to do this yourself. Manual install, if you'd rather:

**Ollama** — https://ollama.com/download, then pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:4b      # default chat model
```

**Qdrant** — the vector database. It runs as a local server bound to `127.0.0.1`; nothing is exposed to the network. The binary is not bundled (~30 MB), so download it once from the [Qdrant releases page](https://github.com/qdrant/qdrant/releases) and unpack it into a `qdrant/` folder next to the start script:

| OS | Asset | Resulting path |
|---|---|---|
| Windows | `qdrant-x86_64-pc-windows-msvc.zip` | `qdrant\qdrant.exe` |
| macOS (Apple Silicon) | `qdrant-aarch64-apple-darwin.tar.gz` | `qdrant/qdrant` |
| macOS (Intel) | `qdrant-x86_64-apple-darwin.tar.gz` | `qdrant/qdrant` |
| Linux | `qdrant-x86_64-unknown-linux-gnu.tar.gz` | `qdrant/qdrant` |

On macOS and Linux, make it executable — and on macOS clear the quarantine flag, or Gatekeeper will kill it silently and the start script will just report a timeout:

```bash
chmod +x qdrant/qdrant
xattr -dr com.apple.quarantine qdrant/qdrant   # macOS only
```

On an air-gapped machine, copy the binary across alongside the release zip. The start script launches and stops Qdrant for you; if one is already running on the port it leaves it alone.

**A C compiler** — needed because `sentence-transformers` and `easyocr` (both optional) may build native extensions if no prebuilt wheel matches your Python version.
- Windows: Visual Studio Build Tools (C++ workload)
- macOS: Xcode Command Line Tools (`xcode-select --install`)
- Linux: `build-essential` (apt) or `gcc` + `python3-devel` (dnf)

**Python 3.10+** with dependencies:

```bash
pip install -r requirements.txt
```

Windows Visual Studio C++ download:
1. open https://visualstudio.microsoft.com/downloads/
2. Click on the free download version
3. During set-up, check the C++ for desktop module
4. Restart computer

## Starting

### Windows

Double-click `start.bat`, or right-click → Run with PowerShell.

`start.ps1` auto-detects your CPU core count and NVIDIA GPU, checks for Ollama and C++ Build Tools (offering to install either via winget if missing), sets Ollama thread/GPU parameters accordingly, verifies models are pulled, starts the local Qdrant server, installs Python dependencies, and opens the browser when ready. On exit it stops the server it started.

To override settings before launching:

```powershell
$env:GEO_CHAT_MODEL   = "qwen3.5:4b"
.\start.ps1
```

### macOS

```bash
./start_mac.sh
```

`start_mac.sh` checks for Ollama (offering `brew install ollama` if missing) and Xcode Command Line Tools, verifies models are pulled, starts the local Qdrant server, installs Python dependencies, and opens the browser when ready.

Or run it manually:

```bash
GEO_CHAT_MODEL=qwen3.5:4b python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

### Linux

```bash
./start_linux.sh
```

`start_linux.sh` checks for Ollama (offering the official install script if missing) and a C compiler (offering an apt/dnf install if missing), verifies models are pulled, starts the local Qdrant server, installs Python dependencies, and opens the browser when ready.

Or run it manually:

```bash
GEO_CHAT_MODEL=qwen3.5:4b python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

Then open [http://localhost:8743](http://localhost:8743).

## Usage

1. Click **Upload** in the sidebar to ingest one or more documents
2. Wait for the progress bar to complete (embedding runs locally)
3. Type a question and press Enter
4. Answers include citations — source file and page number shown below each response

**Supported file types:** `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`

## Optional features

### OCR

Extracts text from images embedded in PDFs, PPTXs, and DOCXs (screenshots, UI captures, text-heavy figures):

```bash
pip install -r requirements-ocr.txt   # ~200 MB — easyocr + PyTorch CPU
GEO_OCR=true python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

## Configuration

All settings are in `config.py`. Key environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GEO_CHAT_MODEL` | `qwen3.5:4b` | Ollama chat model |
| `GEO_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `GEO_EMBED_CONCURRENCY` | `2` | Parallel embedding requests (raise to `4` with a GPU) |
| `GEO_PREPARE_CONCURRENCY` | `min(8, cores)` | Files parsed in parallel during ingest. Lower it if ingest is straining memory |
| `GEO_KEEP_ALIVE` | `5m` | How long Ollama keeps a model in RAM after last use. `-1` never unloads (fast, memory-hungry), `0` unloads immediately |
| `GEO_OCR` | `false` | Set to `true` to enable OCR text extraction from images |
| `GEO_QUERY_EXPANSION` | `false` | Enable multi-query expansion (+20s latency, better cross-doc accuracy) |
| `GEO_RERANK` | `true` | Cross-encoder reranking. Needs `requirements-reranker.txt` and a pre-downloaded model; falls back silently if absent |
| `GEO_QDRANT_PORT` | `6333` | Port the local Qdrant server listens on |

On a 16 GB machine running low on memory during ingest, `GEO_PREPARE_CONCURRENCY=2` is the first knob to reach for.

## Bulk re-indexing

Wipe and re-ingest a directory:

```bash
python3 reindex.py --dir ~/Desktop/my-docs
python3 reindex.py --dir ~/Desktop/my-docs --limit 50   # first 50 files only
```

## Ingesting from a NAS share

Mount the share read-only into the container and point `GEO_NAS_ROOT` at it — see
`deploy/docker-compose.yml` for the bind. The sidebar then shows **Scan NAS
folder**: pick a subfolder, hit Preview for a count of what would be ingested,
then Start. Progress reuses the normal ingest bar.

Re-running a scan only ingests what is new or changed. A local manifest at
`data/nas_manifest.db` records each file's size and mtime, so unchanged files are
skipped without being read — which is what keeps a re-scan of a large share cheap
over SMB. Deleting that file forces a full re-scan: safe, but slow.

NAS subfolders become document folders. A file already in the index is skipped
rather than rewritten, so a document you filed by hand into one folder keeps that
folder even if the identical bytes also live on the share. **The share is never
written to** — the container mounts it `:ro`.

| Variable | Default | Purpose |
|---|---|---|
| `GEO_NAS_ROOT` | `/app/documents` | Share root as seen inside the container |
| `GEO_NAS_IO_ERROR_LIMIT` | `10` | Read failures before a scan aborts with "NAS unreachable" |

If the sidebar button never appears, check `GET /ingest/nas/health`. It reports
`missing` (no bind mount), `unreadable` (usually the container UID not matching
the CIFS mount's `uid=` option), or `ok` — those first two are otherwise
indistinguishable from an empty share.

## Running tests

Tests run fully offline — no Ollama, GPU, or running Qdrant server required. Ollama
is mocked with deterministic fakes and the store runs in Qdrant's embedded mode.

```bash
python3 -m pytest tests/ -v
```

They cover pipeline logic and the air-gap guarantees (`tests/test_airgap.py`), not
real-world speed or answer quality. For those, use `benchmark_models.py` against a
running instance with documents ingested.
