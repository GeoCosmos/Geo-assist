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

**A C compiler** — needed because `chromadb`, `sentence-transformers`, and `easyocr` may build native extensions if no prebuilt wheel matches your Python version.
- Windows: Visual Studio Build Tools (C++ workload)
- macOS: Xcode Command Line Tools (`xcode-select --install`)
- Linux: `build-essential` (apt) or `gcc` + `python3-devel` (dnf)

**Python 3.11+** with dependencies:

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

`start.ps1` auto-detects your CPU core count and NVIDIA GPU, checks for Ollama and C++ Build Tools (offering to install either via winget if missing), sets Ollama thread/GPU parameters accordingly, verifies models are pulled, installs Python dependencies, and opens the browser when ready.

To override settings before launching:

```powershell
$env:GEO_CHAT_MODEL   = "qwen3.5:4b"
.\start.ps1
```

### macOS

```bash
./start_mac.sh
```

`start_mac.sh` checks for Ollama (offering `brew install ollama` if missing) and Xcode Command Line Tools, verifies models are pulled, installs Python dependencies, and opens the browser when ready.

Or run it manually:

```bash
GEO_CHAT_MODEL=qwen3.5:4b python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

### Linux

```bash
./start_linux.sh
```

`start_linux.sh` checks for Ollama (offering the official install script if missing) and a C compiler (offering an apt/dnf install if missing), verifies models are pulled, installs Python dependencies, and opens the browser when ready.

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
| `GEO_OCR` | `false` | Set to `true` to enable OCR text extraction from images |
| `GEO_QUERY_EXPANSION` | `false` | Enable multi-query expansion (+20s latency, better cross-doc accuracy) |

## Bulk re-indexing

Wipe and re-ingest a directory:

```bash
python3 reindex.py --dir ~/Desktop/my-docs
python3 reindex.py --dir ~/Desktop/my-docs --limit 50   # first 50 files only
```

Resume an interrupted re-index without wiping:

```bash
python3 resume_reindex.py --dir ~/Desktop/my-docs
```

## Running tests

Tests run fully offline — no Ollama or GPU required:

```bash
python3 -m pytest tests/ -v
```
