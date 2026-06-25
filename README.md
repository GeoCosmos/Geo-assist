# Geo-Assist

A fully local, air-gapped document Q&A assistant. Upload engineering documents and ask plain-English questions. All inference runs on your machine via Ollama — no cloud, no telemetry, nothing leaves the device.

## Prerequisites

**Ollama** must be installed and running. Pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:latest      # default (3B, fast)
ollama pull llama3.1:8b          # recommended for higher accuracy (8B)
```

**Python 3.11+** with dependencies:

```bash
pip install -r requirements.txt
```

## Starting

### Windows (primary)

Double-click `start.bat`, or right-click → Run with PowerShell.

`start.ps1` auto-detects your CPU core count and NVIDIA GPU, sets Ollama thread/GPU parameters accordingly, verifies models are pulled, installs Python dependencies, and opens the browser when ready.

To override settings before launching:

```powershell
$env:GEO_CHAT_MODEL   = "llama3.1:8b"
$env:GEO_VISION_MODEL = "moondream"    # optional — enables image analysis
.\start.ps1
```

### macOS / Linux

```bash
GEO_CHAT_MODEL=llama3.1:8b python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

Then open [http://localhost:8743](http://localhost:8743).

## Usage

1. Click **Upload** in the sidebar to ingest one or more documents
2. Wait for the progress bar to complete (embedding runs locally)
3. Type a question and press Enter
4. Answers include citations — source file and page number shown below each response

**Supported file types:** `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`, and most code/config formats (`.py`, `.js`, `.yaml`, `.sql`, etc.)

## Optional features

### Image analysis

Set `GEO_VISION_MODEL` to a vision-capable Ollama model to extract and describe figures from PDFs, PPTXs, and DOCXs:

```bash
ollama pull moondream
GEO_VISION_MODEL=moondream python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

To analyze images in documents that were already ingested before vision was enabled, run `vision_index.py` with the server stopped:

```bash
python3 vision_index.py --dir ~/Desktop/my-docs --model moondream
python3 vision_index.py --dir ~/Desktop/my-docs --skip-existing   # skip already-analyzed files
```

### Audio / video transcription

Transcription requires `faster-whisper` and `ffmpeg` (not included by default):

```bash
pip install -r requirements-audio.txt
# Windows: winget install ffmpeg
```

Once installed, `.mp3`, `.mp4`, `.wav`, `.m4a`, and other audio/video files can be uploaded and queried like any other document.

### Authentication

Multi-user auth with per-document access control is disabled by default. Enable it:

```bash
GEO_AUTH=1 python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

On first run, visit `/auth/setup` to create the admin account. Users and roles are managed from the admin panel in the sidebar.

## Configuration

All settings are in `config.py`. Key environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GEO_CHAT_MODEL` | `llama3.2:latest` | Ollama chat model |
| `GEO_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `GEO_VISION_MODEL` | *(empty)* | Vision model — leave empty to skip image analysis |
| `GEO_AUTH` | `0` | Set to `1` to enable multi-user authentication |
| `GEO_EMBED_CONCURRENCY` | `2` | Parallel embedding requests (raise to `4` with a GPU) |
| `GEO_VISION_TIMEOUT` | `600` | Seconds to wait per image for the vision model |
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
