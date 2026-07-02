import os

OLLAMA_BASE = "http://127.0.0.1:11434"
CHAT_MODEL   = os.environ.get("GEO_CHAT_MODEL",   "qwen3.5:4b")
# qwen3.5:4b outperformed both llama3.2:latest (3B, hallucinated wrong values) and
# llama3.1:8b (8B, self-contradicted with multiple wrong values in one answer) on
# grounded factual accuracy in real testing — bigger isn't better here. Requires
# "think": false on every Ollama call (see llm.py) or it silently burns 20-30s+ per
# response on hidden reasoning tokens before streaming any visible output.
EMBED_MODEL  = os.environ.get("GEO_EMBED_MODEL",  "nomic-embed-text")
# Defaults to CHAT_MODEL so Ollama never swaps models mid-request.
# Override with a smaller model (e.g. llama3.2:latest) only if it stays loaded.
EXPAND_MODEL = os.environ.get("GEO_EXPAND_MODEL", CHAT_MODEL)
# Set GEO_OCR=true to enable fast easyocr text extraction from images (screenshots,
# text-heavy figures). Requires: pip install -r requirements-ocr.txt
# Images yielding fewer than OCR_MIN_WORDS words are dropped (diagrams, schematics,
# wiring — no vision-model fallback for these).
OCR_ENABLED   = os.environ.get("GEO_OCR",           "false").lower() == "true"
OCR_MIN_WORDS = int(os.environ.get("GEO_OCR_MIN_WORDS", "10"))

# Query expansion adds one full LLM round-trip (~20-25s) before retrieval.
# Disable for lower latency; enable only if cross-doc accuracy needs improving.
QUERY_EXPANSION = os.environ.get("GEO_QUERY_EXPANSION", "false").lower() == "true"

CHUNK_SIZE = 512     # characters — smaller = more focused chunks
CHUNK_OVERLAP = 128
RETRIEVAL_K = 15     # candidates per query before RRF; final context is 8 chunks
DISTANCE_THRESHOLD = 1.3
EMBED_BATCH        = 32
EMBED_CONCURRENCY  = int(os.environ.get("GEO_EMBED_CONCURRENCY",  "2"))
MIN_IMAGE_BYTES    = 5_000      # skip decorative icons/backgrounds (< 5 KB)
MAX_IMAGE_BYTES    = 10_000_000 # skip enormous images before OCR (> 10 MB)
OCR_IMAGE_MAX_SIDE    = 1024    # resize to this before OCR
OCR_IMAGE_JPEG_QUALITY = 82     # JPEG quality after downscale (strips EXIF implicitly)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
# Original uploaded files, kept so citations can link back to the source document.
# Documents ingested before this was added have no file here (see main.py's
# /documents/{doc_id}/file — 404s gracefully for those).
ORIGINALS_DIR = os.path.join(DATA_DIR, "originals")
API_PORT = 8743

# Cross-encoder re-ranking. Disabled by default — requires sentence-transformers and
# the model to be pre-downloaded before running in an air-gapped environment.
# Pre-download: python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RERANK_ENABLED = os.environ.get("GEO_RERANK", "true").lower() == "true"
RERANK_MODEL   = os.environ.get("GEO_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N   = 20  # candidates fed to the cross-encoder before diversity filtering
