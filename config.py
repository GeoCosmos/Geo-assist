import os

# ── air-gap enforcement ───────────────────────────────────────────────────────
# These MUST be set before haystack is imported anywhere in the process. Every
# module that touches haystack imports `config` first, so setting them here is
# the single choke point. Do not move these below other imports.
#
# Haystack ships usage telemetry that is ON by default and posts to a deepset
# endpoint on pipeline construction. That is a hard violation of requirement 3
# (no data leaves the machine), so it is force-disabled here rather than left to
# an environment variable the user might not set.
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

OLLAMA_BASE = os.environ.get("GEO_OLLAMA_BASE", "http://127.0.0.1:11434")
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
# Bound on concurrent non-interactive LLM generations (ingest summaries, catalog
# field extraction, query expansion, procedure synthesis). Ollama serialises
# generation on CPU, so >1 buys nothing and costs RAM. Raise only on GPU hardware.
CHAT_CONCURRENCY   = int(os.environ.get("GEO_CHAT_CONCURRENCY",   "1"))
# How long Ollama keeps a model resident after its last use.
#
# This was hard-coded to -1 (never unload) at every call site, to avoid the ~30s
# cold load on the first query. On a 16 GB machine that is too aggressive: the
# chat model and the embedding model both pin themselves permanently, and running
# a second model — a benchmark sweep, or switching GEO_CHAT_MODEL — stacks on top
# rather than replacing. The machine ends up swapping.
#
# "5m" keeps an active session warm while letting an idle machine reclaim the RAM.
# Set GEO_KEEP_ALIVE=-1 to restore the always-resident behaviour on a box with
# RAM to spare, or 0 to unload immediately after every call.
KEEP_ALIVE = os.environ.get("GEO_KEEP_ALIVE", "5m")
# Files parsed+embedded concurrently during bulk ingest. Parsing now runs in a
# thread pool, so this maps to real cores rather than to blocked event-loop time.
PREPARE_CONCURRENCY = int(os.environ.get("GEO_PREPARE_CONCURRENCY", str(min(8, (os.cpu_count() or 4)))))
MIN_IMAGE_BYTES    = 5_000      # skip decorative icons/backgrounds (< 5 KB)
MAX_IMAGE_BYTES    = 10_000_000 # skip enormous images before OCR (> 10 MB)
OCR_IMAGE_MAX_SIDE    = 1024    # resize to this before OCR
OCR_IMAGE_JPEG_QUALITY = 82     # JPEG quality after downscale (strips EXIF implicitly)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

# ── Qdrant ────────────────────────────────────────────────────────────────────
# Qdrant runs as a local server process (qdrant.exe on Windows, ./qdrant
# elsewhere) started by the platform start script and bound to loopback only.
#
# Embedded mode (QdrantDocumentStore(path=...)) is deliberately NOT used: the
# Qdrant client's local mode is brute-force only, documented as suitable for
# <20k points, and raises RuntimeError on concurrent access to the same path.
# The production corpus is well past that ceiling (~35k chunks measured), so it
# needs the real HNSW index.
QDRANT_HOST    = os.environ.get("GEO_QDRANT_HOST", "127.0.0.1")
QDRANT_PORT    = int(os.environ.get("GEO_QDRANT_PORT", "6333"))
QDRANT_STORAGE = os.path.join(DATA_DIR, "qdrant")
QDRANT_INDEX   = os.environ.get("GEO_QDRANT_INDEX", "geo_docs")
EMBED_DIM      = int(os.environ.get("GEO_EMBED_DIM", "768"))  # nomic-embed-text
# Payload fields that get a Qdrant index. Without these, every folder-filtered or
# doc_id-filtered query degenerates to a full payload scan.
QDRANT_INDEXED_FIELDS = [
    {"field_name": "meta.doc_id",      "field_schema": "keyword"},
    {"field_name": "meta.filename",    "field_schema": "keyword"},
    {"field_name": "meta.folder",      "field_schema": "keyword"},
    {"field_name": "meta.table_id",    "field_schema": "keyword"},
    {"field_name": "meta.chunk_type",  "field_schema": "keyword"},
    {"field_name": "meta.page",        "field_schema": "integer"},
    {"field_name": "meta.chunk_index", "field_schema": "integer"},
]
QDRANT_WRITE_BATCH = int(os.environ.get("GEO_QDRANT_WRITE_BATCH", "256"))
# Original uploaded files, kept so citations can link back to the source document.
# Documents ingested before this was added have no file here (see main.py's
# /documents/{doc_id}/file — 404s gracefully for those).
ORIGINALS_DIR = os.path.join(DATA_DIR, "originals")
# Extracted figures, kept as JPEGs so a vision model can caption them later and so
# the original image can be shown to the user or passed to a VLM at answer time.
# The previous pipeline OCR'd images and discarded the bytes, which made adding
# vision support impossible without re-ingesting the whole corpus.
IMAGES_DIR = os.path.join(DATA_DIR, "images")
# When easyocr returns fewer than OCR_MIN_WORDS words the image is a diagram or
# schematic rather than a screenshot. Those are no longer dropped: the image is
# persisted and flagged caption_status="pending" so a vision pass can caption it
# without re-parsing the source document.
KEEP_UNCAPTIONED_IMAGES = os.environ.get("GEO_KEEP_IMAGES", "true").lower() == "true"

# ── NAS ingestion ─────────────────────────────────────────────────────────────
# Root of the mounted document share, as seen *inside* this container. The host
# CIFS mount is bind-mounted here read-only (see the compose file on the VM), so
# the app has no write path to the NAS regardless of what the code does.
#
# Whether a NAS is actually present is deliberately not a constant here: an
# os.path.isdir() evaluated at import time goes stale the moment the mount drops
# or appears, and a container that started before the host mount was ready would
# keep the feature disabled until someone restarted it. GET /ingest/nas/health
# checks live on each call.
NAS_ROOT = os.environ.get("GEO_NAS_ROOT", "/app/documents")
NAS_MANIFEST_PATH = os.path.join(DATA_DIR, "nas_manifest.db")
# Read failures during a scan before the job aborts with "NAS unreachable". The
# CIFS mounts use `soft`, so a dropped mount surfaces as EIO on every read rather
# than hanging — without this the job would grind through thousands of guaranteed
# failures before reporting anything.
NAS_IO_ERROR_LIMIT = int(os.environ.get("GEO_NAS_IO_ERROR_LIMIT", "10"))
# Cap on entries in a job's error list. A share holding thousands of unsupported
# files must not produce a status response with one line per file; the counts
# still reflect the true totals.
NAS_MAX_ERRORS = 200

API_PORT = 8743

# Cross-encoder re-ranking. Disabled by default — requires sentence-transformers and
# the model to be pre-downloaded before running in an air-gapped environment.
# Pre-download: python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RERANK_ENABLED = os.environ.get("GEO_RERANK", "true").lower() == "true"
RERANK_MODEL   = os.environ.get("GEO_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N   = 20  # candidates fed to the cross-encoder before diversity filtering
