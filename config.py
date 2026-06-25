import os

OLLAMA_BASE = "http://127.0.0.1:11434"
CHAT_MODEL   = os.environ.get("GEO_CHAT_MODEL",   "llama3.2:latest")
EMBED_MODEL  = os.environ.get("GEO_EMBED_MODEL",  "nomic-embed-text")
# Defaults to CHAT_MODEL so Ollama never swaps models mid-request.
# Override with a smaller model (e.g. llama3.2:latest) only if it stays loaded.
EXPAND_MODEL = os.environ.get("GEO_EXPAND_MODEL", CHAT_MODEL)
# Set to a local vision-capable model (e.g. "llava:7b") to enable image analysis.
# Leave empty to skip image extraction entirely (safe default for non-vision setups).
VISION_MODEL = os.environ.get("GEO_VISION_MODEL", "")

# Query expansion adds one full LLM round-trip (~20-25s) before retrieval.
# Disable for lower latency; enable only if cross-doc accuracy needs improving.
QUERY_EXPANSION = os.environ.get("GEO_QUERY_EXPANSION", "false").lower() == "true"

CHUNK_SIZE = 512     # characters — smaller = more focused chunks
CHUNK_OVERLAP = 80
RETRIEVAL_K = 15     # candidates per query before RRF; final context is 8 chunks
DISTANCE_THRESHOLD = 1.3
EMBED_BATCH        = 32
EMBED_CONCURRENCY  = int(os.environ.get("GEO_EMBED_CONCURRENCY",  "2"))
VISION_CONCURRENCY = int(os.environ.get("GEO_VISION_CONCURRENCY", "1"))
MIN_IMAGE_BYTES    = 5_000      # skip decorative icons/backgrounds (< 5 KB)
MAX_IMAGE_BYTES    = 10_000_000 # skip enormous images that would stall the vision model (> 10 MB)
VISION_TIMEOUT     = float(os.environ.get("GEO_VISION_TIMEOUT", "600"))  # seconds per image
VISION_MAX_SIDE    = 1024       # resize to this before sending to the vision model
VISION_JPEG_QUALITY = 82        # JPEG quality after downscale (strips EXIF implicitly)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
API_PORT = 8743

# Set GEO_AUTH=1 to enable multi-user authentication and per-document access control.
# When disabled (default), the app runs single-user with no login required.
AUTH_ENABLED = os.environ.get("GEO_AUTH", "0") == "1"

# Cross-encoder re-ranking. Disabled by default — requires sentence-transformers and
# the model to be pre-downloaded before running in an air-gapped environment.
# Pre-download: python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RERANK_ENABLED = os.environ.get("GEO_RERANK", "false").lower() == "true"
RERANK_MODEL   = os.environ.get("GEO_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N   = 20  # candidates fed to the cross-encoder before diversity filtering
