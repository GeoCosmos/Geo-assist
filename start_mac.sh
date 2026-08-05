#!/bin/bash
# start_mac.sh — Geo-Assist startup for macOS
#
# Usage: ./start_mac.sh
# Override any setting before launching:
#   GEO_CHAT_MODEL=qwen3.5:4b ./start_mac.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8743
DEFAULT_CHAT_MODEL="qwen3.5:4b"

echo ""
echo "=========================================================="
echo "  Geo-Assist -- Local AI Engineering Assistant (macOS)"
echo "=========================================================="

# -- detect hardware ----------------------------------------------------------
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    CHIP_LABEL="Apple Silicon"
else
    CHIP_LABEL="Intel"
fi
CPU_BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")"
PHYSICAL_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || echo "?")"
LOGICAL_CORES="$(sysctl -n hw.logicalcpu 2>/dev/null || echo "?")"
RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))

echo "  CPU  : $CPU_BRAND ($CHIP_LABEL)"
echo "         $PHYSICAL_CORES physical cores / $LOGICAL_CORES logical threads"
echo "  RAM  : ${RAM_GB} GB"
echo ""

# -- check Ollama installed ----------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
    echo "  [--] Ollama not found."
    if command -v brew >/dev/null 2>&1; then
        read -r -p "      Install Ollama now via Homebrew? [y/N] " REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            brew install ollama
        else
            echo "      Skipping. Install manually: https://ollama.com/download/mac"
            exit 1
        fi
    else
        echo "      Homebrew not found. Install Ollama manually: https://ollama.com/download/mac"
        exit 1
    fi
fi

# -- check Ollama running -------------------------------------------------------
if ! curl -s -o /dev/null -m 3 "http://127.0.0.1:11434/api/tags"; then
    echo "  [--] Ollama is installed but not running."
    echo "       Start it from the menu bar app, or run: ollama serve &"
    exit 1
fi
echo "  [OK] Ollama running"

# -- check C compiler (Xcode Command Line Tools) --------------------------------
if ! command -v cc >/dev/null 2>&1; then
    echo "  [--] C compiler not found (needed to build some Python packages)."
    read -r -p "      Install Xcode Command Line Tools now? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        echo "      This opens Apple's installer -- follow the prompts, then re-run this script."
        xcode-select --install
        exit 0
    else
        echo "      Skipping. pip install may fail without a compiler."
    fi
else
    echo "  [OK] C compiler found"
fi
echo ""

# -- check python ----------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Install from https://python.org or 'brew install python3'."
    exit 1
fi

# -- verify / pull required models ------------------------------------------------
CHAT_MODEL="${GEO_CHAT_MODEL:-$DEFAULT_CHAT_MODEL}"
export GEO_CHAT_MODEL="$CHAT_MODEL"
export GEO_VISION_MODEL="${GEO_VISION_MODEL:-moondream}"

confirm_model() {
    local name="$1"
    local label="$2"
    if ollama list 2>/dev/null | grep -q "^${name}"; then
        echo "  [OK] $label ($name)"
    else
        echo "  [--] $label ($name) not found -- pulling now..."
        ollama pull "$name"
    fi
}

echo "Checking models..."
confirm_model "$CHAT_MODEL" "Chat"
confirm_model "nomic-embed-text" "Embed"
echo ""

# -- start Qdrant -------------------------------------------------------------------
# Qdrant runs as a local server process bound to loopback. Embedded mode is not
# used: it is brute-force only and documented as suitable for under ~20k points,
# where the production corpus is ~148k chunks.
#
# The binary is expected at ./qdrant/qdrant. It is not committed to the repo
# (~30 MB); download it once from https://github.com/qdrant/qdrant/releases for
# your platform and unpack it there. On an air-gapped machine, copy it across
# alongside the release archive.
QDRANT_PORT="${GEO_QDRANT_PORT:-6333}"
QDRANT_BIN="$(dirname "$0")/qdrant/qdrant"
QDRANT_STORAGE="$(dirname "$0")/data/qdrant"
QDRANT_PID=""

qdrant_up() {
    curl -s -o /dev/null -m 2 "http://127.0.0.1:${QDRANT_PORT}/readyz"
}

echo "Checking Qdrant..."
if qdrant_up; then
    echo "  [OK] Qdrant already running on port ${QDRANT_PORT}"
elif [ ! -x "$QDRANT_BIN" ]; then
    echo "  [!!] qdrant binary not found at $QDRANT_BIN"
    echo "       Download it from https://github.com/qdrant/qdrant/releases"
    echo "       and unpack it to ./qdrant/ (chmod +x qdrant/qdrant)"
    exit 1
else
    mkdir -p "$QDRANT_STORAGE"
    # Bind to loopback explicitly — the default 0.0.0.0 would expose the whole
    # document corpus to anything on the local network.
    QDRANT__SERVICE__HOST=127.0.0.1 \
    QDRANT__SERVICE__HTTP_PORT="$QDRANT_PORT" \
    QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE" \
    QDRANT__TELEMETRY_DISABLED=true \
        "$QDRANT_BIN" >/dev/null 2>&1 &
    QDRANT_PID=$!
    echo -n "  Starting Qdrant on 127.0.0.1:${QDRANT_PORT}."
    QDRANT_READY=false
    for i in $(seq 1 30); do
        sleep 0.5
        if qdrant_up; then QDRANT_READY=true; break; fi
        echo -n "."
    done
    echo ""
    if [ "$QDRANT_READY" = true ]; then
        echo "  [OK] Qdrant ready"
    else
        echo "  [!!] Qdrant did not become ready in 15s."
        kill "$QDRANT_PID" 2>/dev/null
        exit 1
    fi
fi
echo ""

# -- install python deps ------------------------------------------------------------
echo "Checking Python dependencies..."
if ! python3 -m pip install -r requirements.txt -q; then
    echo "ERROR: pip install failed. Check requirements.txt and your Python version."
    exit 1
fi
echo "  Dependencies OK"
echo ""

# -- start uvicorn ---------------------------------------------------------------------
echo "Starting Geo-Assist server..."
python3 -m uvicorn main:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "Shutting down Geo-Assist..."
    kill "$SERVER_PID" 2>/dev/null
    # Only stop Qdrant if this script started it — a pre-existing instance may
    # be in use by something else.
    if [ -n "$QDRANT_PID" ]; then
        echo "Stopping Qdrant..."
        kill "$QDRANT_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# -- wait for server ready ---------------------------------------------------------------
READY=false
echo -n "Waiting for server to be ready."
for i in $(seq 1 40); do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:${PORT}/health"; then
        READY=true
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

if [ "$READY" != "true" ]; then
    echo "ERROR: Server did not become ready within 40 seconds."
    exit 1
fi

echo ""
echo "=========================================================="
echo "  Ready:  http://localhost:${PORT}"
echo "  Press Ctrl+C to stop."
echo "=========================================================="
echo ""
open "http://localhost:${PORT}"

wait "$SERVER_PID"
