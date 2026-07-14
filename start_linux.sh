#!/bin/bash
# start_linux.sh — Geo-Assist startup for Linux
#
# Usage: ./start_linux.sh
# Override any setting before launching:
#   GEO_CHAT_MODEL=qwen3.5:4b ./start_linux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8743
DEFAULT_CHAT_MODEL="qwen3.5:4b"

echo ""
echo "=========================================================="
echo "  Geo-Assist -- Local AI Engineering Assistant (Linux)"
echo "=========================================================="

# -- detect package manager ----------------------------------------------------
PKG_MANAGER=""
if command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER="apt"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
fi

# -- detect hardware -------------------------------------------------------------
PHYSICAL_CORES="$(nproc --all 2>/dev/null || echo "?")"
RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1048576 ))

HAS_GPU=false
GPU_NAME="None"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_OUT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
    if [ -n "$GPU_OUT" ]; then
        HAS_GPU=true
        GPU_NAME="$GPU_OUT"
    fi
fi

echo "  CPU  : $PHYSICAL_CORES cores"
echo "  RAM  : ${RAM_GB} GB"
if [ "$HAS_GPU" = "true" ]; then
    echo "  GPU  : $GPU_NAME (CUDA)"
else
    echo "  GPU  : not detected (CPU-only mode)"
fi
echo ""

# -- check Ollama installed ----------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
    echo "  [--] Ollama not found."
    read -r -p "      Install Ollama now via the official install script? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "      Skipping. Install manually: https://ollama.com/download/linux"
        exit 1
    fi
fi

# -- check Ollama running -------------------------------------------------------
if ! curl -s -o /dev/null -m 3 "http://127.0.0.1:11434/api/tags"; then
    echo "  [--] Ollama is installed but not running."
    echo "       Start it with: ollama serve &"
    exit 1
fi
echo "  [OK] Ollama running"

# -- check C compiler -------------------------------------------------------------
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    echo "  [--] C compiler not found (needed to build some Python packages)."
    if [ "$PKG_MANAGER" = "apt" ]; then
        read -r -p "      Install build-essential now via apt (requires sudo)? [y/N] " REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            sudo apt-get update && sudo apt-get install -y build-essential python3-dev
        else
            echo "      Skipping. pip install may fail without a compiler."
        fi
    elif [ "$PKG_MANAGER" = "dnf" ]; then
        read -r -p "      Install gcc + python3-devel now via dnf (requires sudo)? [y/N] " REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            sudo dnf install -y gcc python3-devel
        else
            echo "      Skipping. pip install may fail without a compiler."
        fi
    else
        echo "      No supported package manager (apt/dnf) detected."
        echo "      Install a C compiler manually (e.g. gcc) and re-run."
    fi
else
    echo "  [OK] C compiler found"
fi
echo ""

# -- check python -------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Install via your package manager (e.g. 'sudo apt install python3 python3-pip')."
    exit 1
fi

# -- verify / pull required models ---------------------------------------------------
CHAT_MODEL="${GEO_CHAT_MODEL:-$DEFAULT_CHAT_MODEL}"
export GEO_CHAT_MODEL="$CHAT_MODEL"

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

# -- install python deps ---------------------------------------------------------------
echo "Checking Python dependencies..."
if ! python3 -m pip install -r requirements.txt -q; then
    echo "ERROR: pip install failed. Check requirements.txt and your Python version."
    exit 1
fi
echo "  Dependencies OK"
echo ""

# -- start uvicorn ------------------------------------------------------------------------
echo "Starting Geo-Assist server..."
python3 -m uvicorn main:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "Shutting down Geo-Assist..."
    kill "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

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
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:${PORT}" >/dev/null 2>&1 &
fi

wait "$SERVER_PID"
