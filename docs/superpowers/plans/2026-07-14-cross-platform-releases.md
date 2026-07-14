# Cross-Platform Releases + Prerequisite Auto-Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Geo-Assist parity startup scripts for Windows/macOS/Linux that auto-detect and (with confirmation) auto-install missing prerequisites, and publish tagged GitHub Releases with three OS-labeled download zips.

**Architecture:** Three independent shell/PowerShell scripts sharing one behavioral pattern (detect hardware → check Ollama → check C compiler → pull models → install deps → start server → open browser), a GitHub Actions workflow that zips the repo three times on tag push, and doc updates that stop describing manual steps the scripts now automate.

**Tech Stack:** Bash (macOS/Linux), PowerShell (Windows), GitHub Actions (YAML), `git archive`, winget/brew/apt/dnf, Ollama CLI.

## Global Constraints

- No compiled/bundled installer — the app stays a source tree + start script per OS (per design doc "Non-goals").
- No auto-install of optional features (OCR, reranker, audio) — those stay manual opt-in, unchanged.
- Every prerequisite install (Ollama, C compiler) requires an explicit `y/N` confirmation before anything is downloaded — never silent/unconditional.
- Windows Build Tools install must warn the user it's a 1–2 GB download that can take several minutes (design doc "Open risk").
- All three release zips contain identical full-repo content (via `git archive`, which respects `.gitignore`) — no per-OS content trimming.
- Target port stays `8743` everywhere, matching existing scripts and `config.py`.

---

### Task 1: Rewrite `start_mac.sh` with hardware detection and prerequisite auto-install

**Files:**
- Modify: `start_mac.sh` (currently 5 lines — full rewrite)

**Interfaces:**
- Produces: an executable `start_mac.sh` in the repo root that a user runs as `./start_mac.sh`. Behavior other tasks depend on: prints `[OK] Ollama running` once Ollama is confirmed reachable on `127.0.0.1:11434`; prints `Ready:  http://localhost:8743` once the server passes its health check. README/CLAUDE.md tasks (5, 6) reference this exact invocation and these exact behaviors.

This task is testable end-to-end on this machine (macOS dev box, Ollama already installed).

- [ ] **Step 1: Write the new script**

Replace the full contents of `start_mac.sh` with:

```bash
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

# -- install python deps ------------------------------------------------------------
echo "Checking Python dependencies..."
python3 -m pip install -r requirements.txt -q
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x start_mac.sh
```

- [ ] **Step 3: Syntax-check**

Run: `bash -n start_mac.sh`
Expected: no output (exit code 0)

- [ ] **Step 4: Smoke-test end-to-end**

Run: `./start_mac.sh`
Expected: prints hardware info, `[OK] Ollama running`, model check lines, `Dependencies OK`, then `Waiting for server to be ready....` followed by `Ready:  http://localhost:8743`, and opens the URL in a browser tab. Press `Ctrl+C` — expect `Shutting down Geo-Assist...` and the process exits cleanly (verify with `lsof -i :8743` showing nothing afterward).

If Ollama or a compiler happens to already be present on this machine (expected, since this is the existing dev box), the confirmation prompts won't trigger — that's fine, it confirms the "already satisfied" path works. The confirmation-prompt path itself can't be exercised here without uninstalling something; it's covered by manual review in this step and real-world use.

- [ ] **Step 5: Commit**

```bash
git add start_mac.sh
git commit -m "Rewrite start_mac.sh with hardware detection and prerequisite auto-install"
```

---

### Task 2: Add `start_linux.sh`

**Files:**
- Create: `start_linux.sh`

**Interfaces:**
- Consumes: same behavioral pattern established in Task 1 (`start_mac.sh`), adapted for Linux package managers.
- Produces: an executable `start_linux.sh` in the repo root, invoked as `./start_linux.sh`. Same `[OK] Ollama running` / `Ready:  http://localhost:8743` output contract as Task 1, referenced by Tasks 5 and 6.

This task cannot be executed end-to-end on the macOS dev machine (no apt/dnf, no `/proc/meminfo`). Verification here is limited to syntax-checking; real execution needs a Linux machine.

- [ ] **Step 1: Write the new script**

Create `start_linux.sh`:

```bash
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
python3 -m pip install -r requirements.txt -q
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
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x start_linux.sh
```

- [ ] **Step 3: Syntax-check**

Run: `bash -n start_linux.sh`
Expected: no output (exit code 0)

- [ ] **Step 4: Note the verification gap**

No Linux machine is available in this environment. Add a line to the PR/commit description (or note it directly to the user) that `start_linux.sh` needs a real run on a Linux box (apt-based and, ideally, dnf-based) before being trusted, matching the design doc's "Testing / verification" section.

- [ ] **Step 5: Commit**

```bash
git add start_linux.sh
git commit -m "Add start_linux.sh with hardware detection and prerequisite auto-install"
```

---

### Task 3: Add Ollama and Build Tools auto-install to `start.ps1`

**Files:**
- Modify: `start.ps1`

**Interfaces:**
- Consumes: nothing from Tasks 1–2 (independent platform).
- Produces: same output contract — `[OK] Ollama running` (implicitly, via the existing model-check block) and the existing `Ready:  http://localhost:$Port` line — referenced by Tasks 5 and 6.

No PowerShell interpreter is available in this environment (`pwsh` not installed) — this task is reviewed by careful reading, not executed. It needs a real run on Windows before being trusted, same caveat as Task 2.

- [ ] **Step 1: Replace the existing Ollama check block**

Find this block in `start.ps1` (currently the `# ── check Ollama ──` section):

```powershell
# ── check Ollama ──────────────────────────────────────────────────────────────
try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null }
catch {
    Write-Host "ERROR: Ollama is not running." -ForegroundColor Red
    Write-Host "  Start Ollama from the system tray, or launch it from the Start menu." -ForegroundColor Yellow
    Write-Host "  Not installed? https://ollama.com/download/windows" -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit"
    exit 1
}
```

Replace it with:

```powershell
# ── check Ollama installed ──────────────────────────────────────────────────────
$OllamaInstalled = [bool](Get-Command "ollama" -ErrorAction SilentlyContinue)
if (-not $OllamaInstalled) {
    Write-Host "  [--] Ollama not found." -ForegroundColor Yellow
    $Resp = Read-Host "      Install Ollama now via winget? (~200 MB) [y/N]"
    if ($Resp -match '^[Yy]') {
        winget install --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
        Write-Host "      Ollama installed. It may need a moment to start its background service." -ForegroundColor Cyan
        Start-Sleep -Seconds 5
    } else {
        Write-Host "ERROR: Ollama is required. Install manually: https://ollama.com/download/windows" -ForegroundColor Red
        Read-Host "`nPress Enter to exit"
        exit 1
    }
}

# ── check Ollama running ────────────────────────────────────────────────────────
try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null }
catch {
    Write-Host "ERROR: Ollama is not running." -ForegroundColor Red
    Write-Host "  Start Ollama from the system tray, or launch it from the Start menu." -ForegroundColor Yellow
    Write-Host "  Not installed? https://ollama.com/download/windows" -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit"
    exit 1
}

# ── check C++ Build Tools (needed for native pip extensions) ────────────────────
function Test-VCBuildTools {
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) { return $false }
    $Installed = & $VsWhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    return [bool]$Installed
}

if (-not (Test-VCBuildTools)) {
    Write-Host "  [--] C++ Build Tools not found (needed to compile some Python packages)." -ForegroundColor Yellow
    $Resp = Read-Host "      Install Visual Studio Build Tools now via winget? This is a 1-2 GB download and can take several minutes. [y/N]"
    if ($Resp -match '^[Yy]') {
        winget install --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-package-agreements --accept-source-agreements --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        Write-Host "      Build Tools installed." -ForegroundColor Green
    } else {
        Write-Host "      Skipping. pip install may fail without a compiler." -ForegroundColor Yellow
    }
} else {
    Write-Host "  [OK] C++ Build Tools found" -ForegroundColor Green
}
Write-Host ""
```

- [ ] **Step 2: Review placement**

Open `start.ps1` and confirm the new block sits between the existing `# ── check Python ──` block and the existing `# ── verify / pull required models ──` block — same position the old Ollama check occupied. Confirm there is exactly one `# ── check Ollama installed ──` block and one `# ── check Ollama running ──` block (no duplicate leftover from the original).

- [ ] **Step 3: Commit**

```bash
git add start.ps1
git commit -m "Add winget-based Ollama and Build Tools auto-install to start.ps1"
```

---

### Task 4: Add GitHub Actions release workflow

**Files:**
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: nothing from Tasks 1–3 directly, but the zips it produces will contain whatever `start_mac.sh` / `start_linux.sh` / `start.ps1` look like at the tagged commit — so this task should land after Tasks 1–3 are committed.
- Produces: on any tag push matching `v*`, a GitHub Release containing `geo-assist-windows-<tag>.zip`, `geo-assist-macos-<tag>.zip`, `geo-assist-linux-<tag>.zip`.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  contents: write

jobs:
  build-and-release:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Determine version
        id: version
        run: echo "version=${GITHUB_REF_NAME}" >> "$GITHUB_OUTPUT"

      - name: Build release archives
        run: |
          VERSION="${{ steps.version.outputs.version }}"
          for OS in windows macos linux; do
            git archive --format=zip --output="geo-assist-${OS}-${VERSION}.zip" HEAD
          done

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: |
            geo-assist-windows-${{ steps.version.outputs.version }}.zip
            geo-assist-macos-${{ steps.version.outputs.version }}.zip
            geo-assist-linux-${{ steps.version.outputs.version }}.zip
          generate_release_notes: true
```

- [ ] **Step 2: Validate YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))" && echo VALID`
Expected: `VALID` printed, no exception. (Uses Python's PyYAML — if not installed, run `pip install pyyaml` first, or validate at https://github.com/<repo> UI after pushing, which lints workflow YAML automatically.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "Add GitHub Actions workflow to publish 3 OS-labeled release zips on tag push"
```

- [ ] **Step 4: Verify with a real tag push — ask the user first**

This step pushes a tag to the shared GitHub remote and creates a visible Release, so **stop and ask the user for explicit go-ahead before running it**, per the design doc's testing section. Once approved:

```bash
git tag v0.0.1-test
git push origin v0.0.1-test
```

Then check the Actions tab and the Releases page on GitHub to confirm all three zips were attached correctly. Afterward, clean up the test artifacts (again, confirm with the user before deleting anything on the remote):

```bash
git push origin :refs/tags/v0.0.1-test
git tag -d v0.0.1-test
gh release delete v0.0.1-test --yes
```

---

### Task 5: Rewrite README.md

**Files:**
- Modify: `README.md` (full rewrite of the sections listed below; "Usage", "Optional features", "Configuration", "Bulk re-indexing", and "Running tests" sections are unaffected and stay exactly as they are today)

**Interfaces:**
- Consumes: exact script filenames from Tasks 1–3 (`start_mac.sh`, `start_linux.sh`, `start.ps1`/`start.bat`) and their behavior (Ollama check, compiler check, model pull, browser open).

- [ ] **Step 1: Replace the top of the file through the end of "Starting"**

Replace everything from the start of the file through the end of the existing "Starting" section (i.e. everything before `## Usage`) with:

```markdown
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
```

- [ ] **Step 2: Confirm the rest of the file is untouched**

Run: `grep -n "^## " README.md`
Expected output includes, in order: `## Download`, `## Prerequisites`, `## Starting`, `### Windows`, `### macOS`, `### Linux`, `## Usage`, `## Optional features`, `## Configuration`, `## Bulk re-indexing`, `## Running tests`. If `## Usage` onward is missing or reordered, the replacement in Step 1 went too far or not far enough — fix before continuing.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Rewrite README download/prerequisites/starting sections for 3-platform scripts"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: same script names/behaviors as Task 5.

- [ ] **Step 1: Replace the "Running" section**

Find the existing `## Running` section (from `## Running` through the end of the `### macOS / Linux (dev only)` code block, i.e. everything before `### OCR support (optional)`) and replace it with:

```markdown
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
```

- [ ] **Step 2: Add a Releases note**

Immediately after the `## Running` section (i.e. right before `### OCR support (optional)`), add:

```markdown
## Releases

Pushing a tag matching `v*` (e.g. `git tag v1.1.0 && git push origin v1.1.0`) triggers `.github/workflows/release.yml`, which zips the repo three times via `git archive` (identical content, one zip per OS label) and publishes them to a GitHub Release. All three zips ship all three start scripts — the filename is what tells the user which one is theirs.
```

- [ ] **Step 3: Confirm section order**

Run: `grep -n "^## \|^### " CLAUDE.md | head -30`
Expected: `## Running` followed by `### Windows (primary...`, `### macOS`, `### Linux`, `### Prerequisite auto-install pattern`, `### Manual run (any OS, dev only)`, then `## Releases`, then `### OCR support (optional)` (unchanged, further down).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "Document 3-platform running instructions and release process in CLAUDE.md"
```

---

## Post-plan follow-up (not part of this plan's tasks)

- Real execution verification of `start_linux.sh` on an actual Linux machine, and `start.ps1`'s new winget blocks on an actual Windows machine — flagged as a known gap in the design doc, can't be closed from this environment.
- Cutting the first real release (`git tag v1.0.0 ...`) once the above verification is done.
