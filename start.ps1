# start.ps1 — Geo-Assist startup for Dell Precision 3260 / Windows
# Target hardware: Intel 12th-Gen Core (i5-12500 / i7-12700 / i9-12900),
#                  optional NVIDIA T400 or T600 (4 GB GDDR6), DDR5 RAM.
#
# Usage: double-click start.bat  OR  right-click -> "Run with PowerShell"
# Override any setting before launching:
#   $env:GEO_CHAT_MODEL = "llama3.1:8b"
#   $env:GEO_VISION_MODEL = "llava:7b"
#   .\start.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Port = 8743

# ── detect hardware ───────────────────────────────────────────────────────────
$cpuInfo      = Get-CimInstance -ClassName Win32_Processor
$CpuName      = $cpuInfo[0].Name.Trim()
$PhysicalCores = ($cpuInfo | Measure-Object -Property NumberOfCores -Sum).Sum
$LogicalCores  = ($cpuInfo | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum

# NVIDIA GPU — present on many Precision 3260 configs as T400 or T600
$HasGPU  = $false
$GpuName = "None"
if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    try {
        $gpuOut = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpuOut) {
            $HasGPU  = $true
            $GpuName = $gpuOut.Trim()
        }
    } catch { }
}

# RAM
$RamGB = [Math]::Round(
    ((Get-CimInstance -ClassName Win32_PhysicalMemory |
      Measure-Object -Property Capacity -Sum).Sum) / 1GB
)

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Geo-Assist  --  Local AI Engineering Assistant" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  CPU  : $CpuName"
Write-Host "         $PhysicalCores physical cores / $LogicalCores logical threads"
Write-Host "  RAM  : $RamGB GB"
if ($HasGPU) {
    Write-Host "  GPU  : $GpuName (CUDA)" -ForegroundColor Green
} else {
    Write-Host "  GPU  : not detected  (CPU-only mode)" -ForegroundColor Yellow
}
Write-Host ""

# ── hardware-tuned Ollama settings ───────────────────────────────────────────
# Thread count: physical cores give best LLM throughput on Intel hybrid
# architectures (P-cores dominate; E-cores add marginal LLM benefit).
# Cap at physical cores; leave headroom for OS and I/O.
$OllamaThreads = [Math]::Max(4, $PhysicalCores)
$env:OLLAMA_NUM_THREADS = "$OllamaThreads"

if ($HasGPU) {
    # T400 / T600: 4 GB GDDR6.
    # llama3.1:8b  (Q4_K_M ~4.7 GB): ~26 of 32 layers on GPU, rest CPU -> ~15 tok/s
    # nomic-embed-text (~274 MB): always fits alongside the chat model offload.
    $env:OLLAMA_NUM_GPU      = "999"           # fill VRAM with as many layers as fit
    $DefaultChatModel        = "llama3.1:8b"   # best accuracy for engineering Q&A
    $DefaultEmbedConcurrency = "4"             # GPU can pipeline embed batches
    $DefaultVisionConcurrency= "1"             # vision model shares VRAM; one at a time
} else {
    # CPU-only (Intel UHD 770 integrated).
    # i7-12700 does ~5 tok/s on 8B, ~14 tok/s on 3B at full DDR5 bandwidth.
    $DefaultChatModel        = "llama3.2:latest"  # 3B fits in 16 GB RAM; acceptable latency
    $DefaultEmbedConcurrency = "2"                # avoid saturating memory bandwidth
    $DefaultVisionConcurrency= "1"
}

# Keep both chat and embed models loaded simultaneously so queries don't stall
# waiting for Ollama to reload the evicted model (~30s cold load for nomic-embed-text).
$env:OLLAMA_MAX_LOADED_MODELS = "2"

# Honour explicit overrides set by the user before calling start.bat
if (-not $env:GEO_CHAT_MODEL)          { $env:GEO_CHAT_MODEL          = $DefaultChatModel }
if (-not $env:GEO_EMBED_CONCURRENCY)   { $env:GEO_EMBED_CONCURRENCY   = $DefaultEmbedConcurrency }
if (-not $env:GEO_VISION_CONCURRENCY)  { $env:GEO_VISION_CONCURRENCY  = $DefaultVisionConcurrency }

$ChatModel   = $env:GEO_CHAT_MODEL
$VisionModel = if ($env:GEO_VISION_MODEL) { $env:GEO_VISION_MODEL } else { "" }

Write-Host "  Mode    : $(if ($HasGPU) { 'GPU-accelerated (hybrid offload)' } else { 'CPU-only' })"
Write-Host "  Threads : $OllamaThreads (OLLAMA_NUM_THREADS)"
Write-Host "  Chat    : $ChatModel"
Write-Host "  Embed   : nomic-embed-text (concurrency $($env:GEO_EMBED_CONCURRENCY))"
if ($VisionModel) { Write-Host "  Vision  : $VisionModel" -ForegroundColor Cyan }
Write-Host ""

# ── check Python ──────────────────────────────────────────────────────────────
try { python --version | Out-Null }
catch {
    Write-Host "ERROR: Python not found in PATH." -ForegroundColor Red
    Write-Host "  Download from https://python.org" -ForegroundColor Yellow
    Write-Host "  During install, tick 'Add Python to PATH'." -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit"
    exit 1
}

# ── check Ollama ──────────────────────────────────────────────────────────────
try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null }
catch {
    Write-Host "ERROR: Ollama is not running." -ForegroundColor Red
    Write-Host "  Start Ollama from the system tray, or launch it from the Start menu." -ForegroundColor Yellow
    Write-Host "  Not installed? https://ollama.com/download/windows" -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit"
    exit 1
}

# ── verify / pull required models ─────────────────────────────────────────────
function Confirm-OllamaModel {
    param([string]$Name, [string]$Label)
    try {
        $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
        $hit  = $tags.models | Where-Object { $_.name -like "$Name*" }
        if ($hit) {
            Write-Host "  [OK] $Label ($Name)" -ForegroundColor Green
        } else {
            Write-Host "  [--] $Label ($Name) not found -- pulling now..." -ForegroundColor Cyan
            Write-Host "       (this may take several minutes on first run)" -ForegroundColor Gray
            & ollama pull $Name
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [!!] Could not pull $Name." -ForegroundColor Yellow
                Write-Host "       On an air-gapped machine: pull on an online machine," -ForegroundColor Yellow
                Write-Host "       then copy model files to this PC." -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "  [??] Could not verify $Name (Ollama may be starting up)." -ForegroundColor Yellow
    }
}

Write-Host "Checking models..."
Confirm-OllamaModel -Name $ChatModel          -Label "Chat"
Confirm-OllamaModel -Name "nomic-embed-text"  -Label "Embed"
if ($VisionModel) { Confirm-OllamaModel -Name $VisionModel -Label "Vision" }
Write-Host ""

# ── check ffmpeg (required for audio/video transcription) ────────────────────
if (Get-Command "ffmpeg" -ErrorAction SilentlyContinue) {
    Write-Host "  [OK] ffmpeg found" -ForegroundColor Green
} else {
    Write-Host "  [--] ffmpeg not found." -ForegroundColor Yellow
    Write-Host "       Audio/video upload will not work without it." -ForegroundColor Yellow
    Write-Host "       Install: winget install ffmpeg  (or download from ffmpeg.org)" -ForegroundColor Yellow
}
Write-Host ""

# ── install / verify Python dependencies ─────────────────────────────────────
Write-Host "Checking Python dependencies..."
python -m pip install -r requirements.txt -q --no-warn-script-location
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed. Check requirements.txt and your Python version." -ForegroundColor Red
    Read-Host "`nPress Enter to exit"
    exit 1
}
Write-Host "  Dependencies OK" -ForegroundColor Green
Write-Host ""

# ── start uvicorn ─────────────────────────────────────────────────────────────
# --loop asyncio: explicit asyncio event loop (uvloop is Unix-only; this is
#   the correct Windows backend and avoids any ProactorEventLoop ambiguity).
# --workers 1: FastAPI + ChromaDB share in-process state; multiple workers
#   would each open a separate ChromaDB connection and BM25 index.
# --no-access-log: reduces noise; errors still appear.
Write-Host "Starting Geo-Assist server..."
$proc = Start-Process python `
    -ArgumentList @(
        "-m", "uvicorn", "main:app",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--workers", "1",
        "--loop", "asyncio",
        "--no-access-log"
    ) `
    -PassThru -NoNewWindow

# ── wait for server ready ─────────────────────────────────────────────────────
$ready = $false
Write-Host "Waiting for server to be ready." -NoNewline
for ($i = 0; $i -lt 40; $i++) {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch {
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 1
    }
}
Write-Host ""

if (-not $ready) {
    Write-Host "ERROR: Server did not become ready within 40 seconds." -ForegroundColor Red
    try { $proc.Kill() } catch { }
    Read-Host "`nPress Enter to exit"
    exit 1
}

# ── auto-ingest built-in knowledge guide on first run ────────────────────────
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health"
    $Guide  = Join-Path $ScriptDir "GEO-ASSIST-KNOWLEDGE.txt"
    if ($health.chunks_stored -eq 0 -and (Test-Path $Guide)) {
        Write-Host "Loading built-in knowledge guide..."
        # Use python + httpx (always installed via requirements.txt) so that
        # file paths with spaces and backslashes are handled correctly.
        python -c @"
import sys, httpx
path, port = sys.argv[1], sys.argv[2]
with open(path, 'rb') as f:
    r = httpx.post(f'http://127.0.0.1:{port}/ingest', files={'file': ('file', f)})
status = r.json().get('status', '?')
print(f'  Guide status: {status}')
"@ "$Guide" "$Port"
    }
} catch {
    Write-Host "  Note: Could not auto-load knowledge guide -- upload it manually if needed." -ForegroundColor Yellow
}

# ── open browser ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Ready:  http://localhost:$Port" -ForegroundColor Green
Write-Host "  Press Ctrl+C in this window to stop." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Start-Process "http://localhost:$Port"

# ── keep running until Ctrl+C or window close ─────────────────────────────────
# The finally block ensures uvicorn is always killed when the script exits,
# even on abrupt window close -- prevents ghost processes on Windows.
try {
    $proc.WaitForExit()
} finally {
    if (-not $proc.HasExited) {
        Write-Host "`nShutting down Geo-Assist..." -ForegroundColor Yellow
        $proc.Kill()
    }
}
