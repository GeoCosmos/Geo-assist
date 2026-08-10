# Shutdown button + OCR toggle

## Problem

Today the only way to stop Geo-Assist is Ctrl+C / closing the terminal
window that `start.ps1`/`start_mac.sh`/`start_linux.sh` opened. None of the
three scripts touch Ollama — it (and whatever model it has resident, e.g.
`qwen3.5:4b` kept loaded via `OLLAMA_MAX_LOADED_MODELS=2`) keeps running
afterward, consuming RAM on a 16 GB CPU-only target machine until the user
notices and stops it by hand. There is no in-app way to shut down at all —
you have to go find the terminal window.

Separately, OCR (`GEO_OCR=true` + `pip install -r requirements-ocr.txt`) can
currently only be turned on by setting an environment variable before
launch and restarting the whole app. There's no way to flip it on/off during
a session, and no visibility into whether `easyocr` is even installed.

## Goals

1. A one-click way, from the web UI, to stop Geo-Assist **and** shut down
   Ollama on the same machine.
2. A runtime toggle for OCR that takes effect immediately (next ingest),
   with no restart, and that clearly communicates when `easyocr` isn't
   installed instead of failing silently.
3. Document both in the README; note both as implemented in CLAUDE.md's
   Phase 2 ideas list, matching the existing style.

## Non-goals

- Not touching the Ctrl+C / window-close path in the three start scripts —
  the UI button is the shutdown mechanism; the scripts' existing cleanup
  (kill the uvicorn process) is unchanged.
- Not auto-installing `easyocr` from the UI. The toggle only flips the
  existing `config.OCR_ENABLED` flag; if the package isn't present, the
  response tells the user the exact `pip install` command to run.
- Not persisting the OCR toggle across restarts. It's in-memory only; the
  `GEO_OCR` env var remains the "default at launch" mechanism, documented
  in README/CLAUDE.md already. Restarting reverts to that default.
- No guaranteed kill of Ollama when it's installed/run as a system service
  under a different OS user (e.g. some Linux systemd installs) — that
  requires `sudo` and can't be done non-interactively from a web request.
  This is called out as a known limitation in the README, not solved here.

## Design

### 1. Shutdown endpoint (`main.py`)

`POST /shutdown`:

- Best-effort kill of Ollama, dispatched on `platform.system()`:
  - **Windows**: `taskkill /F /IM ollama.exe` and `taskkill /F /IM "ollama app.exe"`
    (the CLI/server binary and the tray app both need killing or the tray
    app respawns the server).
  - **macOS**: `pkill -x ollama` (the server binary) and
    `osascript -e 'quit app "Ollama"'` (the menu-bar app, if running as a
    GUI app rather than a bare CLI `ollama serve`).
  - **Linux**: `pkill -x ollama`. (`systemctl stop ollama` is not attempted —
    it needs `sudo`, which can't be supplied interactively here; this is the
    known limitation above.)
  - All subprocess calls wrapped in `try/except`, output discarded, exit
    codes ignored — "best effort," never raises.
- Returns `{"status": "shutting down"}` immediately.
- Schedules a `BackgroundTask` that sleeps briefly (~0.3s, to let the
  response flush to the socket) then sends `SIGTERM` to its own process
  (`os.kill(os.getpid(), signal.SIGTERM)`). This is the same signal uvicorn
  already handles gracefully for Ctrl+C, so it reuses the existing shutdown
  path rather than introducing a new one — `start.ps1`'s
  `$proc.WaitForExit()`/`finally` block and `start_mac.sh`/`start_linux.sh`'s
  `trap cleanup EXIT` all still fire correctly, since from their point of
  view the child process just exited.

### 2. Shutdown button (`static/index.html`)

- New button in the sidebar, below the Documents section (bottom of
  `#sidebar`), styled as a danger action reusing the same visual language as
  `#clear-all-btn` (red border/text, subtle fill on hover).
- `onclick`: `confirm("This will stop Geo-Assist and shut down Ollama on this machine. Continue?")`
  → on confirm, `POST /shutdown`, then replace `#app`'s content with a
  static "Geo-Assist has been shut down — you can close this tab." message
  and clear the `setInterval(pollHealth, ...)` timer so it doesn't spam
  failed requests against a dead server.
- No change to `pollHealth` itself — a dead server after shutdown just shows
  "Backend offline" if the user doesn't click through the button (e.g. they
  killed it via Ctrl+C instead), which is already correct behavior.

### 3. OCR status/toggle endpoints (`main.py`)

- `GET /ocr/status` → `{"enabled": config.OCR_ENABLED, "available": <bool>}`.
  `available` is `importlib.util.find_spec("easyocr") is not None` — a cheap
  check, does not import or load the model.
- `POST /ocr/toggle` with body `{"enabled": bool}`:
  - If `enabled=True` and `available=False` → `400` with a message
    containing the exact install command
    (`pip install -r requirements-ocr.txt`).
  - Otherwise sets `config.OCR_ENABLED = enabled` in memory and returns the
    new status. Takes effect on the next file ingested; does not reprocess
    already-ingested documents (matches existing OCR behavior — it's a
    per-ingest decision today too).

### 4. Fix `retriever.py`'s frozen OCR flag

`_FIGURE_NOTE` (retriever.py ~line 527) is currently computed once at module
import time from `config.OCR_ENABLED`, so toggling the flag at runtime would
have no effect on the system prompt. Convert it to a small function called
from `_build_system()` at request time:

```python
def _figure_note() -> str:
    return (
        "Some blocks begin with \"[Figure on page N]:\" ... "
    ) if config.OCR_ENABLED else ""
```

so a runtime toggle takes effect on the very next query, not just the next
ingest.

### 5. OCR toggle switch (`static/index.html`)

- Small labeled switch in the sidebar, near the Documents section header
  (same row as "Clear all" / "All folders", or directly below it — final
  placement decided during implementation to fit the existing cramped
  header row).
- On page load, `GET /ocr/status` sets the switch's initial position; if
  `available=false`, the switch is disabled and its tooltip shows the
  install command.
- On toggle, `POST /ocr/toggle`; on `400` (not installed — a race if the
  status check was stale), revert the switch and show the install-command
  toast via the existing `showToast(msg, isError)` helper.

### 6. Documentation

**README.md**:
- New "Shutting down" section (near "Usage"): describes the sidebar button
  and what it does (stops the server, stops Ollama), plus the Linux/system-
  service caveat.
- "OCR" section under "Optional features" gets a short addition describing
  the in-app toggle as the normal way to turn it on/off once
  `requirements-ocr.txt` is installed, keeping the `GEO_OCR=true` env var
  documented as the "default at every launch" option.

**CLAUDE.md**:
- Add two new entries to the "Phase 2 ideas (not yet implemented)" list,
  struck through as implemented, matching the existing style (e.g. the OCR
  and rerank entries already there), pointing at `main.py`'s `/shutdown` and
  `/ocr/*` endpoints and the `retriever._figure_note()` fix.

## Testing

- `tests/test_api.py`: `POST /shutdown` — assert it returns 200 and that the
  Ollama-kill subprocess calls are attempted (mock `subprocess.run`); do not
  let the test actually send `SIGTERM` to the pytest process — mock
  `os.kill` too.
- `tests/test_api.py`: `GET /ocr/status` and `POST /ocr/toggle` — cover
  enabled/disabled, and the 400 path when `easyocr` availability is mocked
  false.
- `tests/test_retriever.py`: verify `_figure_note()` reflects
  `config.OCR_ENABLED` when flipped at runtime (regression test for the
  frozen-at-import bug being fixed).
- Frontend: manual verification only (no JS test harness in this repo) —
  start the app, click the shutdown button, confirm both the server process
  and `ollama ps`/Activity Monitor show Ollama stopped; toggle OCR on/off
  and confirm an ingested image is/isn't OCR'd accordingly.

## Open risk

Killing Ollama process-wide from a web request has no confirmation from the
OS about *why* it's running — if the user has some other app also depending
on the same local Ollama instance, this button stops that too. The
in-browser `confirm()` dialog text says "on this machine" specifically to
flag that this isn't scoped to Geo-Assist alone.
