# Shutdown Button + OCR Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sidebar "Shut down" button that stops Geo-Assist and kills the local Ollama process, and a runtime OCR on/off toggle that replaces the restart-required `GEO_OCR` env var flow for day-to-day use.

**Architecture:** Two new small endpoint groups in `main.py` (`/shutdown`, `/ocr/status`, `/ocr/toggle`), a one-line fix in `retriever.py` so the OCR system-prompt note reads `config.OCR_ENABLED` live instead of once at import, and matching sidebar controls in `static/index.html`. No new files, no new dependencies.

**Tech Stack:** FastAPI (`BackgroundTasks`), Python stdlib (`platform`, `subprocess`, `signal`, `importlib.util`), vanilla JS/CSS in the single-file frontend.

## Global Constraints

- No new files — everything lands in `main.py`, `retriever.py`, `static/index.html`, `tests/test_api.py`, `tests/test_retriever.py`, `README.md`, `CLAUDE.md`.
- The OCR toggle is in-memory only (not persisted across restarts); `GEO_OCR` env var remains the launch-time default, unchanged.
- Killing Ollama is best-effort — never raises, never blocks the `/shutdown` response.
- The three start scripts (`start.ps1`, `start_mac.sh`, `start_linux.sh`) are NOT modified — their existing Ctrl+C/window-close cleanup is untouched.
- Every new endpoint needs a `tests/test_api.py` test per repo convention (CLAUDE.md: "every new feature or bug fix must include tests").

---

### Task 1: `/shutdown` endpoint

**Files:**
- Modify: `main.py:1-20` (imports), append new endpoint after `clear_all_documents` (`main.py:179-183`)
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `POST /shutdown` → `{"status": "shutting down"}` (200). Internal helpers `_stop_ollama()` and `_terminate_self()` in `main.py`, not used elsewhere.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py` (needs `patch` already imported at top of the file — it is):

```python
def test_shutdown_kills_ollama_and_schedules_exit():
    with patch("main.subprocess.run") as mock_run, \
         patch("main.os.kill") as mock_kill, \
         patch("main.asyncio.sleep", new_callable=AsyncMock):
        r = client.post("/shutdown")

    assert r.status_code == 200
    assert r.json() == {"status": "shutting down"}
    assert mock_run.called, "expected an attempt to kill the Ollama process"
    mock_kill.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_api.py::test_shutdown_kills_ollama_and_schedules_exit -v`
Expected: FAIL with `404` (no such route) or `AttributeError` — the endpoint doesn't exist yet.

- [ ] **Step 3: Add imports to `main.py`**

Change the top of `main.py` from:

```python
import asyncio
import glob
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
```

to:

```python
import asyncio
import glob
import importlib.util
import json
import logging
import os
import platform
import re
import signal
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
```

- [ ] **Step 4: Implement the endpoint**

Add this after `clear_all_documents` (right after `main.py:183`, before the `_background_tasks: set = set()` line):

```python
def _stop_ollama() -> None:
    """Best-effort kill of the local Ollama process(es). Never raises.

    Windows ships both the server binary and a tray app that respawns it,
    so both image names are targeted. macOS's menu-bar app is killed via
    `osascript` (graceful quit) in addition to the bare server binary.
    Linux systemd installs running Ollama under a different user require
    `sudo systemctl stop ollama` run by hand — that can't be done
    non-interactively from a web request.
    """
    system = platform.system()
    try:
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], capture_output=True)
        elif system == "Darwin":
            subprocess.run(["pkill", "-x", "ollama"], capture_output=True)
            subprocess.run(["osascript", "-e", 'quit app "Ollama"'], capture_output=True)
        else:
            subprocess.run(["pkill", "-x", "ollama"], capture_output=True)
    except FileNotFoundError:
        pass


async def _terminate_self() -> None:
    """Exit this process the same way Ctrl+C would, after the HTTP response has flushed."""
    await asyncio.sleep(0.3)
    os.kill(os.getpid(), signal.SIGTERM)


@app.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    _stop_ollama()
    background_tasks.add_task(_terminate_self)
    return {"status": "shutting down"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_api.py::test_shutdown_kills_ollama_and_schedules_exit -v`
Expected: PASS. (Test runs in well under a second since `asyncio.sleep` is mocked — if it hangs or takes ~0.3s+, the mock target string is wrong; confirm it's `"main.asyncio.sleep"` not `"asyncio.sleep"`.)

- [ ] **Step 6: Run the full test suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: all tests PASS (same count as before plus 1).

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: add /shutdown endpoint that stops Ollama and exits"
```

---

### Task 2: OCR status/toggle endpoints + fix frozen `_FIGURE_NOTE`

**Files:**
- Modify: `main.py` (append endpoints after Task 1's `/shutdown`)
- Modify: `retriever.py:525-532` (constant → function), `retriever.py:1191`, `retriever.py:1195` (call sites)
- Test: `tests/test_api.py`, `tests/test_retriever.py`

**Interfaces:**
- Consumes: `config.OCR_ENABLED` (existing, `config.py:18`)
- Produces: `GET /ocr/status` → `{"enabled": bool, "available": bool}`; `POST /ocr/toggle` body `{"enabled": bool}` → `{"enabled": bool}` (200) or `400` with a `detail` string containing `"requirements-ocr.txt"`. `retriever._figure_note() -> str` replaces the old `retriever._FIGURE_NOTE` constant — later frontend tasks don't touch this, but any future code must call it as a function, not read it as a constant.

- [ ] **Step 1: Write the failing tests**

Add `import config` to the top of `tests/test_api.py` (it currently only imports `json`, `patch`/`AsyncMock`, `TestClient`, `main`):

```python
import config
```

Add to `tests/test_api.py`:

```python
def test_ocr_status_shape():
    r = client.get("/ocr/status")
    assert r.status_code == 200
    body = r.json()
    assert "enabled" in body
    assert "available" in body


def test_ocr_toggle_rejected_when_not_installed(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    monkeypatch.setattr("main.importlib.util.find_spec", lambda name: None)
    r = client.post("/ocr/toggle", json={"enabled": True})
    assert r.status_code == 400
    assert "requirements-ocr.txt" in r.json()["detail"]
    assert config.OCR_ENABLED is False


def test_ocr_toggle_enable_when_installed(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    monkeypatch.setattr("main.importlib.util.find_spec", lambda name: object())
    r = client.post("/ocr/toggle", json={"enabled": True})
    assert r.status_code == 200
    assert r.json()["enabled"] is True
    assert config.OCR_ENABLED is True


def test_ocr_toggle_disable_always_allowed(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr("main.importlib.util.find_spec", lambda name: None)
    r = client.post("/ocr/toggle", json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    assert config.OCR_ENABLED is False
```

(`monkeypatch.setattr(config, "OCR_ENABLED", ...)` establishes the value pytest restores after the test, so the endpoint's in-test mutation of `config.OCR_ENABLED` never leaks into other tests.)

Add to `tests/test_retriever.py` (needs `config` and `retriever` already imported at the top of that file — they are):

```python
def test_figure_note_reflects_runtime_toggle(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    assert retriever._figure_note() == ""

    monkeypatch.setattr(config, "OCR_ENABLED", True)
    assert "[Figure on page" in retriever._figure_note()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_api.py -k ocr -v tests/test_retriever.py -k figure_note -v`
Expected: FAIL — `/ocr/status` and `/ocr/toggle` don't exist (404s), and `retriever._figure_note` doesn't exist (`AttributeError`, since today it's `retriever._FIGURE_NOTE`, a constant not a function).

- [ ] **Step 3: Add the OCR endpoints to `main.py`**

Append after the `/shutdown` endpoint added in Task 1:

```python
@app.get("/ocr/status")
async def ocr_status():
    return {
        "enabled": config.OCR_ENABLED,
        "available": importlib.util.find_spec("easyocr") is not None,
    }


class OcrToggleRequest(BaseModel):
    enabled: bool


@app.post("/ocr/toggle")
async def ocr_toggle(req: OcrToggleRequest):
    if req.enabled and importlib.util.find_spec("easyocr") is None:
        raise HTTPException(
            400,
            "easyocr is not installed. Run: pip install -r requirements-ocr.txt",
        )
    config.OCR_ENABLED = req.enabled
    return {"enabled": config.OCR_ENABLED}
```

- [ ] **Step 4: Fix the frozen `_FIGURE_NOTE` in `retriever.py`**

Replace (`retriever.py:525-532`):

```python
# Included in the system prompt only when OCR is enabled, so the LLM is not
# primed to look for figure annotations that will never appear.
_FIGURE_NOTE = (
    "Some blocks begin with \"[Figure on page N]:\" — these are OCR-extracted text "
    "from images found on that document page. Treat them as you would textual content: "
    "cite the source file and page when referencing them, and flag any uncertainty if the "
    "extracted text seems garbled or incomplete.\n\n"
) if config.OCR_ENABLED else ""
```

with:

```python
# Included in the system prompt only when OCR is enabled, so the LLM is not
# primed to look for figure annotations that will never appear. A function
# (not a module-level constant) so a runtime OCR toggle takes effect on the
# very next query instead of requiring a restart.
def _figure_note() -> str:
    return (
        "Some blocks begin with \"[Figure on page N]:\" — these are OCR-extracted text "
        "from images found on that document page. Treat them as you would textual content: "
        "cite the source file and page when referencing them, and flag any uncertainty if the "
        "extracted text seems garbled or incomplete.\n\n"
    ) if config.OCR_ENABLED else ""
```

Then update both call sites in `_build_system` (`retriever.py:1167-1195`). Change:

```python
            figure_note=_FIGURE_NOTE,
```

to:

```python
            figure_note=_figure_note(),
```

and change:

```python
    return _SYSTEM.format(context=context, figure_note=_FIGURE_NOTE, source_list=source_list)
```

to:

```python
    return _SYSTEM.format(context=context, figure_note=_figure_note(), source_list=source_list)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_api.py -k ocr -v` and `python3 -m pytest tests/test_retriever.py -k figure_note -v`
Expected: all PASS.

- [ ] **Step 6: Run the full test suite to confirm no regressions**

Run: `python3 -m pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add main.py retriever.py tests/test_api.py tests/test_retriever.py
git commit -m "feat: add OCR runtime toggle endpoints, fix frozen OCR system-prompt flag"
```

---

### Task 3: Sidebar shutdown button (frontend)

**Files:**
- Modify: `static/index.html` (CSS in `<style>`, HTML in `#sidebar`, JS at bottom of `<script>`)

**Interfaces:**
- Consumes: `POST /shutdown` from Task 1.
- Produces: global JS function `shutdownApp()`, DOM ids `#sidebar-footer`, `#shutdown-btn`, `#shutdown-screen`, top-level variable `_healthPollTimer`.

- [ ] **Step 1: Store the health-poll interval handle so it can be cancelled**

In `static/index.html`, change (around line 694-695):

```javascript
pollHealth();
setInterval(pollHealth, 8000);
```

to:

```javascript
pollHealth();
const _healthPollTimer = setInterval(pollHealth, 8000);
```

- [ ] **Step 2: Add CSS for the button and post-shutdown screen**

Insert immediately after the `#clear-all-btn:hover` rule (`static/index.html:248`):

```css
#sidebar-footer {
  padding: 8px 12px 12px;
  border-top: 1px solid rgba(14,165,233,.08);
  flex-shrink: 0;
}
#shutdown-btn {
  width: 100%;
  background: none;
  border: 1px solid rgba(239,68,68,.25);
  border-radius: 6px;
  color: rgba(239,68,68,.75);
  font-size: 12px;
  padding: 7px 0;
  cursor: pointer;
  transition: background .12s, border-color .12s;
}
#shutdown-btn:hover:not(:disabled) { background: rgba(239,68,68,.08); border-color: rgba(239,68,68,.5); color: #f87171; }
#shutdown-btn:disabled { opacity: .6; cursor: default; }

#shutdown-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  width: 100%;
  color: var(--text);
  text-align: center;
  gap: 8px;
}
#shutdown-screen p { color: var(--text-muted); }
```

- [ ] **Step 3: Add the button to the sidebar**

In `static/index.html`, the `#doc-section` div currently closes and `#sidebar` (`nav`) closes right after (`static/index.html:632-634`):

```html
      <div id="doc-list"><div style="padding:6px 16px;color:#4a4f5a;font-size:12px">No documents yet</div></div>
    </div>
  </nav>
```

Change to:

```html
      <div id="doc-list"><div style="padding:6px 16px;color:#4a4f5a;font-size:12px">No documents yet</div></div>
    </div>
    <div id="sidebar-footer">
      <button id="shutdown-btn" onclick="shutdownApp()">⏻ Shut down (stops Ollama too)</button>
    </div>
  </nav>
```

- [ ] **Step 4: Add the `shutdownApp()` function**

Add anywhere in the `<script>` block after `showToast` is defined (e.g. right after the `showToast` function, `static/index.html:1018-1023`):

```javascript
async function shutdownApp() {
  if (!confirm('This will stop Geo-Assist and shut down Ollama on this machine. Continue?')) return;
  const btn = document.getElementById('shutdown-btn');
  btn.disabled = true;
  btn.textContent = 'Shutting down…';
  try {
    await _apiFetch(`${API}/shutdown`, { method: 'POST' });
  } catch {
    // the connection can drop before the response arrives once the server exits — expected
  }
  clearInterval(_healthPollTimer);
  document.getElementById('app').innerHTML =
    '<div id="shutdown-screen"><h2>Geo-Assist has been shut down</h2>' +
    '<p>Ollama has been stopped. You can close this tab.</p></div>';
}
```

- [ ] **Step 5: Manual verification**

Run: `GEO_CHAT_MODEL=qwen3.5:4b python3 -m uvicorn main:app --host 127.0.0.1 --port 8743` (requires a real Ollama running locally — this step is manual, not part of the pytest suite, same as every other frontend change in this repo).

In the browser at `http://localhost:8743`:
1. Confirm the "⏻ Shut down (stops Ollama too)" button renders at the bottom of the sidebar.
2. Click it, confirm the browser `confirm()` dialog text, accept it.
3. Confirm the sidebar/chat UI is replaced with the "Geo-Assist has been shut down" screen.
4. In a terminal, confirm the uvicorn process has exited (the process you started in this step is gone) and `ollama ps` / Activity Monitor / Task Manager shows no `ollama` process running.

- [ ] **Step 6: Commit**

```bash
git add static/index.html
git commit -m "feat: add sidebar shutdown button that stops Ollama"
```

---

### Task 4: OCR toggle switch (frontend)

**Files:**
- Modify: `static/index.html` (CSS, HTML, JS)

**Interfaces:**
- Consumes: `GET /ocr/status`, `POST /ocr/toggle` from Task 2; `showToast(msg, isError)` (existing, `static/index.html:1018`).
- Produces: global JS functions `loadOcrStatus()`, `toggleOcr(enabled)`; DOM ids `#ocr-toggle-row`, `#ocr-toggle-checkbox`.

- [ ] **Step 1: Add CSS for the toggle row**

Insert immediately after the `.section-label` rule (`static/index.html:203-211`):

```css
#ocr-toggle-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 12px 4px;
  font-size: 11px;
  color: var(--sidebar-muted);
  flex-shrink: 0;
}
#ocr-toggle-row input[type="checkbox"] { accent-color: var(--accent); cursor: pointer; }
#ocr-toggle-row label { cursor: pointer; flex: 1; }
#ocr-toggle-row input:disabled, #ocr-toggle-row input:disabled + label { cursor: not-allowed; opacity: .5; }
```

- [ ] **Step 2: Add the toggle row to the sidebar**

The Documents header row and doc-list currently read (`static/index.html:625-632`):

```html
      <div style="display:flex;align-items:center;padding:5px 12px 0;">
        <div class="section-label" style="padding:0;flex:1">Documents</div>
        <div id="scope-bar" style="padding:0">
          <button id="clear-all-btn" onclick="clearAllDocs()">Clear all</button>
          <button id="scope-btn" onclick="toggleScope()">All folders</button>
        </div>
      </div>
      <div id="doc-list"><div style="padding:6px 16px;color:#4a4f5a;font-size:12px">No documents yet</div></div>
```

Change to:

```html
      <div style="display:flex;align-items:center;padding:5px 12px 0;">
        <div class="section-label" style="padding:0;flex:1">Documents</div>
        <div id="scope-bar" style="padding:0">
          <button id="clear-all-btn" onclick="clearAllDocs()">Clear all</button>
          <button id="scope-btn" onclick="toggleScope()">All folders</button>
        </div>
      </div>
      <div id="ocr-toggle-row">
        <input type="checkbox" id="ocr-toggle-checkbox" onchange="toggleOcr(this.checked)">
        <label for="ocr-toggle-checkbox">🔍 OCR (image text extraction)</label>
      </div>
      <div id="doc-list"><div style="padding:6px 16px;color:#4a4f5a;font-size:12px">No documents yet</div></div>
```

- [ ] **Step 3: Add `loadOcrStatus()` and `toggleOcr()`**

Add right after the `clearAllDocs` function (`static/index.html:899-911`):

```javascript
async function loadOcrStatus() {
  const box = document.getElementById('ocr-toggle-checkbox');
  try {
    const s = await _apiFetch(`${API}/ocr/status`).then(r => r.json());
    box.checked = s.enabled;
    box.disabled = !s.available;
    box.title = s.available ? '' : 'Install with: pip install -r requirements-ocr.txt';
  } catch {
    box.disabled = true;
  }
}

async function toggleOcr(enabled) {
  const box = document.getElementById('ocr-toggle-checkbox');
  box.disabled = true;
  try {
    const r = await _apiFetch(`${API}/ocr/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!r.ok) {
      const body = await r.json();
      showToast(body.detail || 'Could not enable OCR', true);
      box.checked = !enabled;
    }
  } finally {
    box.disabled = false;
  }
}
```

- [ ] **Step 4: Call `loadOcrStatus()` on page load**

`static/index.html:925` currently reads:

```javascript
loadDocs();
```

Change to:

```javascript
loadDocs();
loadOcrStatus();
```

- [ ] **Step 5: Manual verification**

With the server running from Task 3 (or restarted):
1. Confirm the OCR checkbox appears below the Documents header row, unchecked by default (matches `GEO_OCR` default `false`).
2. If `easyocr` is not installed (`python3 -c "import easyocr"` fails), confirm the checkbox is disabled and hovering shows the install-command tooltip.
3. If `easyocr` is installed, toggle it on — confirm the checkbox stays checked and no error toast appears; toggle it off — confirm it unchecks. Reload the page and confirm the box reflects the last-set value (since the backend keeps `config.OCR_ENABLED` in memory for the life of the process, this should match — restarting the server resets it to the `GEO_OCR` env var default, which is expected).

- [ ] **Step 6: Commit**

```bash
git add static/index.html
git commit -m "feat: add OCR runtime toggle switch to sidebar"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md:90-121`
- Modify: `CLAUDE.md` (Phase 2 ideas list)

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a "Shutting down" section to README.md**

Replace (`README.md:90-99`):

```markdown
## Usage

1. Click **Upload** in the sidebar to ingest one or more documents
2. Wait for the progress bar to complete (embedding runs locally)
3. Type a question and press Enter
4. Answers include citations — source file and page number shown below each response

**Supported file types:** `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`

## Optional features
```

with:

```markdown
## Usage

1. Click **Upload** in the sidebar to ingest one or more documents
2. Wait for the progress bar to complete (embedding runs locally)
3. Type a question and press Enter
4. Answers include citations — source file and page number shown below each response

**Supported file types:** `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`

## Shutting down

Click **⏻ Shut down** at the bottom of the sidebar. This stops the Geo-Assist
server and also kills the local Ollama process, so nothing keeps running (or
keeps a model loaded in RAM) in the background after you're done. It asks for
confirmation first, since it stops Ollama machine-wide, not just for
Geo-Assist.

On Linux, if Ollama was installed as a system service running under a
different user (some `systemctl`-based installs), this can't stop it without
`sudo` — in that case stop it manually with `sudo systemctl stop ollama`.

## Optional features
```

- [ ] **Step 2: Update the OCR section**

Replace (`README.md:101-108`, the `### OCR` subsection):

```markdown
### OCR

Extracts text from images embedded in PDFs, PPTXs, and DOCXs (screenshots, UI captures, text-heavy figures):

```bash
pip install -r requirements-ocr.txt   # ~200 MB — easyocr + PyTorch CPU
GEO_OCR=true python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```
```

with:

```markdown
### OCR

Extracts text from images embedded in PDFs, PPTXs, and DOCXs (screenshots, UI captures, text-heavy figures). Install once:

```bash
pip install -r requirements-ocr.txt   # ~200 MB — easyocr + PyTorch CPU
```

Then use the **OCR** checkbox in the sidebar (next to the Documents list) to
turn it on or off for the current session — no restart needed, and it takes
effect on the next document you ingest. The checkbox is disabled with an
install hint if `easyocr` isn't installed yet.

To have OCR on by default at every launch instead, set `GEO_OCR=true` before
starting the app (see Configuration below) — the sidebar toggle still lets
you flip it off for a given session either way.
```

- [ ] **Step 3: Add Phase 2 entries to CLAUDE.md**

In `CLAUDE.md`, under the "## Phase 2 ideas (not yet implemented)" heading, the list currently ends with the OCR entry. Add two new entries after it:

```markdown
- ~~Easy shutdown (stops Ollama too)~~ — implemented. Sidebar "⏻ Shut down" button calls `POST /shutdown`, which best-effort kills the local Ollama process (`taskkill` on Windows, `pkill`/`osascript` on macOS, `pkill` on Linux) then exits itself via a `BackgroundTask`. See `main.py`'s `_stop_ollama()`, `_terminate_self()`, and the `/shutdown` route. The three start scripts are unchanged — this is a UI-triggered path, not a change to Ctrl+C/window-close behavior.
- ~~OCR runtime toggle~~ — implemented. `GET /ocr/status` / `POST /ocr/toggle` flip `config.OCR_ENABLED` in memory without a restart, replacing the old "set `GEO_OCR=true` and restart" as the normal way to use it day-to-day (the env var is still the launch-time default). `retriever._figure_note()` (previously the frozen-at-import `_FIGURE_NOTE` constant) now reads the flag per-request so the toggle takes effect on the very next query. UI: checkbox next to the Documents list in `static/index.html`, disabled with an install hint if `easyocr` isn't importable.
```

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document shutdown button and OCR toggle"
```

---

## Final verification

- [ ] Run: `python3 -m pytest tests/ -v`
  Expected: all tests pass, including the new `test_shutdown_kills_ollama_and_schedules_exit`, `test_ocr_status_shape`, `test_ocr_toggle_*`, and `test_figure_note_reflects_runtime_toggle`.
- [ ] Run: `ruff check .` (if `ruff` is installed per CLAUDE.md's linting section)
  Expected: no new lint errors in `main.py` or `retriever.py`.
- [ ] Manual: full click-through in the browser per Task 3 Step 5 and Task 4 Step 5.
