# Cross-platform releases + prerequisite auto-install

## Problem

Geo-Assist currently only has a polished startup path for Windows (`start.bat` →
`start.ps1`: hardware detection, dependency install, model pulls, browser
launch). `start_mac.sh` is a 4-line script with none of that. There is no
Linux script at all. The repo has no GitHub Releases — the only way to get the
app today is `git clone` against a single `main` branch.

Separately, first-time Windows setup requires the user to manually install
Ollama and Visual Studio Build Tools (needed because `chromadb`,
`sentence-transformers`, and `easyocr` may compile native extensions if no
prebuilt wheel matches the installed Python version) before `start.ps1` will
succeed. Nothing in the script surfaces or automates this.

## Goals

1. Bring macOS and Linux startup scripts to parity with `start.ps1`.
2. Auto-detect missing prerequisites (Ollama, a C compiler) on all three
   platforms and offer to install them, with user confirmation before any
   download.
3. Publish tagged GitHub Releases with three OS-labeled zips so "download
   Geo-Assist for your OS" is a real, discoverable action.
4. Update README/CLAUDE.md so instructions match the new scripts instead of
   describing manual steps the scripts now perform.

## Non-goals

- No compiled/bundled installer (no `.exe`/`.pkg`/single-file binary that
  embeds Python). The app remains a source tree + start script per OS.
- No auto-install of optional features (OCR, reranker, audio) — those stay
  manual opt-in via `pip install -r requirements-*.txt`, unchanged.
- No CI test matrix running the actual scripts on real Windows/Linux/macOS
  runners. Verification is manual on real hardware after this ships (see
  Testing section).

## Design

### 1. Common script pattern (all three platforms)

Each script performs, in order:

1. **Hardware detection** (unchanged behavior, extended to Linux):
   - CPU: physical/logical core count (`nproc` on Linux, existing WMI query
     on Windows, `sysctl` on macOS)
   - GPU: NVIDIA via `nvidia-smi` on Windows/Linux; macOS has no discrete GPU
     path since Ollama uses Metal automatically on Apple Silicon and there's
     nothing to detect/toggle
2. **Ollama check** — is `127.0.0.1:11434/api/tags` reachable?
   - If not reachable and the binary isn't found either: print what's
     missing, ask the user to confirm (`y/N`), then install:
     - Windows: `winget install Ollama.Ollama`
     - macOS: `brew install ollama` if Homebrew is present; otherwise print
       the ollama.com/download link (a `.dmg` GUI install can't be scripted
       without brew)
     - Linux: `curl -fsSL https://ollama.com/install.sh | sh` (the official
       installer script, detects distro itself)
   - If the user declines, print the manual install link and exit cleanly
     (same failure mode as today, just clearer).
3. **C compiler check** — look for `cl.exe` (Windows, via `vswhere` or PATH),
   `cc`/`gcc` (Linux), `cc` (macOS, part of Xcode CLT).
   - If missing: print why it's needed (native extension builds for
     chromadb/sentence-transformers/easyocr), ask to confirm, then install:
     - Windows: `winget install Microsoft.VisualStudio.2022.BuildTools`
       with a silent override adding only the C++ workload
       (`--add Microsoft.VisualStudio.Workload.VCTools`)
     - macOS: `xcode-select --install` — this always opens Apple's own GUI
       dialog; the script triggers it and cannot make it silent, then waits
       for it to complete before continuing
     - Linux: `build-essential` via `apt`, or `gcc`/`python3-devel` via `dnf`,
       whichever package manager is detected (`command -v apt` / `dnf`)
   - If the user declines, continue anyway — pip install may fail later with
     a compiler error; that's an acceptable outcome for a declined optional
     install.
4. Existing behavior, unchanged: verify/pull required Ollama models, `pip
   install -r requirements.txt`, start uvicorn, poll `/health` until ready,
   open the browser.

### 2. Per-file changes

- **`start.ps1`**: add steps 2–3 above (winget-based). Existing hardware
  detection, model pull, dependency install, health check, browser open stay
  as-is.
- **`start_mac.sh`**: rewritten from its current 4 lines into a full script
  following the common pattern (hardware detection, Ollama check, Xcode CLT
  check, model pull, dependency install, health check, browser open via
  `open`).
- **`start_linux.sh`** (new): same pattern, `xdg-open` for the browser,
  apt/dnf detection for the compiler install.
- **`start.bat`**: unchanged (already just delegates to `start.ps1`).

### 3. GitHub Actions release workflow

New `.github/workflows/release.yml`:

- Trigger: push of a tag matching `v*` (e.g. `v1.1.0`)
- Job: `git archive` the tagged commit into three identically-named-content
  zips — `geo-assist-windows-vX.Y.Z.zip`, `geo-assist-macos-vX.Y.Z.zip`,
  `geo-assist-linux-vX.Y.Z.zip`. `git archive` naturally excludes anything
  gitignored (`data/`, caches, etc.), so no separate exclude list is needed.
  All three zips contain the full repo, including all three start scripts —
  the filename is what tells the user which one is theirs; there's no
  per-OS content trimming to maintain.
- Publish a GitHub Release at that tag with all three zips attached.
- Going forward, cutting a release is: `git tag vX.Y.Z && git push origin
  vX.Y.Z`.

### 4. Documentation updates

**README.md** — this is a rewrite of the affected sections, not just an
addition, because the current text describes manual steps the scripts now
perform:

- New "Download" section near the top, linking to the Releases page, one
  line per OS ("Windows → download, unzip, double-click `start.bat`", etc.)
- **Prerequisites section trimmed**: it currently instructs the user to
  manually install Ollama, pull models, and `pip install` before starting.
  That framing is now wrong for anyone using a start script — those steps
  happen automatically (with a confirmation prompt for anything that
  installs software). The section becomes a short note: "the start script
  checks for and can install these for you"; the manual commands are kept
  as a collapsed/secondary reference for anyone who wants to do it by hand
  or is running the app without the provided scripts.
- **Starting section restructured** into three coequal subsections (Windows
  / macOS / Linux), each led by "download the zip / run the script" as the
  primary instruction. The current bare `python3 -m uvicorn ...` invocation
  for macOS/Linux is demoted to a secondary "or run it manually" note under
  each OS's subsection rather than being the only documented path.
- "Optional features" (OCR, reranker, audio) section is unaffected — those
  remain manual opt-in steps regardless of this change.

**CLAUDE.md**:
- "Running" section updated to document Linux as a first-class target
  alongside Windows/macOS, matching the new script.
- Short note added on the prerequisite auto-install pattern (Ollama + C
  compiler check, confirm-before-install) so a future session extending
  these scripts follows the same shape instead of reintroducing a
  bare-bones script.

## Testing / verification

Shell and PowerShell scripts aren't covered by the existing pytest suite
(which mocks Ollama/embeddings and never touches the startup scripts). For
this change:

- `bash -n` / `zsh -n` syntax-check `start_mac.sh` and `start_linux.sh`;
  `start_mac.sh` gets an actual run on the development macOS machine.
- `start.ps1` and `start_linux.sh` cannot be executed end-to-end from this
  (macOS) dev environment — their winget/apt/dnf install paths need manual
  verification on real Windows and Linux machines after this ships. This is
  a known limitation, not something this plan can close.
- The GitHub Actions workflow can be verified by pushing a test tag (e.g.
  `v0.0.1-test`) to a scratch branch/tag and confirming the Release and its
  three zips are produced correctly, then deleting the test tag/release.

## Open risk

`winget install Microsoft.VisualStudio.2022.BuildTools` is a large (1–2 GB)
download and can take several minutes; the confirmation prompt should say so
explicitly rather than leaving the user staring at a silent wait.
