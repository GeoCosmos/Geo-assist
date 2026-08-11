# Citations and Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group citations by document, stop offering links to source files that do not exist, and show extracted figures beneath the answers that cited them.

**Architecture:** `retriever` accumulates one source entry per document instead of one per `(doc_id, page)`, carrying `pages`, `has_original`, and `figures`. A new route serves figure JPEGs from a path built server-side. The frontend renders grouped chips and thumbnails, and keeps rendering the old payload shape so saved conversations survive.

**Tech Stack:** Python 3.10+, FastAPI, vanilla JS (no build step), pytest.

## Global Constraints

- **No new pip dependencies.**
- **Saved conversations must keep rendering.** Messages and their `sources` persist to `localStorage` under `ga-convs` and replay through `renderMsgRow`. The renderer must accept both the old `{filename, page, doc_id}` shape and the new one.
- **No client-supplied paths.** The figure route takes `doc_id`, `page`, `idx` and builds the path itself.
- **`page` stays on every source entry**, equal to the lowest cited page, so any consumer reading the old field keeps working.
- Air-gap holds: no new network calls.
- Every change ships with tests (`CLAUDE.md`: "Every new feature or bug fix must include tests").

---

### Task 1: `has_original` and honest 404 messages

Both concern whether a document's source file exists, so they share a test cycle.

**Files:**
- Modify: `ingest.py` (add `has_original` and `figure_index` near `_remove_files`, line ~1106)
- Modify: `main.py:243-250` (the `/documents/{doc_id}/file` route)
- Test: `tests/test_api.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces:
  - `ingest.has_original(doc_id: str) -> bool`
  - `ingest.figure_index(chunk_index: int) -> int` — recovers an image's per-page index from its stored `chunk_index`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
def test_has_original_true_after_ingest_writes_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "originals"))
    os.makedirs(config.ORIGINALS_DIR)
    open(os.path.join(config.ORIGINALS_DIR, "abc123def4567890.pdf"), "wb").write(b"x")
    assert ingest.has_original("abc123def4567890") is True


def test_has_original_false_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "originals"))
    os.makedirs(config.ORIGINALS_DIR)
    assert ingest.has_original("abc123def4567890") is False


def test_has_original_false_when_directory_missing(tmp_path, monkeypatch):
    """The deployment case: /app/data was never persisted, so nothing exists."""
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "gone"))
    assert ingest.has_original("abc123def4567890") is False


def test_figure_index_inverts_the_chunk_index_encoding():
    for idx in (0, 1, 7):
        chunk_index = ingest._IMAGE_CHUNK_IDX_BASE - idx
        assert ingest.figure_index(chunk_index) == idx
```

Add `import os` to `tests/test_ingest.py` if absent.

Append to `tests/test_api.py`:

```python
def test_file_route_reports_deployment_fault_when_directory_missing(tmp_path, monkeypatch):
    """A missing originals directory is a storage-configuration problem, not a
    property of the document — the old message asserted the wrong cause."""
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "never-created"))
    r = client.get("/documents/abc123def4567890/file")
    assert r.status_code == 404
    assert "not persisted" in r.json()["detail"]


def test_file_route_reports_per_document_absence_when_directory_exists(tmp_path, monkeypatch):
    originals = tmp_path / "originals"
    originals.mkdir()
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(originals))
    r = client.get("/documents/abc123def4567890/file")
    assert r.status_code == 404
    assert "No source file stored for this document" in r.json()["detail"]


def test_file_route_still_rejects_a_malformed_doc_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path))
    assert client.get("/documents/..%2F..%2Fetc/file").status_code == 404
    assert client.get("/documents/nothex/file").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ingest.py -k "has_original or figure_index" tests/test_api.py -k "file_route" -v`
Expected: FAIL — `AttributeError: module 'ingest' has no attribute 'has_original'`, and the message assertions fail against the current wording.

- [ ] **Step 3: Add the helpers to `ingest.py`**

Insert directly above `def _remove_files(doc_id: str) -> None:` (line ~1106):

```python
def has_original(doc_id: str) -> bool:
    """Whether this document's source file is on disk.

    Used to decide whether a citation should be a link. Without it the UI offers
    every citation as a link and the only way to find a dead one is to click it
    and get raw JSON in a new tab.
    """
    return bool(glob.glob(os.path.join(config.ORIGINALS_DIR, f"{doc_id}.*")))


def figure_index(chunk_index: int) -> int:
    """Recover an image's per-page index from its stored chunk_index.

    Inverts the encoding in `_analyze_and_store_images`, which writes
    `_IMAGE_CHUNK_IDX_BASE - img_idx` so image chunks never collide with text
    chunks (>= 0) or the summary chunk (-1).
    """
    return _IMAGE_CHUNK_IDX_BASE - chunk_index
```

- [ ] **Step 4: Fix the route messages in `main.py`**

Replace the body of `get_document_file` (lines 244-250):

```python
@app.get("/documents/{doc_id}/file")
async def get_document_file(doc_id: str):
    if not _DOC_ID_RE.match(doc_id):
        raise HTTPException(404, "Document not found")
    # Distinguish a storage-configuration fault from a per-document one. The
    # previous single message asserted "ingested before this feature was added"
    # even when the entire originals directory was missing — which happens when
    # the data directory is not persisted, and is not a property of the document.
    if not os.path.isdir(config.ORIGINALS_DIR):
        raise HTTPException(
            404,
            "Source files are not available on this server — the data directory "
            "is not persisted.",
        )
    matches = glob.glob(os.path.join(config.ORIGINALS_DIR, f"{doc_id}.*"))
    if not matches:
        raise HTTPException(
            404,
            "No source file stored for this document. It was ingested before "
            "source files were kept, or the file was removed.",
        )
    return FileResponse(matches[0], headers={"Content-Disposition": "inline"})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ingest.py tests/test_api.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ingest.py main.py tests/test_ingest.py tests/test_api.py
git commit -m "fix: distinguish missing originals directory from missing source file"
```

---

### Task 2: Group sources by document, with figures

**Files:**
- Modify: `retriever.py` — three source-append sites (lines ~88, ~540, ~759)
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `ingest.has_original(doc_id)`, `ingest.figure_index(chunk_index)`
- Produces: source entries shaped
  ```python
  {"filename": str, "doc_id": str, "page": int,        # lowest cited page
   "pages": [int, ...],                                 # sorted, deduped
   "has_original": bool,
   "figures": [{"page": int, "idx": int}, ...]}         # key absent when empty
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retriever.py`:

```python
import ingest
import retriever


def test_source_accumulator_groups_pages_under_one_document():
    acc = retriever._source_acc()
    for page in (10, 1, 30, 10):
        retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "m.pdf", "page": page})
    out = retriever._finalize_sources(acc)
    assert len(out) == 1
    assert out[0]["pages"] == [1, 10, 30]
    assert out[0]["page"] == 1          # legacy field: lowest cited page


def test_source_accumulator_preserves_first_citation_order():
    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "b" * 16, "filename": "second.pdf", "page": 2})
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "first.pdf", "page": 1})
    retriever._add_source(acc, {"doc_id": "b" * 16, "filename": "second.pdf", "page": 9})
    assert [s["filename"] for s in retriever._finalize_sources(acc)] == ["second.pdf", "first.pdf"]


def test_figure_chunk_attaches_to_the_same_document_entry():
    """The bug grouping fixes: under the old (doc_id, page) dedup a figure on a
    page that also contributed a text chunk was dropped, so it never surfaced."""
    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "m.pdf", "page": 3})
    retriever._add_source(acc, {
        "doc_id": "a" * 16, "filename": "m.pdf", "page": 3,
        "chunk_type": "image",
        "chunk_index": ingest._IMAGE_CHUNK_IDX_BASE - 0,
    })
    out = retriever._finalize_sources(acc)
    assert len(out) == 1
    assert out[0]["figures"] == [{"page": 3, "idx": 0}]


def test_figures_are_deduped():
    acc = retriever._source_acc()
    meta = {"doc_id": "a" * 16, "filename": "m.pdf", "page": 3,
            "chunk_type": "image", "chunk_index": ingest._IMAGE_CHUNK_IDX_BASE - 2}
    retriever._add_source(acc, meta)
    retriever._add_source(acc, dict(meta))
    assert retriever._finalize_sources(acc)[0]["figures"] == [{"page": 3, "idx": 2}]


def test_documents_without_figures_omit_the_key():
    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "m.pdf", "page": 1})
    assert "figures" not in retriever._finalize_sources(acc)[0]


def test_has_original_is_reported_per_document(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "originals"))
    os.makedirs(config.ORIGINALS_DIR)
    open(os.path.join(config.ORIGINALS_DIR, f"{'a' * 16}.pdf"), "wb").write(b"x")

    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "kept.pdf", "page": 1})
    retriever._add_source(acc, {"doc_id": "c" * 16, "filename": "gone.pdf", "page": 1})
    out = {s["filename"]: s["has_original"] for s in retriever._finalize_sources(acc)}
    assert out == {"kept.pdf": True, "gone.pdf": False}
```

Add `import os` and `import config` to `tests/test_retriever.py` if absent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_retriever.py -k "source_acc or figure or has_original or accumulator" -v`
Expected: FAIL — `AttributeError: module 'retriever' has no attribute '_source_acc'`

- [ ] **Step 3: Add the accumulator helpers to `retriever.py`**

Insert above `_collect` (near line 700):

```python
def _source_acc() -> dict:
    """Accumulator for citation sources, keyed by doc_id.

    One entry per document rather than per (doc_id, page): a document cited on
    three pages previously produced three chips, and — more importantly — a
    figure on a page that had also contributed a text chunk was dropped by the
    old dedup, so it could never be displayed.

    Python dicts preserve insertion order, so first-citation order (most
    relevant first) is retained for free.
    """
    return {}


def _add_source(acc: dict, meta: dict) -> None:
    """Record one retrieved chunk's provenance against its document."""
    doc_id = meta["doc_id"]
    entry = acc.get(doc_id)
    if entry is None:
        entry = {"filename": meta["filename"], "doc_id": doc_id, "pages": [], "figures": []}
        acc[doc_id] = entry

    page = meta.get("page", 1)
    if page not in entry["pages"]:
        entry["pages"].append(page)

    if meta.get("chunk_type") == "image":
        ref = {"page": page, "idx": ingest.figure_index(meta.get("chunk_index", 0))}
        if ref not in entry["figures"]:
            entry["figures"].append(ref)


def _finalize_sources(acc: dict) -> list[dict]:
    """Turn the accumulator into the payload sent to the client."""
    out = []
    for entry in acc.values():
        pages = sorted(entry["pages"])
        source = {
            "filename": entry["filename"],
            "doc_id": entry["doc_id"],
            "pages": pages,
            # Retained so anything reading the pre-grouping field keeps working.
            "page": pages[0] if pages else 1,
            "has_original": ingest.has_original(entry["doc_id"]),
        }
        if entry["figures"]:
            source["figures"] = entry["figures"]
        out.append(source)
    return out
```

- [ ] **Step 4: Run the new tests**

Run: `python3 -m pytest tests/test_retriever.py -k "source_acc or figure or has_original or accumulator" -v`
Expected: PASS

- [ ] **Step 5: Switch the three source-building sites to the accumulator**

In `_collect` (around line 754), replace:

```python
        context_parts.append(f"[{meta['filename']}, page {meta['page']}]\n{text}")
        key = (meta["doc_id"], meta["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"filename": meta["filename"], "page": meta["page"], "doc_id": meta["doc_id"]})

    return context_parts, sources
```

with:

```python
        context_parts.append(f"[{meta['filename']}, page {meta['page']}]\n{text}")
        _add_source(source_acc, meta)

    return context_parts, _finalize_sources(source_acc)
```

and change that function's initialisation line `context_parts, sources, seen = [], [], set()` to:

```python
    context_parts, source_acc = [], _source_acc()
```

At line ~88 (the catalog path), replace:

```python
        sources.append({"filename": doc["filename"], "page": 1, "doc_id": doc["doc_id"]})
```

with:

```python
        _add_source(source_acc, {"filename": doc["filename"], "page": 1, "doc_id": doc["doc_id"]})
```

initialising `source_acc = _source_acc()` where `sources = []` was, and yielding
`_finalize_sources(source_acc)` where `sources` was yielded.

At line ~540 (the named-document path), apply the same three changes: replace the
`sources = []` initialisation with `source_acc = _source_acc()`, replace the
`sources.append({...})` call with `_add_source(source_acc, meta)`, and return
`_finalize_sources(source_acc)`.

- [ ] **Step 6: Run the whole suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, no regressions. If a pre-existing test asserts on `sources[i]["page"]`, it still passes — `page` is retained.

- [ ] **Step 7: Commit**

```bash
git add retriever.py tests/test_retriever.py
git commit -m "feat: group citation sources by document and carry figure refs"
```

---

### Task 3: Figure route

**Files:**
- Modify: `main.py` (add beside the `/file` route, after line ~258)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `ingest._image_path(doc_id, page, idx)`, `_DOC_ID_RE`
- Produces: `GET /documents/{doc_id}/figures/{page}/{idx}`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_figure_route_serves_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    doc_id = "a" * 16
    os.makedirs(os.path.join(config.IMAGES_DIR, doc_id))
    open(os.path.join(config.IMAGES_DIR, doc_id, "p3_i0.jpg"), "wb").write(b"\xff\xd8jpegbytes")

    r = client.get(f"/documents/{doc_id}/figures/3/0")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8jpegbytes"


def test_figure_route_404s_for_missing_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    assert client.get(f"/documents/{'a' * 16}/figures/9/9").status_code == 404


def test_figure_route_rejects_malformed_doc_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    assert client.get("/documents/nothex/figures/1/0").status_code == 404


def test_figure_route_rejects_non_integer_page_and_idx(tmp_path, monkeypatch):
    """FastAPI's int path converter rejects these before any path is built."""
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    assert client.get(f"/documents/{'a' * 16}/figures/..%2F..%2Fetc/0").status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_api.py -k figure_route -v`
Expected: FAIL — 404 on every route (not yet registered)

- [ ] **Step 3: Implement the route**

Add to `main.py` after `get_document_file`:

```python
@app.get("/documents/{doc_id}/figures/{page}/{idx}")
async def get_document_figure(doc_id: str, page: int, idx: int):
    """Serve one extracted figure.

    The path is built from three validated components rather than accepted from
    the client, so there is no traversal surface: doc_id must be 16 hex chars and
    page/idx are integers by FastAPI's path converter.
    """
    if not _DOC_ID_RE.match(doc_id):
        raise HTTPException(404, "Document not found")
    path = ingest._image_path(doc_id, page, idx)
    if not os.path.isfile(path):
        raise HTTPException(404, "Figure not available")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Content-Disposition": "inline"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_api.py -k figure_route -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: serve extracted figures by doc_id, page, and index"
```

---

### Task 4: Render grouped citations and figure thumbnails

**Files:**
- Modify: `static/index.html` — `renderSourceChip` (line ~1564) and the `.source-chip` styles (line ~478)

- [ ] **Step 1: Replace `renderSourceChip`**

```javascript
function renderSourceChip(s) {
  // Sources replayed from localStorage predate grouping and carry a single
  // `page` with no `pages` array. Without this branch every saved conversation
  // loses its citations.
  if (!Array.isArray(s.pages)) return renderLegacySourceChip(s);

  const name = escHtml(s.filename);
  const pages = s.pages.filter(p => p > 0);

  let pagePart = '';
  if (pages.length) {
    const rendered = pages.map(p => s.has_original
      ? `<a class="src-page" href="/documents/${encodeURIComponent(s.doc_id)}/file#page=${p}"
             target="_blank" rel="noopener">${p}</a>`
      : `<span class="src-page">${p}</span>`).join(', ');
    pagePart = ` · p.${rendered}`;
  }

  const title = s.has_original ? '' :
    ' title="Source file not stored for this document — nothing to open."';
  const chip = `<span class="source-chip${s.has_original ? '' : ' no-file'}"${title}>${name}${pagePart}</span>`;

  const figures = (s.figures || []).map(f =>
    `<a href="/documents/${encodeURIComponent(s.doc_id)}/figures/${f.page}/${f.idx}"
        target="_blank" rel="noopener" class="fig-thumb">
       <img src="/documents/${encodeURIComponent(s.doc_id)}/figures/${f.page}/${f.idx}"
            alt="Figure from page ${f.page}" loading="lazy"
            onerror="this.parentNode.style.display='none'">
     </a>`).join('');

  return chip + (figures ? `<span class="fig-row">${figures}</span>` : '');
}

function renderLegacySourceChip(s) {
  const label = `${escHtml(s.filename)}${s.page > 0 ? `, p.${s.page}` : ''}`;
  if (!s.doc_id) return `<span class="source-chip">${label}</span>`;
  const href = `/documents/${encodeURIComponent(s.doc_id)}/file${s.page > 0 ? `#page=${s.page}` : ''}`;
  return `<a class="source-chip" href="${href}" target="_blank" rel="noopener">${label}</a>`;
}
```

Both call sites (`sources.map(renderSourceChip)` at lines ~1440 and ~1537) are unchanged.

- [ ] **Step 2: Add the styles**

Insert after the existing `.source-chip::before` rule (line ~492):

```css
.src-page { color: var(--accent); text-decoration: none; }
a.src-page:hover { text-decoration: underline; }
/* No source file stored: the chip stays readable but is visibly not a link. */
.source-chip.no-file { opacity: .65; }
.source-chip.no-file .src-page { color: inherit; }
.fig-row { display: flex; gap: 6px; flex-basis: 100%; margin: 2px 0 4px; }
.fig-thumb img {
  max-height: 64px; max-width: 96px;
  border: 1px solid rgba(14,165,233,.25); border-radius: 4px;
  object-fit: cover; display: block;
}
.fig-thumb:hover img { border-color: var(--accent); }
```

- [ ] **Step 3: Syntax-check the page**

```bash
cd ~/Desktop/geo-assist
python3 - <<'EOF'
import re, pathlib
html = pathlib.Path("static/index.html").read_text()
pathlib.Path("/tmp/ga_check.js").write_text(
    "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)))
EOF
node --check /tmp/ga_check.js && echo "JS SYNTAX OK" && rm /tmp/ga_check.js
```

Expected: `JS SYNTAX OK`

- [ ] **Step 4: Verify both payload shapes render**

```bash
cd ~/Desktop/geo-assist
node -e '
const fs=require("fs");
const html=fs.readFileSync("static/index.html","utf8");
const js=html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];
const ctx={};
new Function("exports", js.replace(/^loadConvs\(\);[\s\S]*$/m,"") +
  ";exports.r=renderSourceChip;")(ctx);
console.log("GROUPED:", ctx.r({filename:"m.pdf",doc_id:"a".repeat(16),
  pages:[1,10,30],has_original:true,figures:[{page:3,idx:0}]}).slice(0,200));
console.log("NO FILE:", ctx.r({filename:"g.pdf",doc_id:"b".repeat(16),
  pages:[2],has_original:false}).slice(0,160));
console.log("LEGACY :", ctx.r({filename:"old.pdf",doc_id:"c".repeat(16),page:4}));
'
```

Expected: grouped chip lists `1, 10, 30` as links plus a thumbnail; the no-file chip carries `no-file` and no `<a href>` to `/file`; the legacy shape renders as an `<a class="source-chip">` exactly as before.

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: group citation chips by document and show cited figures"
```

---

## Self-Review

**Spec coverage:** §1 grouping → Task 2; §2 `has_original` → Tasks 1 and 2; §3 figure refs → Task 2; §4 figure route → Task 3; §5 honest messages → Task 1; §6 rendering incl. backward compatibility → Task 4; §7 OCR is deployment-only and carries no code, so it has no task — it is a Dockerfile and compose change recorded in the spec.

**Type consistency:** `has_original` (bool), `pages` (sorted list[int]), `figures` (list of `{page, idx}`) are used identically in Tasks 2, 3, and 4. `ingest.figure_index` is defined in Task 1 and consumed in Task 2. The figure URL `/documents/{doc_id}/figures/{page}/{idx}` matches between the route in Task 3 and the frontend in Task 4.

**Deliberately not covered:** the per-document figure gallery, vision captioning, and an OCR backfill pass — all out of scope per the spec.
