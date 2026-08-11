# Citations and Figures — Design

**Date:** 2026-08-11
**Status:** Approved

## Problem

Three separate complaints about the same strip of UI under each answer:

1. **Citations are bulky and repetitive.** `retriever.py` emits one source per
   `(doc_id, page)`, so a document cited on three pages produces three chips. A
   real answer produced six chips over four rows, three of them the same file.
2. **Citation links can 404.** Clicking a chip navigates to
   `/documents/{doc_id}/file`, which returns raw JSON —
   `{"detail":"Original file not available (ingested before this feature was added)"}` —
   in a new tab. The UI renders every chip as a link whether or not a source file
   exists, so the only way to discover a dead one is to click it.
3. **Extracted figures are invisible.** Ingest pulls images out of every
   PDF/DOCX/PPTX and writes JPEGs to `data/images/{doc_id}/`, but no route serves
   them and no view displays them. `image_path` is written into chunk metadata and
   read by nothing.

## Root cause of the 404 — partially open

`main.py:247` globs `data/originals/{doc_id}.*` and 404s when nothing matches. The
file is genuinely absent. Two causes are possible and not yet distinguished:

- **Historical.** Originals storage and click-to-open citations both landed in
  `acf83df` (2026-07-02). Anything ingested earlier never had a source file saved.
- **Erased.** `data/originals/` sits under `/app/data`, which the deployment does
  not bind-mount, so every image rebuild deletes it. Documents stay listed because
  chunks live in Qdrant, a separate container with its own bind.

Distinguishing them needs VM access. This design does not depend on the answer:
the `has_original` flag below makes the state visible in the UI for every
document at once, which is a better diagnostic than the command would have been.
Persisting `/app/data` remains a prerequisite either way; recovering source files
for already-ingested documents requires re-ingesting them.

## Constraints

- **No new pip dependencies** for the figure work. OCR uses the existing
  `requirements-ocr.txt`.
- **Saved conversations must keep rendering.** Messages and their `sources` are
  persisted to `localStorage` under `ga-convs` and replayed through
  `renderMsgRow` on load. Changing the source payload shape breaks every past
  conversation unless the renderer accepts both shapes.
- **No client-supplied paths.** The figure route takes `doc_id`, `page`, and
  `idx` and builds the path itself.
- Air-gap holds: no new network calls.

## Design

### 1. Group sources by document (`retriever.py`)

`_collect` currently dedups on `(doc_id, page)` and appends
`{filename, page, doc_id}`. It instead accumulates one entry per `doc_id`:

```python
{
  "filename": "d102909_IS452_BIOS_Configuration.pdf",
  "doc_id":   "a1b2c3d4e5f60718",
  "pages":    [1, 10, 30],          # sorted, deduped
  "has_original": True,
  "figures":  [{"page": 3, "idx": 0}]   # omitted when empty
}
```

`page` is retained alongside `pages` as the lowest cited page, so any consumer
reading the old field keeps working. Order is preserved by first citation, which
keeps the most relevant document first.

### 2. `has_original` (`ingest.py` → `retriever.py`)

A new `ingest.has_original(doc_id: str) -> bool` wraps the same glob the route
uses, so the check lives beside `ORIGINALS_DIR` rather than being duplicated.
`retriever` already imports `ingest`, so no new coupling. Cost is one glob per
cited document per answer — a handful of stat calls, negligible next to
retrieval.

The UI renders page numbers as links only when this is true. Dead links stop
being reachable rather than being explained after the fact.

### 3. Figure references (`retriever.py`)

When a retrieved chunk carries `chunk_type == "image"`, its image index is
recovered from the stored `chunk_index` (`idx = -100 - chunk_index`, inverting
`ingest._IMAGE_CHUNK_IDX_BASE`) and appended to that document's `figures` list,
deduped on `(page, idx)`.

**The subtlety this exists to avoid:** grouping by document rather than by
`(doc_id, page)` is what makes this work at all. Under the old dedup, a figure on
page 3 of a document that also contributed a text chunk from page 3 was silently
dropped, so the thumbnail would never appear.

### 4. Figure route (`main.py`)

```
GET /documents/{doc_id}/figures/{page}/{idx}
```

Validates `doc_id` against the existing `_DOC_ID_RE`, takes `page` and `idx` as
integers, and serves `ingest._image_path(doc_id, page, idx)` as a `FileResponse`.
Nothing from the client reaches the filesystem as a path, so there is no
traversal surface. 404 when the file is absent.

### 5. Honest error messages (`main.py`)

The existing `/documents/{doc_id}/file` route asserts a cause it never checked.
It now distinguishes:

- `ORIGINALS_DIR` missing entirely → "Source files are not available on this
  server — the data directory is not persisted." A deployment fault, not a
  property of the document.
- Directory present, no match → "No source file stored for this document. It was
  ingested before source files were kept, or the file was removed."

### 6. Rendering (`static/index.html`)

`renderSourceChip` becomes `renderSource`, emitting one chip per document:

```
📄 d102909_IS452_BIOS_Configuration.pdf · p.1, 10, 30
```

Each page number is an individual link to `#page=N` when `has_original` is true;
otherwise the chip renders as plain text with a `title` explaining why. Figures
render as a row of ≤64px thumbnails beneath the chips, each linking to the
full-size image, with `onerror` hiding any that fail to load.

**Backward compatibility:** the renderer branches on whether `pages` is present.
Sources replayed from `localStorage` in the old `{filename, page, doc_id}` shape
render exactly as they do today. This is not optional — without it every saved
conversation loses its citations.

### 7. OCR enablement (deployment, no code)

Already implemented behind `GEO_OCR`; it needs three things in the deployment:

- `pip install -r requirements-ocr.txt` in the app Dockerfile (~200 MB)
- the easyocr model pre-downloaded **during the image build**, or it fetches to
  an ephemeral `~/.EasyOCR` on first use, re-downloading on every restart and
  requiring internet at runtime
- `GEO_OCR=true` in compose

OCR runs inside `_analyze_and_store_images`, a fire-and-forget task that starts
after text chunks commit, so text stays queryable while captions fill in behind.

**Timing matters and is a one-way door.** No backfill pass exists —
`caption_status="pending"` was designed for one that was never built. Figures
ingested with OCR off stay uncaptioned unless their document is re-ingested. So
OCR must be enabled *before* the bulk NAS scan, or the scan has to be redone.

## Error handling

- Figure route: 404 on a missing file; the UI hides broken thumbnails via
  `onerror` so a partially-wiped `data/images/` degrades to no thumbnails rather
  than broken-image icons.
- `has_original` false: chip renders as text, no link, with an explanatory
  `title`.
- Documents ingested before figures were persisted have no `figures` key; the
  renderer omits the thumbnail row entirely.

## Testing

- `retriever`: sources group by document with sorted deduped `pages`; `page`
  still present and equal to the lowest; figure chunks attach to `figures` and
  survive a text chunk from the same page (the bug the grouping fixes);
  `has_original` reflects the presence of a file in `ORIGINALS_DIR`.
- `main`: figure route serves bytes, 404s on unknown doc/page/idx, rejects a
  non-hex `doc_id`; `/file` returns the deployment-fault message when
  `ORIGINALS_DIR` is absent and the per-document message when it is present but
  empty.
- Frontend has no test harness; the old-shape rendering path is verified by
  loading a conversation saved before the change.

## Out of scope

Per-document figure gallery (cheap to add later once the route exists), vision
captioning of diagrams, and any OCR backfill pass. Vision captioning was
explicitly rejected: per-image inference at ingest is the wrong cost to add to a
bulk NAS scan.

## Open questions

1. **The app `Dockerfile`.** Not tracked in this repo — the deployment builds
   from `/opt/geo-assist/app/`. Section 7 is mostly an edit to that file, which
   cannot be written until it is available.
2. **Which of the two 404 causes applies.** Does not block this work; the
   `has_original` flag will answer it once deployed.
