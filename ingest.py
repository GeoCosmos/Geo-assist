import asyncio
import glob
import hashlib
import io
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bm25_index
import config
import llm
import store

log = logging.getLogger(__name__)

_lock = asyncio.Lock()
_background_tasks: set = set()
_doc_cache: list[dict] | None = None

# Document parsing (PyMuPDF, python-docx, python-pptx) is pure CPU work that
# holds the GIL. Running it inline in an async function — as this module used to —
# meant `_PREPARE_CONCURRENCY = 16` bought no parallelism at all: every file was
# parsed one after another on the event loop, which also stalled `/ingest/status`
# polls and any in-flight query. A process pool gives real multi-core throughput
# on the target i7-12700 and keeps the loop responsive.
_parse_pool: ProcessPoolExecutor | None = None


def _pool() -> ProcessPoolExecutor:
    global _parse_pool
    if _parse_pool is None:
        _parse_pool = ProcessPoolExecutor(max_workers=config.PREPARE_CONCURRENCY)
    return _parse_pool


def shutdown_pool() -> None:
    global _parse_pool
    if _parse_pool is not None:
        _parse_pool.shutdown(wait=False, cancel_futures=True)
        _parse_pool = None


def _invalidate_cache() -> None:
    global _doc_cache
    _doc_cache = None


# ── text extraction ───────────────────────────────────────────────────────────

def _table_to_markdown(table) -> str:
    """Convert a PyMuPDF table object to a markdown table string."""
    rows = table.extract()
    if not rows:
        return ""
    md = []
    for i, row in enumerate(rows):
        cells = [str(c or "").replace("|", "\\|").replace("\n", " ").strip() for c in row]
        md.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(md)


def _pdf_body_font_size(page_dict: dict) -> float:
    """Return the median font size on a page — used as the body text baseline."""
    sizes = [
        span["size"]
        for block in page_dict.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"].strip()
    ]
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _is_heading_block(block: dict, heading_min_size: float) -> bool:
    """True when a dict-mode block looks like a section heading."""
    spans = [
        s for line in block.get("lines", [])
        for s in line.get("spans", [])
        if s["text"].strip()
    ]
    if not spans:
        return False
    full_text = "".join(s["text"] for s in spans).strip()
    if not full_text or len(full_text) > 150:
        return False
    avg_size = sum(s["size"] for s in spans) / len(spans)
    if avg_size >= heading_min_size:
        return True
    # Bold + short = heading even at body size (e.g. "Rev History" in same-size bold)
    return len(full_text) <= 80 and all(
        "Bold" in s.get("font", "") or "bold" in s.get("font", "") for s in spans
    )


def _extract_pdf(data: bytes) -> list[tuple[int, str]]:
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        tables = page.find_tables()
        table_rects = [fitz.Rect(t.bbox) for t in tables.tables] if tables.tables else []

        page_dict = page.get_text("dict")
        body_size = _pdf_body_font_size(page_dict)
        heading_min = body_size * 1.15

        # (y0, text, is_heading) — interleave text blocks and table markdown in reading order
        content: list[tuple[float, str, bool]] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_rect = fitz.Rect(block["bbox"])
            if table_rects and any(block_rect.intersects(tr) for tr in table_rects):
                continue
            lines_text = [
                "".join(s["text"] for s in line.get("spans", []))
                for line in block.get("lines", [])
            ]
            block_text = "\n".join(t for t in lines_text if t.strip())
            if not block_text.strip():
                continue
            content.append((block["bbox"][1], block_text, _is_heading_block(block, heading_min)))

        for table_idx, t in enumerate(tables.tables):
            md = _table_to_markdown(t)
            if md:
                content.append((t.bbox[1], f"Table {table_idx}\n{md}", False))

        content.sort(key=lambda x: x[0])

        current_heading: str | None = None
        parts: list[str] = []
        for _, text, is_heading in content:
            stripped = text.strip()
            if not stripped:
                continue
            if is_heading:
                current_heading = stripped
            else:
                parts.append(f"{current_heading}\n{stripped}" if current_heading else stripped)
        if not parts and current_heading:
            parts.append(current_heading)

        pages.append((i + 1, "\n\n".join(parts)))

    return pages


def _extract_docx(data: bytes) -> list[tuple[int, str]]:
    from docx import Document
    doc = Document(io.BytesIO(data))
    current_heading: str | None = None
    paras: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            current_heading = text
        else:
            paras.append(f"{current_heading}\n{text}" if current_heading else text)
    # Flush a trailing heading that had no content under it
    if current_heading and (not paras or not paras[-1].startswith(current_heading)):
        paras.append(current_heading)
    for tbl_idx, table in enumerate(doc.tables):
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("|", "\\|").replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        if rows:
            # "Table N" prefix mirrors PDF extraction so _tag_table_chunks can
            # detect DOCX tables and assign table_id for completeness injection.
            paras.append(f"Table {tbl_idx}\n" + "\n".join(rows))
    return [(1, "\n\n".join(paras))]


def _pptx_shape_texts(shape) -> list[str]:
    """Recursively extract text from a shape, its table cells, and grouped children."""
    parts = []
    if shape.has_text_frame:
        t = shape.text_frame.text
        if t.strip():
            parts.append(t)
    if shape.has_table:
        rows = []
        for i, row in enumerate(shape.table.rows):
            cells = [cell.text.strip().replace("|", "\\|").replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        if rows:
            parts.append("\n".join(rows))
    if hasattr(shape, "shapes"):
        for child in shape.shapes:
            parts.extend(_pptx_shape_texts(child))
    return parts


def _extract_pptx(data: bytes) -> list[tuple[int, str]]:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    pages = []
    for i, slide in enumerate(prs.slides):
        parts = []
        for shape in slide.shapes:
            parts.extend(_pptx_shape_texts(shape))
        text = "\n".join(parts)
        if text:
            pages.append((i + 1, text))
    return pages


_TELEMETRY_ROW = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}")
_TABLE_LABEL_RE = re.compile(r"Table\s+\d+\b", re.IGNORECASE)


def _filter_telemetry(text: str) -> str:
    """Drop timestamped sensor-data rows — they flood BM25 with numeric noise."""
    lines = [line for line in text.splitlines() if not _TELEMETRY_ROW.match(line)]
    return "\n".join(lines)


def _extract_csv(data: bytes) -> list[tuple[int, str]]:
    import csv
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        return [(1, "")]
    md = []
    for i, row in enumerate(rows):
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in row]
        md.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return [(1, "\n".join(md))]


def _extract_text(data: bytes) -> list[tuple[int, str]]:
    text = data.decode("utf-8", errors="replace")
    return [(1, _filter_telemetry(text))]


def _extract_doc_title(pages: list[tuple[int, str]], filename: str) -> str:
    """Pull the document title from a 'Title : ...' header line, fall back to filename.

    Scans the first few pages, not just page 1 — cover/contact-info front matter
    (company address blocks, revision tables) commonly pushes the actual title
    page to page 2 or 3 in real-world manuals.
    """
    for _, page_text in pages[:3]:
        for line in page_text.splitlines()[:40]:
            m = re.match(r"^\s*Title\s*:\s*(.+)", line, re.IGNORECASE)
            if m and m.group(1).strip():
                return m.group(1).strip()[:120]
    return filename


def extract_pages(data: bytes, filename: str) -> list[tuple[int, str]]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    if ext == ".pptx":
        return _extract_pptx(data)
    if ext == ".csv":
        return _extract_csv(data)
    return _extract_text(data)


# ── image extraction ──────────────────────────────────────────────────────────

def _extract_images_pdf(data: bytes) -> list[tuple[int, int, bytes]]:
    """Extract images from a PDF. Returns (page_num, img_idx, img_bytes).

    Deduplicates by xref so repeated images (e.g. company logos) are only
    returned once, attributed to the first page they appear on.
    """
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    results: list[tuple[int, int, bytes]] = []
    seen_xrefs: set[int] = set()
    for page_num, page in enumerate(doc, start=1):
        img_idx = 0
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            base_image = doc.extract_image(xref)
            if base_image:
                img_bytes = base_image["image"]
                if config.MIN_IMAGE_BYTES <= len(img_bytes) <= config.MAX_IMAGE_BYTES:
                    results.append((page_num, img_idx, img_bytes))
                    img_idx += 1
    return results


def _extract_images_pptx(data: bytes) -> list[tuple[int, int, bytes]]:
    """Extract images from PPTX slides. Returns (slide_num, img_idx, img_bytes)."""
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    results: list[tuple[int, int, bytes]] = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        img_idx = 0
        for shape in slide.shapes:
            try:
                img_bytes = shape.image.blob
            except AttributeError:
                continue
            if config.MIN_IMAGE_BYTES <= len(img_bytes) <= config.MAX_IMAGE_BYTES:
                results.append((slide_num, img_idx, img_bytes))
                img_idx += 1
    return results


def _extract_images_docx(data: bytes) -> list[tuple[int, int, bytes]]:
    """Extract images from a DOCX file. Returns (page_num=1, img_idx, img_bytes).

    DOCX has no native page boundaries, so all images are attributed to page 1.
    Deduplicates by relationship id so the same image embedded multiple times is
    only returned once.
    """
    from docx import Document
    doc = Document(io.BytesIO(data))
    results: list[tuple[int, int, bytes]] = []
    seen: set[str] = set()
    img_idx = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        rid = rel.reltype + "|" + rel.target_ref
        if rid in seen:
            continue
        seen.add(rid)
        try:
            img_bytes = rel.target_part.blob
        except Exception:
            continue
        if config.MIN_IMAGE_BYTES <= len(img_bytes) <= config.MAX_IMAGE_BYTES:
            results.append((1, img_idx, img_bytes))
            img_idx += 1
    return results


def extract_images(data: bytes, filename: str) -> list[tuple[int, int, bytes]]:
    """Return (page_num, img_idx, img_bytes) for each qualifying image in the file."""
    if not config.OCR_ENABLED:
        return []
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_images_pdf(data)
    if ext == ".pptx":
        return _extract_images_pptx(data)
    if ext == ".docx":
        return _extract_images_docx(data)
    return []


def _preprocess_image(img_bytes: bytes) -> bytes:
    """Resize to OCR_IMAGE_MAX_SIDE, re-encode as JPEG at OCR_IMAGE_JPEG_QUALITY, strip metadata."""
    import io

    from PIL import Image
    with Image.open(io.BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        if max(img.size) > config.OCR_IMAGE_MAX_SIDE:
            img.thumbnail((config.OCR_IMAGE_MAX_SIDE, config.OCR_IMAGE_MAX_SIDE), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=config.OCR_IMAGE_JPEG_QUALITY, optimize=True)
        return out.getvalue()


def _image_path(doc_id: str, page_num: int, img_idx: int) -> str:
    return os.path.join(config.IMAGES_DIR, doc_id, f"p{page_num}_i{img_idx}.jpg")


async def _analyze_images(
    image_refs: list[tuple[int, int, str]], filename: str, doc_id: str
) -> list[dict]:
    """Caption each already-persisted image. Returns one record per retained image.

    The images were already downscaled, re-encoded and written to disk by the
    parse worker. Two behaviours differ from the previous pipeline, both aimed at
    not throwing away information that is expensive to recover:

    * The JPEG is kept. Previously the bytes were discarded after OCR, so adding a
      vision model later would have required re-parsing every source document.
      Retrieval still runs purely over the text caption — joint image/text
      embedding models are strongly biased toward text and retrieve images poorly
      — but the image is now available to show the user or pass to a vision model
      at answer time.
    * Images whose OCR output is too sparse (diagrams, schematics, wiring) are no
      longer dropped. They are stored with `caption_status="pending"` so a later
      captioning pass can find them by filter without re-parsing anything.

    Image failures never abort an ingest.
    """
    if not image_refs:
        return []

    loop = asyncio.get_running_loop()
    # easyocr holds the GIL and is memory-hungry; bound it rather than letting the
    # default executor spawn ~32 threads each loading a PyTorch model.
    ocr_sem = asyncio.Semaphore(config.PREPARE_CONCURRENCY)

    async def _one(page_num: int, img_idx: int, path: str) -> dict | None:
        caption, status = "", "pending"
        if config.OCR_ENABLED:
            async with ocr_sem:
                ocr_text = await loop.run_in_executor(None, _ocr_image_file, path)
            if ocr_text:
                caption, status = ocr_text, "ocr"
        if not caption and not config.KEEP_UNCAPTIONED_IMAGES:
            return None
        return {
            "page": page_num,
            "img_idx": img_idx,
            "caption": caption,
            "caption_status": status,
            "image_path": path,
        }

    results = await asyncio.gather(*[_one(p, i, path) for p, i, path in image_refs])
    return [r for r in results if r is not None]


def _ocr_image_file(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return _ocr_image(f.read())
    except OSError:
        log.warning("could not read image for OCR: %s", path)
        return None


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


# Image chunks use negative chunk_index values starting here to avoid colliding
# with text chunks (≥ 0) and the summary chunk (−1 on page 1).
_IMAGE_CHUNK_IDX_BASE = -100

_ocr_reader = None
_ocr_available: bool | None = None  # None = not yet tried; False = permanently unavailable


def _get_ocr_reader():
    global _ocr_reader, _ocr_available
    if _ocr_available is False:
        return None
    if _ocr_reader is not None:
        return _ocr_reader
    try:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ocr_available = True
        log.info("OCR reader loaded (easyocr)")
    except ImportError:
        _ocr_available = False
        log.warning(
            "easyocr not installed — OCR disabled. "
            "Install with: pip install -r requirements-ocr.txt"
        )
    except Exception as exc:
        _ocr_available = False
        log.warning("OCR reader unavailable (%s) — OCR disabled", exc)
    return _ocr_reader


def _ocr_image(img_bytes: bytes) -> str | None:
    """Synchronous OCR via easyocr. Must be called via run_in_executor — blocks the thread."""
    reader = _get_ocr_reader()
    if reader is None:
        return None
    try:
        lines = reader.readtext(img_bytes, detail=0, paragraph=True)
        text = " ".join(lines).strip()
        return text if len(text.split()) >= config.OCR_MIN_WORDS else None
    except Exception:
        log.warning("OCR failed for image", exc_info=True)
        return None


# ── table detection ───────────────────────────────────────────────────────────

def _find_table_spans(text: str) -> list[tuple[int, int]]:
    """Return (start_char, end_char) for each table region in text.

    A region starts at any 'Table N' label and ends at the next such label,
    or at the first occurrence of a sentence-starting paragraph (capital letter
    after a blank line) that follows at least 80 chars of table content.
    """
    matches = list(_TABLE_LABEL_RE.finditer(text))
    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        next_table_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Find a paragraph break that looks like prose rather than table data
        para = re.search(r"\n\n[A-Z][a-z]", text[m.end():next_table_start])
        if para and (m.end() + para.start()) > start + 80:
            end = m.end() + para.start()
        else:
            end = next_table_start
        spans.append((start, end))
    return spans


def _tag_table_chunks(doc_id: str, pages: list[tuple[int, str]], chunks: list[dict]) -> None:
    """Add table_id to chunks that overlap a detected table region.

    The table_id is globally unique within the doc: f"{doc_id}_t{n}".
    Chunks without a table region get no table_id key.
    """
    page_texts = {pnum: text for pnum, text in pages}
    table_counter = 0

    pages_with_tables = sorted(
        {pnum for pnum, text in pages if _TABLE_LABEL_RE.search(text)}
    )
    for page_num in pages_with_tables:
        page_text = page_texts[page_num]
        spans = _find_table_spans(page_text)
        if not spans:
            continue
        table_ids = [f"{doc_id}_t{table_counter + i}" for i in range(len(spans))]
        table_counter += len(spans)

        for chunk in chunks:
            if chunk["page"] != page_num or chunk.get("chunk_index", 0) < 0:
                continue
            raw = chunk["text"]
            # Locate chunk in page text via its first non-whitespace 50 chars
            key = raw.lstrip()[:50]
            pos = page_text.find(key)
            if pos == -1:
                key = raw.lstrip()[:25]
                pos = page_text.find(key)
            if pos == -1:
                continue
            chunk_end = pos + len(raw)
            for (tstart, tend), tid in zip(spans, table_ids):
                if pos < tend and chunk_end > tstart:
                    chunk["table_id"] = tid
                    break


# ── chunking ──────────────────────────────────────────────────────────────────

def _split(text: str, seps: list[str], max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    if not seps:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    chunks, current = [], ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.extend(_split(current, rest, max_chars))
            current = part
    if current:
        chunks.extend(_split(current, rest, max_chars))
    return chunks


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    seps = ["\n\n\n", "\n\n", "\n", ". ", " "]
    chunks = []
    for page_num, text in pages:
        raw = _split(text, seps, config.CHUNK_SIZE)
        for i, chunk in enumerate(raw):
            # prepend overlap from previous chunk
            if i > 0:
                prev = raw[i - 1]
                tail = prev[-config.CHUNK_OVERLAP :]
                chunk = tail + " " + chunk
            if chunk.strip():
                chunks.append({"page": page_num, "chunk_index": i, "text": chunk})
    return chunks


# ── ingest pipeline ───────────────────────────────────────────────────────────

_KV_ROW = re.compile(r"^\s{2}(\w[\w ()/-]{2,40}?)\s{3,}([\d.]+)\s+(.+?)\s*$")


def _summary_chunk(title: str, full_text: str) -> str | None:
    """Build a dense natural-language summary from spec-table key-value rows.

    Converts tabular lines like:
        System Mass                            59.473       kg
    into:
        Test Report — FLUX-9 Thruster | System Mass: 59.473 kg | ...
    which embeds far better for factual queries than raw table text.
    """
    pairs = []
    for line in full_text.splitlines():
        m = _KV_ROW.match(line)
        if m:
            key, val, unit = m.group(1).strip(), m.group(2), m.group(3)
            pairs.append(f"{key}: {val} {unit}")
    if not pairs:
        return None
    return f"{title} | " + " | ".join(pairs)


async def _analyze_and_store_images(
    doc_id: str, title: str, filename: str, folder: str, raw_images: list,
) -> None:
    """Background task: caption images then append image chunks to the store.

    Fires after text chunks are already committed, so ingest is never blocked
    waiting on OCR.
    """
    records = await _analyze_images(raw_images, filename, doc_id)
    if not records:
        return

    specs, texts, bm_ids, bm_texts = [], [], [], []
    for rec in records:
        page_num = rec["page"]
        chunk_index = _IMAGE_CHUNK_IDX_BASE - rec["img_idx"]
        cid = f"{doc_id}_{page_num}_{chunk_index}"
        caption = rec["caption"] or "(uncaptioned figure — pending vision pass)"
        text = f"[{title}] [Figure on page {page_num}]: {caption}"
        specs.append((cid, text, {
            "doc_id": doc_id,
            "filename": filename,
            "page": page_num,
            "chunk_index": chunk_index,
            "folder": folder,
            "chunk_type": "image",
            "caption_status": rec["caption_status"],
            "image_path": rec["image_path"],
        }))
        texts.append(text)
        bm_ids.append(cid)
        bm_texts.append(f"[{filename}] {text}")

    embeddings = await llm.embed(texts)
    documents = [
        store.to_document(cid, text, meta, emb)
        for (cid, text, meta), emb in zip(specs, embeddings)
    ]

    async with _lock:
        # Skip if the document was deleted while we were captioning.
        if not await store.get_by_filter(store.eq("doc_id", doc_id)):
            return
        await store.add(documents)
        bm25_index.add(bm_ids, bm_texts, {cid: folder for cid in bm_ids})
        bm25_index.commit()
        _invalidate_cache()
    log.info("stored %d image chunk(s) for %s", len(documents), filename)


_LLM_SUMMARY_SYSTEM = (
    "You are a technical document indexer. Write a 2-3 sentence summary covering: "
    "(1) what this document is and its purpose, (2) the main system or subject, "
    "(3) any key identifiers such as document number, revision, date, or critical specs. "
    "Include specific values and numbers where present. "
    "Output only the summary — no preamble, no labels, no trailing remarks."
)


async def _llm_summary(title: str, front_matter_text: str) -> str | None:
    """Generate a natural-language summary of a document's front matter.

    Called only when KV-pattern extraction produces nothing, so non-spec documents
    (manuals, reports, procedures) still get a summary chunk for cross-doc queries.

    Takes the first few pages, not just page 1 — the actual product/system name
    is frequently on page 2 or 3 (after a cover/contact-info page), and a summary
    that misses it can't be matched against a question naming that product.
    """
    if not front_matter_text.strip():
        return None
    try:
        result = await llm.chat(
            system=_LLM_SUMMARY_SYSTEM,
            user=f"Document: {title}\n\nFirst pages:\n{front_matter_text[:3000]}",
        )
        return result.strip() or None
    except Exception:
        log.warning("LLM summary generation failed for %s", title, exc_info=True)
        return None


_CATALOG_FIELDS_SYSTEM = (
    "You are a technical document indexer. From the front matter of an engineering "
    "document, extract the document/reference number and the latest revision or "
    "issue number.\n"
    "Reply with exactly two lines and nothing else:\n"
    "DOC: <document number, or — if absent>\n"
    "REV: <revision or issue number, or — if absent>\n"
    "Copy values verbatim. Never invent a number that is not in the text."
)
_DOC_LINE = re.compile(r"^\s*DOC\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_REV_LINE = re.compile(r"^\s*REV\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


async def _extract_catalog_fields(filename: str, front_matter: str) -> tuple[str, str]:
    """Resolve document number and revision once, at ingest time.

    These used to be recomputed per query by ~280 lines of regex tuned to one
    customer's "Is. 3 - Rev. 2" / "E3R10" conventions, which meant every catalog
    query re-parsed every document and any unfamiliar convention silently produced
    an em dash. Extracting once and storing the result in metadata makes catalog
    queries a metadata read, and makes a wrong value fixable by re-ingesting one
    file rather than by adding another regex branch.
    """
    if not front_matter.strip():
        return "—", "—"
    try:
        raw = await llm.chat(
            system=_CATALOG_FIELDS_SYSTEM,
            user=f"Document: {filename}\n\nFront matter:\n{front_matter[:2000]}",
        )
    except Exception:  # a missing catalog field must not fail ingest
        log.warning("catalog field extraction failed for %s", filename, exc_info=True)
        return "—", "—"

    def _pick(pattern: re.Pattern) -> str:
        m = pattern.search(raw)
        val = (m.group(1).strip(" \t\"'") if m else "")
        return val if val and val not in {"-", "--"} else "—"

    return _pick(_DOC_LINE), _pick(_REV_LINE)


def _parse_worker(original_path: str, filename: str, doc_id: str) -> dict:
    """Parse + chunk + extract images. Runs in a worker process.

    Everything here is CPU-bound and GIL-holding, which is exactly why it must not
    run on the event loop. Images are written to disk inside the worker so large
    byte buffers are never pickled back across the process boundary.
    """
    with open(original_path, "rb") as f:
        data = f.read()

    pages = extract_pages(data, filename)
    chunks = chunk_pages(pages)
    _tag_table_chunks(doc_id, pages, chunks)

    image_refs: list[tuple[int, int, str]] = []
    for page_num, img_idx, img_bytes in extract_images(data, filename):
        try:
            processed = _preprocess_image(img_bytes)
        except Exception:
            processed = img_bytes
        path = _image_path(doc_id, page_num, img_idx)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _write_bytes(path, processed)
            image_refs.append((page_num, img_idx, path))
        except OSError:
            continue

    return {
        "pages": pages,
        "chunks": chunks,
        "title": _extract_doc_title(pages, filename),
        "image_refs": image_refs,
    }


async def _prepare(data: bytes, filename: str, folder: str = "General") -> dict:
    """Parse, chunk, and embed one file. The slow part — safe to run concurrently."""
    doc_id = hashlib.sha256(data).hexdigest()[:16]
    loop = asyncio.get_running_loop()

    # Keep the original file so citations can link back to it. doc_id is a content
    # hash, so this is naturally idempotent on re-ingest — no lock needed.
    os.makedirs(config.ORIGINALS_DIR, exist_ok=True)
    ext = Path(filename).suffix.lower()
    original_path = os.path.join(config.ORIGINALS_DIR, f"{doc_id}{ext}")
    # Offloaded: a blocking write of a 100 MB upload stalls every other request.
    await loop.run_in_executor(None, _write_bytes, original_path, data)
    del data  # release the upload buffer before parsing allocates its own

    parsed = await loop.run_in_executor(_pool(), _parse_worker, original_path, filename, doc_id)
    pages, chunks, title = parsed["pages"], parsed["chunks"], parsed["title"]
    if not chunks:
        return {"doc_id": doc_id, "filename": filename, "status": "empty"}

    front_matter = "\n".join(text for _, text in pages[:3])
    full_text = "\n".join(text for _, text in pages)

    # Synthetic summary chunk — KV-pattern extraction for spec tables; LLM fallback
    # for manuals, reports, and procedures where no KV rows are present.
    summary = _summary_chunk(title, full_text)
    if not summary:
        llm_sum = await _llm_summary(title, front_matter)
        if llm_sum:
            summary = f"{title} — {llm_sum}"

    doc_number, revision = await _extract_catalog_fields(filename, front_matter)

    all_chunks = ([{"page": 1, "chunk_index": -1, "text": summary}] if summary else []) + chunks
    texts = ([summary] if summary else []) + [f"[{title}] {c['text']}" for c in chunks]
    embeddings = await llm.embed(texts)

    # Image captioning is fire-and-forget so text chunks commit immediately.
    if parsed["image_refs"]:
        task = asyncio.create_task(
            _analyze_and_store_images(doc_id, title, filename, folder, parsed["image_refs"])
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "folder": folder,
        "title": title,
        "doc_number": doc_number,
        "revision": revision,
        "status": "ok",
        "_chunks": all_chunks,
        "_texts": texts,
        "_embeddings": embeddings,
    }


async def _write(payload: dict) -> dict:
    """Write one prepared payload to the store. Caller must hold _lock."""
    doc_id = payload["doc_id"]
    filename = payload["filename"]
    if payload["status"] == "empty":
        return {"doc_id": doc_id, "filename": filename, "chunks": 0, "status": "empty"}

    chunks = payload["_chunks"]
    texts = payload["_texts"]
    embeddings = payload["_embeddings"]
    folder = payload.get("folder", "General")

    # Replace text chunks, preserving image chunks written by the background pass.
    await store.delete_doc_text_chunks(doc_id)

    documents, bm_ids, bm_texts = [], [], []
    for chunk, text, emb in zip(chunks, texts, embeddings):
        cid = f"{doc_id}_{chunk['page']}_{chunk['chunk_index']}"
        meta = {
            "doc_id": doc_id,
            "filename": filename,
            "page": chunk["page"],
            # chunk_index was previously omitted from stored metadata even though
            # several call sites read it back and silently defaulted to 0, which
            # made "exclude summary chunks" filters no-ops. It is stored now.
            "chunk_index": chunk["chunk_index"],
            "folder": folder,
            "doc_number": payload.get("doc_number", "—"),
            "revision": payload.get("revision", "—"),
        }
        if chunk.get("table_id"):
            meta["table_id"] = chunk["table_id"]
        if chunk.get("chunk_type"):
            meta["chunk_type"] = chunk["chunk_type"]
        documents.append(store.to_document(cid, text, meta, emb))
        bm_ids.append(cid)
        bm_texts.append(f"[{filename}] {text}")

    await store.add(documents)
    bm25_index.remove_doc(doc_id)
    bm25_index.add(bm_ids, bm_texts, {cid: folder for cid in bm_ids})
    _invalidate_cache()
    return {
        "doc_id": doc_id, "filename": filename, "folder": folder,
        "doc_number": payload.get("doc_number", "—"),
        "revision": payload.get("revision", "—"),
        "chunks": len(documents), "status": "ok",
    }


async def ingest(data: bytes, filename: str, folder: str = "General") -> dict:
    payload = await _prepare(data, filename, folder=folder)
    async with _lock:
        result = await _write(payload)
        bm25_index.commit()
    return result


# Files written to the store per flush. Bounds how many prepared payloads
# (chunks + texts + 768-float embeddings) are held in RAM at once — the previous
# implementation gathered every payload for the whole batch before writing, which
# on a 500-file drop meant gigabytes resident on a 16 GB machine.
_WRITE_FLUSH_EVERY = 25


async def _run_batch(
    files: list[tuple[bytes, str]],
    folder: str,
    on_prepared=None,
) -> tuple[list[dict], list[str]]:
    """Prepare files concurrently and flush to the store in batches.

    Returns (results, errors). A file that fails to parse is recorded as an error
    and the rest of the batch still completes — previously a single corrupt PDF
    among hundreds raised out of `asyncio.gather` and discarded every prepared
    payload in the job.
    """
    sem = asyncio.Semaphore(config.PREPARE_CONCURRENCY)
    results: list[dict] = []
    errors: list[str] = []
    pending: list[dict] = []

    async def _bounded(data: bytes, fn: str) -> dict:
        async with sem:
            try:
                payload = await _prepare(data, fn, folder=folder)
            except Exception as exc:  # per-file isolation is the point
                log.warning("ingest failed for %s", fn, exc_info=True)
                payload = {"filename": fn, "status": "error", "error": str(exc)}
            if on_prepared:
                on_prepared()
            return payload

    async def _flush() -> None:
        if not pending:
            return
        async with _lock:
            for payload in pending:
                results.append(await _write(payload))
            bm25_index.commit()
        pending.clear()

    tasks = [asyncio.create_task(_bounded(data, fn)) for data, fn in files]
    for coro in asyncio.as_completed(tasks):
        payload = await coro
        if payload.get("status") == "error":
            errors.append(f"{payload['filename']}: {payload['error']}")
            continue
        pending.append(payload)
        if len(pending) >= _WRITE_FLUSH_EVERY:
            await _flush()
    await _flush()
    return results, errors


async def ingest_many(files: list[tuple[bytes, str]], folder: str = "General") -> list[dict]:
    """Ingest multiple files with concurrent parse+embed and batched writes."""
    results, _errors = await _run_batch(files, folder)
    return results


async def ingest_many_tracked(files: list[tuple[bytes, str]], job, folder: str = "General") -> None:
    """Background variant of ingest_many. Updates job.prepared as each file
    clears the embed phase, then flips job.status to done at the end.

    job.prepared is safe to increment without a lock because asyncio is
    single-threaded — only one coroutine runs at a time, and increments happen
    between awaits, so there are no races.

    The job only fails outright on an error that is not attributable to a single
    file; per-file failures are collected into job.errors and the rest proceed.
    """
    def _tick() -> None:
        job.prepared += 1

    try:
        results, errors = await _run_batch(files, folder, on_prepared=_tick)
        job.results = results
        job.errors.extend(errors)
        job.status = "done"
    except Exception as exc:
        log.exception("bulk ingest job %s failed", job.id)
        job.status = "failed"
        job.errors.append(str(exc))


async def list_documents() -> list[dict]:
    """Document-level listing, cached until the next write.

    Backed by a metadata aggregation in Qdrant rather than by pulling every
    chunk's metadata into Python, which is what the ChromaDB version did on every
    cache miss — a full scan of every chunk payload to produce a list of a few
    dozen rows.
    """
    global _doc_cache
    if _doc_cache is not None:
        return _doc_cache
    _doc_cache = await store.doc_summaries()
    return _doc_cache


async def list_folders() -> list[str]:
    return await store.folders()


async def move_document(doc_id: str, new_folder: str) -> int:
    chunk_ids = list((await store.get_by_filter(store.eq("doc_id", doc_id))).keys())
    if not chunk_ids:
        return 0
    moved = await store.move_doc(doc_id, new_folder)
    _invalidate_cache()
    bm25_index.update_folders({cid: new_folder for cid in chunk_ids})
    return moved


def _remove_files(doc_id: str) -> None:
    for path in glob.glob(os.path.join(config.ORIGINALS_DIR, f"{doc_id}.*")):
        os.remove(path)
    img_dir = os.path.join(config.IMAGES_DIR, doc_id)
    if os.path.isdir(img_dir):
        for name in os.listdir(img_dir):
            os.remove(os.path.join(img_dir, name))
        os.rmdir(img_dir)


async def delete_document(doc_id: str) -> int:
    removed = await store.delete_doc(doc_id)
    if removed:
        _invalidate_cache()
        # Targeted removal — no full corpus re-tokenisation for one deleted file.
        bm25_index.remove_doc(doc_id)
        bm25_index.commit()
        _remove_files(doc_id)
    return removed


async def clear_all_documents() -> int:
    count = await store.clear()
    _invalidate_cache()
    bm25_index.clear()
    for path in glob.glob(os.path.join(config.ORIGINALS_DIR, "*")):
        os.remove(path)
    if os.path.isdir(config.IMAGES_DIR):
        for doc_dir in os.listdir(config.IMAGES_DIR):
            full = os.path.join(config.IMAGES_DIR, doc_dir)
            if os.path.isdir(full):
                for name in os.listdir(full):
                    os.remove(os.path.join(full, name))
                os.rmdir(full)
    return count
