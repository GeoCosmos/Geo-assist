import asyncio
import hashlib
import io
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

import chromadb
from chromadb.config import Settings

import bm25_index
import config
import llm

_lock = asyncio.Lock()
_background_tasks: set = set()
_PREPARE_CONCURRENCY = 16  # max files parsed+embedded concurrently; bounds peak RAM on large batches
_chroma_client: chromadb.ClientAPI | None = None
_chroma: chromadb.Collection | None = None
_doc_cache: list[dict] | None = None


def _invalidate_cache() -> None:
    global _doc_cache
    _doc_cache = None


def _client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(config.CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _db() -> chromadb.Collection:
    global _chroma
    if _chroma is None:
        _chroma = _client().get_or_create_collection(
            "docs", metadata={"hnsw:space": "cosine"}
        )
    return _chroma


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


_AUDIO_VIDEO_EXTS = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mkv", ".mov", ".avi"}

_whisper_model = None  # lazy singleton — avoids reloading 39 MB on every audio file


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "Audio/video transcription requires faster-whisper and ffmpeg. "
                "Install with: pip install faster-whisper  "
                "(Windows: winget install ffmpeg)"
            )
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model


def _extract_audio(data: bytes, filename: str) -> list[tuple[int, str]]:
    """Transcribe audio/video using faster-whisper. Returns one page per detected segment group."""
    import tempfile
    import os

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        model = _get_whisper_model()
        segments, info = model.transcribe(tmp_path, beam_size=5)
        # Group segments into ~30-second pages to match chunking expectations
        pages: list[tuple[int, str]] = []
        page_lines: list[str] = []
        page_num = 1
        page_start = 0.0
        for seg in segments:
            page_lines.append(seg.text.strip())
            if seg.end - page_start >= 30.0:
                pages.append((page_num, " ".join(page_lines)))
                page_lines = []
                page_num += 1
                page_start = seg.end
        if page_lines:
            pages.append((page_num, " ".join(page_lines)))
        return pages if pages else [(1, "")]
    finally:
        os.unlink(tmp_path)


def _extract_doc_title(pages: list[tuple[int, str]], filename: str) -> str:
    """Pull the document title from a 'Title : ...' header line, fall back to filename."""
    if pages:
        _, first_page = pages[0]
        for line in first_page.splitlines()[:40]:
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
    if ext in _AUDIO_VIDEO_EXTS:
        return _extract_audio(data, filename)
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
    if not config.VISION_MODEL and not config.OCR_ENABLED:
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
    """Resize to VISION_MAX_SIDE, re-encode as JPEG at VISION_JPEG_QUALITY, strip metadata."""
    import io
    from PIL import Image
    with Image.open(io.BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        if max(img.size) > config.VISION_MAX_SIDE:
            img.thumbnail((config.VISION_MAX_SIDE, config.VISION_MAX_SIDE), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=config.VISION_JPEG_QUALITY, optimize=True)
        return out.getvalue()


async def _analyze_images(
    images: list[tuple[int, int, bytes]], filename: str
) -> list[tuple[int, int, str]]:
    """Analyze each image with the vision model. Returns (page_num, img_idx, description).

    Images that fail analysis are silently dropped so a bad image never aborts ingest.
    """
    if not images:
        return []

    async def _one(page_num: int, img_idx: int, img_bytes: bytes):
        try:
            processed = _preprocess_image(img_bytes)
        except Exception:
            log.warning("image preprocessing failed for %s page %d", filename, page_num, exc_info=True)
            processed = img_bytes

        if config.OCR_ENABLED:
            loop = asyncio.get_event_loop()
            ocr_text = await loop.run_in_executor(None, _ocr_image, processed)
            if ocr_text:
                return (page_num, img_idx, ocr_text)

        if config.VISION_MODEL:
            desc = await llm.analyze_image(processed, filename, page_num)
            return (page_num, img_idx, desc) if desc else None

        return None

    results = await asyncio.gather(*[_one(p, i, b) for p, i, b in images])
    return [r for r in results if r is not None]


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
    doc_id: str, title: str, filename: str,
    folder: str, access: str, owner: str,
    raw_images: list,
) -> None:
    """Background task: run vision analysis then append image chunks to ChromaDB.

    Fires after text chunks are already committed, so ingest is never blocked
    waiting for the vision model.
    """
    image_descriptions = await _analyze_images(raw_images, filename)
    if not image_descriptions:
        return
    image_chunks = [
        {
            "page": page_num,
            "chunk_index": _IMAGE_CHUNK_IDX_BASE - img_idx,
            "text": f"[Figure on page {page_num}]: {desc}",
            "chunk_type": "image",
        }
        for page_num, img_idx, desc in image_descriptions
    ]
    image_texts = [f"[{title}] {c['text']}" for c in image_chunks]
    image_embeddings = await llm.embed(image_texts)
    async with _lock:
        col = _db()
        # Skip if the document was deleted while we were analyzing images
        if not col.get(where={"doc_id": doc_id})["ids"]:
            return
        ids = [f"{doc_id}_{c['page']}_{c['chunk_index']}" for c in image_chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "page": c["page"],
                "folder": folder,
                "access": access,
                "owner": owner,
                "chunk_type": "image",
            }
            for c in image_chunks
        ]
        col.add(ids=ids, embeddings=image_embeddings, documents=image_texts, metadatas=metadatas)
        _invalidate_cache()
    log.info("stored %d image chunk(s) for %s", len(image_chunks), filename)


_LLM_SUMMARY_SYSTEM = (
    "You are a technical document indexer. Write a 2-3 sentence summary covering: "
    "(1) what this document is and its purpose, (2) the main system or subject, "
    "(3) any key identifiers such as document number, revision, date, or critical specs. "
    "Include specific values and numbers where present. "
    "Output only the summary — no preamble, no labels, no trailing remarks."
)


async def _llm_summary(title: str, first_page_text: str) -> str | None:
    """Generate a natural-language summary of a document's first page.

    Called only when KV-pattern extraction produces nothing, so non-spec documents
    (manuals, reports, procedures) still get a summary chunk for cross-doc queries.
    """
    if not first_page_text.strip():
        return None
    try:
        result = await llm.chat(
            system=_LLM_SUMMARY_SYSTEM,
            user=f"Document: {title}\n\nFirst page:\n{first_page_text[:2000]}",
        )
        return result.strip() or None
    except Exception:
        log.warning("LLM summary generation failed for %s", title, exc_info=True)
        return None


async def _prepare(data: bytes, filename: str, folder: str = "General",
                   access: str = "public", owner: str = "") -> dict:
    """Parse, chunk, and embed one file. The slow part — safe to run concurrently."""
    doc_id = hashlib.sha256(data).hexdigest()[:16]
    pages = extract_pages(data, filename)
    chunks = chunk_pages(pages)
    if not chunks:
        return {"doc_id": doc_id, "filename": filename, "status": "empty"}

    title = _extract_doc_title(pages, filename)
    full_text = "\n".join(text for _, text in pages)

    # Synthetic summary chunk — KV-pattern extraction for spec tables; LLM fallback
    # for manuals, reports, and procedures where no KV rows are present.
    summary = _summary_chunk(title, full_text)
    if not summary:
        first_page_text = pages[0][1] if pages else ""
        llm_sum = await _llm_summary(title, first_page_text)
        if llm_sum:
            summary = f"{title} — {llm_sum}"
    summary_chunk_list = [{"page": 1, "chunk_index": -1, "text": summary}] if summary else []

    _tag_table_chunks(doc_id, pages, chunks)

    text_embeddings = await llm.embed(
        ([summary] if summary else []) + [f"[{title}] {c['text']}" for c in chunks]
    )

    # Kick off image analysis as a fire-and-forget background task so text chunks
    # are committed immediately without waiting for the vision model.
    raw_images = extract_images(data, filename)
    if raw_images:
        task = asyncio.create_task(
            _analyze_and_store_images(doc_id, title, filename, folder, access, owner, raw_images)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    all_chunks = summary_chunk_list + chunks
    texts = ([summary] if summary else []) + [f"[{title}] {c['text']}" for c in chunks]

    return {
        "doc_id": doc_id,
        "filename": filename,
        "folder": folder,
        "access": access,
        "owner": owner,
        "status": "ok",
        "_chunks": all_chunks,
        "_texts": texts,
        "_embeddings": text_embeddings,
    }


def _write(col, payload: dict) -> dict:
    """Write one prepared payload into ChromaDB. Caller must hold _lock."""
    doc_id = payload["doc_id"]
    filename = payload["filename"]
    if payload["status"] == "empty":
        return {"doc_id": doc_id, "filename": filename, "chunks": 0, "status": "empty"}
    chunks = payload["_chunks"]
    texts = payload["_texts"]
    embeddings = payload["_embeddings"]
    existing = col.get(where={"doc_id": doc_id}, include=["metadatas"])
    if existing["ids"]:
        # Preserve image chunks added by vision_index.py — only replace text chunks
        non_image_ids = [
            eid for eid, emeta in zip(existing["ids"], existing["metadatas"])
            if emeta.get("chunk_type") != "image"
        ]
        if non_image_ids:
            col.delete(ids=non_image_ids)
    ids = [f"{doc_id}_{c['page']}_{c['chunk_index']}" for c in chunks]
    folder = payload.get("folder", "General")
    access = payload.get("access", "public")
    owner  = payload.get("owner", "")
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "page": c["page"],
            "folder": folder,
            "access": access,
            "owner": owner,
            **({"table_id": c["table_id"]} if c.get("table_id") else {}),
            **({"chunk_type": c["chunk_type"]} if c.get("chunk_type") else {}),
        }
        for c in chunks
    ]
    col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    _invalidate_cache()
    return {"doc_id": doc_id, "filename": filename, "folder": folder, "access": access, "owner": owner, "chunks": len(chunks), "status": "ok"}


async def ingest(data: bytes, filename: str, folder: str = "General",
                 access: str = "public", owner: str = "") -> dict:
    payload = await _prepare(data, filename, folder=folder, access=access, owner=owner)
    async with _lock:
        col = _db()
        result = _write(col, payload)
        bm25_index.rebuild(col)
    return result


async def ingest_many(files: list[tuple[bytes, str]], folder: str = "General",
                      access: str = "public", owner: str = "") -> list[dict]:
    """Ingest multiple files with concurrent parse+embed, then one write pass and one BM25 rebuild."""
    sem = asyncio.Semaphore(_PREPARE_CONCURRENCY)

    async def _bounded(data: bytes, fn: str) -> dict:
        async with sem:
            return await _prepare(data, fn, folder=folder, access=access, owner=owner)

    payloads = await asyncio.gather(*[_bounded(data, fn) for data, fn in files])
    async with _lock:
        col = _db()
        results = [_write(col, p) for p in payloads]
        bm25_index.rebuild(col)
    return results


async def ingest_many_tracked(files: list[tuple[bytes, str]], job,
                               folder: str = "General", access: str = "public", owner: str = "") -> None:
    """Background variant of ingest_many. Updates job.prepared as each file
    clears the embed phase, then flips job.status to done/failed at the end.

    job.prepared is safe to increment without a lock because asyncio is
    single-threaded — only one coroutine runs at a time, and increments happen
    between awaits, so there are no races.
    """
    sem = asyncio.Semaphore(_PREPARE_CONCURRENCY)

    async def _prepare_and_tick(data: bytes, fn: str) -> dict:
        async with sem:
            result = await _prepare(data, fn, folder=folder, access=access, owner=owner)
        job.prepared += 1
        return result

    try:
        payloads = await asyncio.gather(*[_prepare_and_tick(data, fn) for data, fn in files])
        async with _lock:
            col = _db()
            job.results = [_write(col, p) for p in payloads]
            bm25_index.rebuild(col)
        job.status = "done"
    except Exception as exc:
        job.status = "failed"
        job.errors.append(str(exc))


def list_documents() -> list[dict]:
    global _doc_cache
    if _doc_cache is not None:
        return _doc_cache
    col = _db()
    all_meta = col.get(include=["metadatas"])["metadatas"] or []
    seen: dict[str, dict] = {}
    for m in all_meta:
        doc_id = m["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": m["filename"],
                "folder": m.get("folder", "General"),
                "access": m.get("access", "public"),
                "owner": m.get("owner", ""),
                "chunks": 0,
            }
        seen[doc_id]["chunks"] += 1
    _doc_cache = list(seen.values())
    return _doc_cache


def list_folders() -> list[str]:
    return sorted({d.get("folder", "General") for d in list_documents()})


def move_document(doc_id: str, new_folder: str) -> int:
    col = _db()
    existing = col.get(where={"doc_id": doc_id}, include=["metadatas"])
    if not existing["ids"]:
        return 0
    new_metas = [{**m, "folder": new_folder} for m in existing["metadatas"]]
    col.update(ids=existing["ids"], metadatas=new_metas)
    _invalidate_cache()
    bm25_index.update_folders({cid: new_folder for cid in existing["ids"]})
    return len(existing["ids"])


def delete_document(doc_id: str) -> int:
    col = _db()
    existing = col.get(where={"doc_id": doc_id})
    if existing["ids"]:
        col.delete(ids=existing["ids"])
        _invalidate_cache()
        bm25_index.rebuild(col)
    return len(existing["ids"])


def clear_all_documents() -> int:
    global _chroma
    col = _db()
    count = col.count()
    if count == 0:
        return 0
    _client().delete_collection("docs")
    _chroma = _client().create_collection("docs", metadata={"hnsw:space": "cosine"})
    _invalidate_cache()
    bm25_index.rebuild(_chroma)
    return count
