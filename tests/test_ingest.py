"""Tests for document parsing and chunking."""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import bm25_index
import config
import ingest
import llm


# ── text extraction ───────────────────────────────────────────────────────────

def _txt(text: str) -> bytes:
    return text.encode()


def test_extract_plain_text():
    pages = ingest.extract_pages(_txt("Hello world"), "doc.txt")
    assert len(pages) == 1
    page_num, text = pages[0]
    assert page_num == 1
    assert "Hello world" in text


def test_extract_csv():
    csv = "name,grade,credits\nMATH101,A,3\nPHYS201,B+,4"
    pages = ingest.extract_pages(csv.encode(), "transcript.csv")
    text = pages[0][1]
    assert "MATH101" in text
    assert "A" in text


def test_extract_csv_with_date_column_keeps_data():
    # Rows with dates in a data column must not be stripped by the telemetry filter.
    csv = "date,sensor,value\n2024-01-15 08:00:00,pressure,340\n2024-01-15 08:01:00,pressure,341"
    pages = ingest.extract_pages(csv.encode(), "readings.csv")
    text = pages[0][1]
    assert "340" in text
    assert "341" in text
    assert "pressure" in text


def test_extract_csv_renders_as_markdown_table():
    csv = "part,mass_kg\nThruster,12.5\nFuel Tank,46.0"
    pages = ingest.extract_pages(csv.encode(), "bom.csv")
    text = pages[0][1]
    # Header separator row must be present
    assert "---" in text
    assert "Thruster" in text
    assert "46.0" in text


def test_extract_csv_empty_file():
    pages = ingest.extract_pages(b"", "empty.csv")
    assert pages == [(1, "")]


def test_extract_pptx_tables():
    """Tables on PPTX slides must be extracted; previously they were silently dropped."""
    from unittest.mock import MagicMock
    slide = MagicMock()
    shape_text = MagicMock()
    shape_text.has_text_frame = True
    shape_text.text_frame.text = "Slide title"
    shape_text.has_table = False
    shape_text.shapes = None  # not a group

    cell1, cell2 = MagicMock(), MagicMock()
    cell1.text = "Part"
    cell2.text = "Mass kg"
    cell3, cell4 = MagicMock(), MagicMock()
    cell3.text = "Thruster"
    cell4.text = "12.5"
    row1 = MagicMock()
    row1.cells = [cell1, cell2]
    row2 = MagicMock()
    row2.cells = [cell3, cell4]
    tbl = MagicMock()
    tbl.rows = [row1, row2]
    shape_table = MagicMock()
    shape_table.has_text_frame = False
    shape_table.has_table = True
    shape_table.table = tbl

    # Make hasattr(shape, "shapes") return False for non-group shapes
    del shape_text.shapes
    del shape_table.shapes

    slide.shapes = [shape_text, shape_table]
    prs = MagicMock()
    prs.slides = [slide]

    with MagicMock() as mock_pptx_module:
        mock_pptx_module.Presentation.return_value = prs
        import sys
        sys.modules["pptx"] = mock_pptx_module
        pages = ingest._extract_pptx(b"fake")
        del sys.modules["pptx"]

    text = pages[0][1]
    assert "Thruster" in text
    assert "12.5" in text
    assert "Part" in text


def test_extract_pptx_grouped_shapes():
    """Text inside grouped shapes must be extracted."""
    from unittest.mock import MagicMock
    child = MagicMock()
    child.has_text_frame = True
    child.text_frame.text = "Callout inside group"
    child.has_table = False
    del child.shapes  # not a group itself

    group = MagicMock()
    group.has_text_frame = False
    group.has_table = False
    group.shapes = [child]

    slide = MagicMock()
    slide.shapes = [group]
    prs = MagicMock()
    prs.slides = [slide]

    with MagicMock() as mock_pptx_module:
        mock_pptx_module.Presentation.return_value = prs
        import sys
        sys.modules["pptx"] = mock_pptx_module
        pages = ingest._extract_pptx(b"fake")
        del sys.modules["pptx"]

    assert pages
    assert "Callout inside group" in pages[0][1]


def test_extract_docx_heading_prepended_to_content():
    """Each content paragraph carries its section heading so chunks don't lose context."""
    from unittest.mock import MagicMock, patch

    def make_para(text, style_name):
        p = MagicMock()
        p.text = text
        p.style.name = style_name
        return p

    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        make_para("Revision History", "Heading 1"),
        make_para("Rev 3.1 — updated thrust specs", "Normal"),
        make_para("Rev 3.0 — initial release", "Normal"),
    ]
    mock_doc.tables = []

    with patch("docx.Document", return_value=mock_doc):
        pages = ingest._extract_docx(b"fake")

    text = pages[0][1]
    for part in text.split("\n\n"):
        if "Rev 3." in part:
            assert "Revision History" in part, f"Content chunk is missing its section heading: {part!r}"


def test_extract_docx_content_before_first_heading_preserved():
    """Paragraphs appearing before any heading are not dropped."""
    from unittest.mock import MagicMock, patch

    def make_para(text, style_name):
        p = MagicMock()
        p.text = text
        p.style.name = style_name
        return p

    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        make_para("Preamble text with no heading yet.", "Normal"),
        make_para("Section A", "Heading 1"),
        make_para("Section A content.", "Normal"),
    ]
    mock_doc.tables = []

    with patch("docx.Document", return_value=mock_doc):
        pages = ingest._extract_docx(b"fake")

    text = pages[0][1]
    assert "Preamble text" in text
    assert "Section A content" in text


def test_extract_docx_trailing_heading_preserved():
    """A heading with no content beneath it must still appear in the output."""
    from unittest.mock import MagicMock, patch

    def make_para(text, style_name):
        p = MagicMock()
        p.text = text
        p.style.name = style_name
        return p

    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        make_para("Section A", "Heading 1"),
        make_para("Content here.", "Normal"),
        make_para("Section B", "Heading 1"),  # no content follows
    ]
    mock_doc.tables = []

    with patch("docx.Document", return_value=mock_doc):
        pages = ingest._extract_docx(b"fake")

    assert "Section B" in pages[0][1]


def test_extract_unsupported_falls_back_to_text():
    pages = ingest.extract_pages(b"some content", "file.log")
    assert len(pages) == 1
    assert "some content" in pages[0][1]


# ── chunking ──────────────────────────────────────────────────────────────────

def test_chunk_short_text_is_single_chunk():
    pages = [(1, "This is a short document.")]
    chunks = ingest.chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "This is a short document."
    assert chunks[0]["page"] == 1


def test_chunk_long_text_splits():
    long = "word " * 500   # 2500 chars >> CHUNK_SIZE of 512
    pages = [(1, long)]
    chunks = ingest.chunk_pages(pages)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= ingest.config.CHUNK_SIZE + ingest.config.CHUNK_OVERLAP + 10


def test_chunk_preserves_page_number():
    pages = [(3, "content on page 3")]
    chunks = ingest.chunk_pages(pages)
    assert all(c["page"] == 3 for c in chunks)


def test_chunk_overlap_carries_context():
    # Two chunks: overlap from first should appear in second
    long = ("A" * ingest.config.CHUNK_SIZE) + " " + ("B" * ingest.config.CHUNK_SIZE)
    pages = [(1, long)]
    chunks = ingest.chunk_pages(pages)
    assert len(chunks) >= 2
    # Second chunk should contain tail of first (overlap)
    assert "A" in chunks[1]["text"]


def test_empty_text_produces_no_chunks():
    chunks = ingest.chunk_pages([(1, "   \n\n  ")])
    assert chunks == []


def test_multipage_chunks_have_correct_pages():
    pages = [(1, "page one content"), (2, "page two content"), (3, "page three content")]
    chunks = ingest.chunk_pages(pages)
    page_nums = {c["page"] for c in chunks}
    assert page_nums == {1, 2, 3}


# ── ingest pipeline ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_txt(mock_ollama):
    result = await ingest.ingest(b"The student GPA is 3.85.", "transcript.txt")
    assert result["status"] == "ok"
    assert result["chunks"] >= 1
    assert len(result["doc_id"]) == 16


@pytest.mark.asyncio
async def test_ingest_idempotent(mock_ollama):
    data = b"Repeated document content."
    r1 = await ingest.ingest(data, "doc.txt")
    r2 = await ingest.ingest(data, "doc.txt")
    assert r1["doc_id"] == r2["doc_id"]
    # Should not double-store; chunk count stays the same
    docs = ingest.list_documents()
    assert sum(d["doc_id"] == r1["doc_id"] for d in docs) == 1


@pytest.mark.asyncio
async def test_ingest_empty_file(mock_ollama):
    result = await ingest.ingest(b"   ", "empty.txt")
    assert result["status"] == "empty"


@pytest.mark.asyncio
async def test_list_documents(mock_ollama):
    await ingest.ingest(b"Document one.", "a.txt")
    await ingest.ingest(b"Document two.", "b.txt")
    docs = ingest.list_documents()
    names = {d["filename"] for d in docs}
    assert "a.txt" in names
    assert "b.txt" in names


@pytest.mark.asyncio
async def test_delete_document(mock_ollama):
    result = await ingest.ingest(b"To be deleted.", "del.txt")
    doc_id = result["doc_id"]
    removed = ingest.delete_document(doc_id)
    assert removed > 0
    docs = ingest.list_documents()
    assert not any(d["doc_id"] == doc_id for d in docs)


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_zero():
    removed = ingest.delete_document("deadbeef12345678")
    assert removed == 0


@pytest.mark.asyncio
async def test_clear_all_empty_returns_zero():
    assert ingest.clear_all_documents() == 0


@pytest.mark.asyncio
async def test_clear_all_removes_all_documents(mock_ollama):
    await ingest.ingest(b"Document Alpha content.", "alpha.txt")
    await ingest.ingest(b"Document Beta content.", "beta.txt")
    assert ingest._db().count() > 0

    removed = ingest.clear_all_documents()
    assert removed > 0
    assert ingest._db().count() == 0
    assert ingest.list_documents() == []


@pytest.mark.asyncio
async def test_clear_all_rebuilds_bm25_as_empty(mock_ollama):
    await ingest.ingest(b"Unique token XYZZY_CLEAR_TEST appears here.", "clear_test.txt")
    assert bm25_index.size() > 0

    ingest.clear_all_documents()
    assert bm25_index.size() == 0


# ── ingest_many ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_many_all_ok(mock_ollama):
    files = [
        (b"Alpha document content.", "alpha.txt"),
        (b"Beta document content.", "beta.txt"),
        (b"Gamma document content.", "gamma.txt"),
    ]
    results = await ingest.ingest_many(files)
    assert len(results) == 3
    assert all(r["status"] == "ok" for r in results)
    assert {r["filename"] for r in results} == {"alpha.txt", "beta.txt", "gamma.txt"}


@pytest.mark.asyncio
async def test_ingest_many_partial_empty(mock_ollama):
    files = [
        (b"Real content here.", "real.txt"),
        (b"   ", "empty.txt"),
    ]
    results = await ingest.ingest_many(files)
    by_name = {r["filename"]: r for r in results}
    assert by_name["real.txt"]["status"] == "ok"
    assert by_name["empty.txt"]["status"] == "empty"


@pytest.mark.asyncio
async def test_ingest_many_bm25_rebuilt_once(mock_ollama, monkeypatch):
    rebuild_calls = []
    original = bm25_index.rebuild
    monkeypatch.setattr(bm25_index, "rebuild", lambda col: rebuild_calls.append(1) or original(col))

    files = [(f"document {i} content".encode(), f"doc{i}.txt") for i in range(4)]
    await ingest.ingest_many(files)
    assert len(rebuild_calls) == 1


@pytest.mark.asyncio
async def test_ingest_many_all_stored(mock_ollama):
    files = [(b"Content alpha.", "alpha.txt"), (b"Content beta.", "beta.txt")]
    await ingest.ingest_many(files)
    docs = ingest.list_documents()
    names = {d["filename"] for d in docs}
    assert {"alpha.txt", "beta.txt"}.issubset(names)


# ── BM25 persistence ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bm25_saved_after_ingest(mock_ollama):
    import os
    await ingest.ingest(b"Turbine blade inspection protocol.", "turbine.txt")
    assert os.path.exists(config.BM25_PATH), "BM25 index was not persisted after ingest"


@pytest.mark.asyncio
async def test_bm25_load_restores_index(mock_ollama):
    await ingest.ingest(b"Hydraulic pressure nominal at 340 bar.", "hyd.txt")
    original_size = bm25_index._index.size

    # Simulate a server restart: wipe the in-memory index, then load from disk
    bm25_index._index.build([], [])
    assert bm25_index._index.size == 0

    loaded = bm25_index._index.load(config.BM25_PATH)
    assert loaded
    assert bm25_index._index.size == original_size


@pytest.mark.asyncio
async def test_bm25_load_falls_back_on_missing_file(mock_ollama):
    await ingest.ingest(b"Fuel cell efficiency report.", "fuel.txt")
    bm25_index._index.build([], [])

    # Point to a non-existent path — load should return False
    loaded = bm25_index._index.load("/tmp/does_not_exist_geo_assist.pkl")
    assert not loaded
    assert bm25_index._index.size == 0


@pytest.mark.asyncio
async def test_load_or_rebuild_uses_disk_when_available(mock_ollama, monkeypatch):
    await ingest.ingest(b"Satellite orbit decay analysis.", "orbit.txt")
    original_size = bm25_index._index.size

    # Wipe in-memory index, then call load_or_rebuild — should load from disk
    bm25_index._index.build([], [])
    rebuild_calls = []
    original_rebuild = bm25_index.rebuild
    monkeypatch.setattr(bm25_index, "rebuild", lambda col: rebuild_calls.append(1) or original_rebuild(col))

    bm25_index.load_or_rebuild(ingest._db())

    assert bm25_index._index.size == original_size
    assert len(rebuild_calls) == 0, "rebuild() was called even though disk index was available"


@pytest.mark.asyncio
async def test_load_or_rebuild_falls_back_to_rebuild(mock_ollama, monkeypatch):
    await ingest.ingest(b"Radar cross-section measurement.", "radar.txt")

    # Remove the persisted file so load fails
    monkeypatch.setattr(config, "BM25_PATH", "/tmp/nonexistent_geo_assist.pkl")
    bm25_index._index.build([], [])

    bm25_index.load_or_rebuild(ingest._db())

    assert bm25_index._index.size > 0, "load_or_rebuild did not fall back to ChromaDB rebuild"


# ── ingest_many_tracked ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tracked_job_reaches_done(mock_ollama):
    import jobs
    files = [(b"Thermal analysis report.", "thermal.txt"),
             (b"Stress distribution data.", "stress.txt")]
    job = jobs.create(total=len(files))
    await ingest.ingest_many_tracked(files, job)
    assert job.status == "done"
    assert job.prepared == 2
    assert len(job.results) == 2
    assert all(r["status"] == "ok" for r in job.results)


@pytest.mark.asyncio
async def test_tracked_job_increments_prepared(mock_ollama):
    import jobs
    files = [(f"document content {i}".encode(), f"doc{i}.txt") for i in range(5)]
    job = jobs.create(total=len(files))
    await ingest.ingest_many_tracked(files, job)
    assert job.prepared == 5


@pytest.mark.asyncio
async def test_tracked_job_fails_gracefully(mock_ollama, monkeypatch):
    import jobs
    monkeypatch.setattr(ingest, "_prepare", lambda data, fn, folder="General", access="public", owner="": (_ for _ in ()).throw(RuntimeError("embed failure")))
    job = jobs.create(total=1)
    await ingest.ingest_many_tracked([(b"data", "file.txt")], job)
    assert job.status == "failed"
    assert any("embed failure" in e for e in job.errors)


# ── vision / image extraction ─────────────────────────────────────────────────

def test_extract_images_skipped_when_no_vision_model(monkeypatch):
    """extract_images returns [] when neither VISION_MODEL nor OCR_ENABLED is configured."""
    monkeypatch.setattr(config, "VISION_MODEL", "")
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    result = ingest.extract_images(b"fake pdf", "diagram.pdf")
    assert result == []


def test_extract_images_skipped_for_unsupported_type(monkeypatch):
    """extract_images returns [] for file types that have no image extractor."""
    monkeypatch.setattr(config, "VISION_MODEL", "llava:7b")
    result = ingest.extract_images(b"fake txt", "report.txt")
    assert result == []


def test_extract_images_docx_returns_qualifying_images(monkeypatch):
    """_extract_images_docx returns images that meet the size threshold."""
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    good_bytes = b"x" * config.MIN_IMAGE_BYTES

    mock_rel = MagicMock()
    mock_rel.reltype = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    mock_rel.target_ref = "media/image1.png"
    mock_rel.target_part.blob = good_bytes

    mock_doc = MagicMock()
    mock_doc.part.rels = {"rId1": mock_rel}

    with patch("docx.Document", return_value=mock_doc):
        results = ingest._extract_images_docx(b"fake docx")

    assert len(results) == 1
    assert results[0] == (1, 0, good_bytes)


def test_extract_images_docx_skips_small_images(monkeypatch):
    """_extract_images_docx drops images below MIN_IMAGE_BYTES."""
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    tiny_bytes = b"x" * (config.MIN_IMAGE_BYTES - 1)

    mock_rel = MagicMock()
    mock_rel.reltype = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    mock_rel.target_ref = "media/icon.png"
    mock_rel.target_part.blob = tiny_bytes

    mock_doc = MagicMock()
    mock_doc.part.rels = {"rId1": mock_rel}

    with patch("docx.Document", return_value=mock_doc):
        results = ingest._extract_images_docx(b"fake docx")

    assert results == []


def test_extract_images_docx_deduplicates(monkeypatch):
    """_extract_images_docx returns each unique image only once."""
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    img_bytes = b"x" * config.MIN_IMAGE_BYTES

    def make_rel(ref):
        r = MagicMock()
        r.reltype = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        r.target_ref = ref
        r.target_part.blob = img_bytes
        return r

    mock_doc = MagicMock()
    mock_doc.part.rels = {
        "rId1": make_rel("media/image1.png"),
        "rId2": make_rel("media/image1.png"),  # same ref = duplicate
    }

    with patch("docx.Document", return_value=mock_doc):
        results = ingest._extract_images_docx(b"fake docx")

    assert len(results) == 1


def test_extract_images_routes_docx(monkeypatch):
    """extract_images routes .docx files to _extract_images_docx."""
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    called = []
    monkeypatch.setattr(ingest, "_extract_images_docx", lambda d: called.append(1) or [])
    ingest.extract_images(b"fake", "report.docx")
    assert called


def test_extract_images_pdf_size_filter(monkeypatch):
    """Images smaller than MIN_IMAGE_BYTES or larger than MAX_IMAGE_BYTES are dropped."""
    monkeypatch.setattr(config, "VISION_MODEL", "llava:7b")

    tiny = b"x" * (config.MIN_IMAGE_BYTES - 1)
    big  = b"x" * (config.MAX_IMAGE_BYTES + 1)
    good = b"x" * config.MIN_IMAGE_BYTES

    mock_page = MagicMock()
    mock_page.get_images.return_value = [(1, 0, 0, 0, 0, "", ""), (2, 0, 0, 0, 0, "", ""), (3, 0, 0, 0, 0, "", "")]

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.extract_image.side_effect = lambda xref: {"image": [tiny, big, good][xref - 1]}

    with patch("fitz.open", return_value=mock_doc):
        results = ingest._extract_images_pdf(b"fake")

    assert len(results) == 1
    assert results[0][2] == good


def test_extract_images_pdf_deduplicates_by_xref(monkeypatch):
    """The same xref appearing on multiple pages is only returned once."""
    monkeypatch.setattr(config, "VISION_MODEL", "llava:7b")
    img_bytes = b"x" * config.MIN_IMAGE_BYTES

    page1 = MagicMock()
    page1.get_images.return_value = [(42, 0, 0, 0, 0, "", "")]
    page2 = MagicMock()
    page2.get_images.return_value = [(42, 0, 0, 0, 0, "", "")]

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([page1, page2]))
    mock_doc.extract_image.return_value = {"image": img_bytes}

    with patch("fitz.open", return_value=mock_doc):
        results = ingest._extract_images_pdf(b"fake")

    assert len(results) == 1
    assert results[0][0] == 1  # attributed to page 1


@pytest.mark.asyncio
async def test_analyze_images_returns_descriptions(monkeypatch):
    """_analyze_images calls llm.analyze_image for each image and returns descriptions."""
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    img_bytes = b"x" * 100
    images = [(1, 0, img_bytes), (2, 0, img_bytes)]

    with patch.object(llm, "analyze_image", new=AsyncMock(return_value="A satellite dish.")):
        results = await ingest._analyze_images(images, "test.pdf")

    assert len(results) == 2
    assert all(desc == "A satellite dish." for _, _, desc in results)


@pytest.mark.asyncio
async def test_analyze_images_drops_empty_descriptions(monkeypatch):
    """Images where llm.analyze_image returns None/empty are silently dropped."""
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    monkeypatch.setattr(config, "OCR_ENABLED", False)
    img_bytes = b"x" * 100
    images = [(1, 0, img_bytes), (2, 0, img_bytes)]

    async def fake_analyze(img, filename, page):
        return "A diagram." if page == 1 else None

    with patch.object(llm, "analyze_image", new=AsyncMock(side_effect=fake_analyze)):
        results = await ingest._analyze_images(images, "test.pdf")

    assert len(results) == 1
    assert results[0][0] == 1


@pytest.mark.asyncio
async def test_ingest_pdf_with_images_stores_image_chunks(mock_ollama, monkeypatch):
    """When vision is enabled, image chunks are stored as background tasks after text ingest."""
    import asyncio
    monkeypatch.setattr(config, "VISION_MODEL", "llava:7b")

    fake_images = [(1, 0, b"x" * config.MIN_IMAGE_BYTES)]
    monkeypatch.setattr(ingest, "extract_images", lambda data, fn: fake_images)
    monkeypatch.setattr(ingest, "extract_pages", lambda data, fn: [(1, "Specification document text.")])

    with patch.object(llm, "analyze_image", new=AsyncMock(return_value="A rocket thruster.")):
        result = await ingest.ingest(b"fake pdf bytes", "spec.pdf")
        # Drain the event loop so the background image-analysis task completes
        pending = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert result["status"] == "ok"

    col = ingest._db()
    all_items = col.get(include=["documents", "metadatas"])
    image_chunks = [
        doc for doc in all_items["documents"]
        if doc and "[Figure on page" in doc
    ]
    assert image_chunks, "Expected at least one image chunk stored in ChromaDB"
    assert any("rocket thruster" in c for c in image_chunks)


# ── OCR two-tier pipeline ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_images_ocr_fast_path(monkeypatch):
    """OCR result is used and vision model is never called when OCR yields enough words."""
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(config, "OCR_MIN_WORDS", 3)
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")

    # Patch _ocr_image to return a text-rich result (simulates a screenshot)
    monkeypatch.setattr(ingest, "_ocr_image", lambda b: "Error code forty two occurred")

    analyze_mock = AsyncMock(return_value="This should not be called")
    with patch.object(llm, "analyze_image", new=analyze_mock):
        results = await ingest._analyze_images([(1, 0, b"x" * 100)], "test.pdf")

    assert len(results) == 1
    assert results[0][2] == "Error code forty two occurred"
    analyze_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_images_ocr_falls_back_to_vision(monkeypatch):
    """When OCR yields sparse text, the vision model is called as fallback."""
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(config, "OCR_MIN_WORDS", 10)
    monkeypatch.setattr(config, "VISION_MODEL", "moondream")

    # Patch _ocr_image to return None (simulates a diagram with little/no text)
    monkeypatch.setattr(ingest, "_ocr_image", lambda b: None)

    with patch.object(llm, "analyze_image", new=AsyncMock(return_value="Wiring diagram.")):
        results = await ingest._analyze_images([(1, 0, b"x" * 100)], "test.pdf")

    assert len(results) == 1
    assert results[0][2] == "Wiring diagram."


@pytest.mark.asyncio
async def test_analyze_images_ocr_only_no_vision_model(monkeypatch):
    """When OCR is enabled but VISION_MODEL is empty, only OCR runs (no Moondream call)."""
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(config, "OCR_MIN_WORDS", 3)
    monkeypatch.setattr(config, "VISION_MODEL", "")

    monkeypatch.setattr(ingest, "_ocr_image", lambda b: "OK status all systems nominal")

    analyze_mock = AsyncMock()
    with patch.object(llm, "analyze_image", new=analyze_mock):
        results = await ingest._analyze_images([(1, 0, b"x" * 100)], "test.pdf")

    assert len(results) == 1
    assert "nominal" in results[0][2]
    analyze_mock.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_images_ocr_sparse_no_vision_drops_image(monkeypatch):
    """When OCR yields nothing AND VISION_MODEL is empty, the image is dropped."""
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(config, "OCR_MIN_WORDS", 10)
    monkeypatch.setattr(config, "VISION_MODEL", "")

    monkeypatch.setattr(ingest, "_ocr_image", lambda b: None)

    results = await ingest._analyze_images([(1, 0, b"x" * 100)], "test.pdf")
    assert results == []


def test_extract_images_allowed_when_ocr_enabled_no_vision(monkeypatch):
    """extract_images proceeds when OCR_ENABLED is True even if VISION_MODEL is empty."""
    monkeypatch.setattr(config, "VISION_MODEL", "")
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    called = []
    monkeypatch.setattr(ingest, "_extract_images_pdf", lambda d: called.append(1) or [])
    ingest.extract_images(b"fake", "scan.pdf")
    assert called
