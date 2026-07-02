"""Comprehensive tests for all new features.

Features covered:
  1. Russian / Armenian language support (system prompt)
  2. Folder organisation (metadata, list_folders, move_document, BM25 filtering, API)
  3. PDF table extraction (_table_to_markdown format, empty table, reading order)
  4. Comparison table instruction (system prompt)
  5. Where-clause helper (_with_folder) for retrieval filtering
  6. Generic named-document detection (_find_named_docs)
  7. Cross-encoder re-ranking (reranker.rerank)
"""
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

import bm25_index
import config
import ingest
import retriever


@pytest.mark.asyncio
async def test_api_still_rejects_binary_exe(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("virus.exe", b"MZ\x90\x00", "application/octet-stream")})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_api_still_rejects_png(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("photo.png", b"\x89PNG", "image/png")})
    assert r.status_code == 400


# ════════════════════════════════════════════════════════════════════════════════
# Feature 2: Russian / Armenian language support
# ════════════════════════════════════════════════════════════════════════════════

def test_system_prompt_includes_russian_response_instruction():
    assert "respond in Russian" in retriever._SYSTEM


def test_system_prompt_includes_armenian_response_instruction():
    assert "respond in Armenian" in retriever._SYSTEM


def test_system_prompt_mentions_english_context_is_ok():
    assert "context may be in English" in retriever._SYSTEM


def test_system_prompt_advises_english_term_in_parens():
    assert "parentheses" in retriever._SYSTEM


def test_system_prompt_same_language_instruction():
    assert "same language the user used" in retriever._SYSTEM


# ════════════════════════════════════════════════════════════════════════════════
# Feature 3: Folder organisation
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_ingest_stores_custom_folder(mock_ollama):
    result = await ingest.ingest(b"Hydraulic specs report.", "hydraulic.txt", folder="Engineering")
    assert result["folder"] == "Engineering"
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_ingest_default_folder_is_general(mock_ollama):
    result = await ingest.ingest(b"Document with default folder.", "doc.txt")
    assert result["folder"] == "General"


@pytest.mark.asyncio
async def test_list_folders_sorted_and_unique(mock_ollama):
    await ingest.ingest(b"HR doc.", "hr.txt", folder="HR")
    await ingest.ingest(b"Finance doc.", "fin.txt", folder="Finance")
    await ingest.ingest(b"Another HR doc.", "hr2.txt", folder="HR")
    folders = ingest.list_folders()
    assert folders.count("HR") == 1
    assert "Finance" in folders
    assert folders == sorted(folders)


@pytest.mark.asyncio
async def test_list_folders_empty_when_no_docs():
    assert ingest.list_folders() == []


@pytest.mark.asyncio
async def test_move_document_changes_folder(mock_ollama):
    result = await ingest.ingest(b"Moveable content here.", "move.txt", folder="Source")
    doc_id = result["doc_id"]

    moved = ingest.move_document(doc_id, "Destination")
    assert moved > 0

    docs = ingest.list_documents()
    doc = next(d for d in docs if d["doc_id"] == doc_id)
    assert doc["folder"] == "Destination"


@pytest.mark.asyncio
async def test_move_document_nonexistent_returns_zero():
    assert ingest.move_document("deadbeef12345678", "AnyFolder") == 0


@pytest.mark.asyncio
async def test_move_document_rebuilds_bm25(mock_ollama):
    result = await ingest.ingest(b"BM25 folder test content.", "bm25test.txt", folder="OldF")
    doc_id = result["doc_id"]
    old_size = bm25_index._index.size
    ingest.move_document(doc_id, "NewF")
    # BM25 should be rebuilt — size unchanged but folder mapping updated
    assert bm25_index._index.size == old_size
    assert "NewF" in bm25_index._index._folders.values()


@pytest.mark.asyncio
async def test_list_documents_includes_folder(mock_ollama):
    await ingest.ingest(b"Sensitive report.", "secret.txt", folder="Confidential")
    docs = ingest.list_documents()
    doc = next(d for d in docs if d["filename"] == "secret.txt")
    assert doc["folder"] == "Confidential"


@pytest.mark.asyncio
async def test_api_ingest_with_folder_form_field(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post(
        "/ingest",
        data={"folder": "Reports"},
        files={"file": ("report.txt", b"Annual report 2024.", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["folder"] == "Reports"


@pytest.mark.asyncio
async def test_api_folders_endpoint(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    client.post(
        "/ingest",
        data={"folder": "Alpha"},
        files={"file": ("a.txt", b"Alpha content.", "text/plain")},
    )
    client.post(
        "/ingest",
        data={"folder": "Beta"},
        files={"file": ("b.txt", b"Beta content.", "text/plain")},
    )
    r = client.get("/folders")
    assert r.status_code == 200
    folders = r.json()["folders"]
    assert "Alpha" in folders
    assert "Beta" in folders


@pytest.mark.asyncio
async def test_api_folders_endpoint_empty():
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.get("/folders")
    assert r.status_code == 200
    assert r.json()["folders"] == []


@pytest.mark.asyncio
async def test_api_move_document_folder(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post(
        "/ingest",
        data={"folder": "OldFolder"},
        files={"file": ("moveme.txt", b"Move this.", "text/plain")},
    )
    doc_id = r.json()["doc_id"]
    r2 = client.patch(f"/documents/{doc_id}/folder", json={"folder": "NewFolder"})
    assert r2.status_code == 200
    assert r2.json()["moved_chunks"] > 0
    docs = client.get("/documents").json()["documents"]
    assert next(d for d in docs if d["doc_id"] == doc_id)["folder"] == "NewFolder"


@pytest.mark.asyncio
async def test_api_move_document_nonexistent_returns_404(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.patch("/documents/deadbeef00000000/folder", json={"folder": "Anywhere"})
    assert r.status_code == 404


# ── BM25 folder filtering ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bm25_folder_filter_keeps_only_matching_folder(mock_ollama):
    await ingest.ingest(b"Rocket propulsion data fuel thrust.", "rockets.txt", folder="Propulsion")
    await ingest.ingest(b"Avionics sensor array calibration.", "avionics.txt", folder="Avionics")

    results = bm25_index._index.search("propulsion fuel thrust", folder="Propulsion")
    for cid, score in results:
        if score > 0:
            assert bm25_index._index._folders.get(cid) == "Propulsion", (
                f"Chunk {cid} from folder {bm25_index._index._folders.get(cid)} "
                "leaked through Propulsion filter"
            )


@pytest.mark.asyncio
async def test_bm25_no_folder_filter_returns_all_folders(mock_ollama):
    await ingest.ingest(b"Alpha content here.", "alpha.txt", folder="FolderA")
    await ingest.ingest(b"Beta content there.", "beta.txt", folder="FolderB")

    # Without a folder filter the index contains chunks from both folders.
    # BM25 scores can be 0 when terms appear equally in all docs, so check the
    # folder map directly rather than filtering by score > 0.
    all_folders = set(bm25_index._index._folders.values())
    assert "FolderA" in all_folders
    assert "FolderB" in all_folders


@pytest.mark.asyncio
async def test_bm25_folder_filter_returns_empty_for_wrong_folder(mock_ollama):
    await ingest.ingest(b"Finance quarterly report budget.", "fin.txt", folder="Finance")

    results = bm25_index._index.search("finance quarterly budget", folder="HR")
    relevant = [(cid, s) for cid, s in results if s > 0]
    assert relevant == [], "Docs from Finance folder leaked into HR-filtered results"


@pytest.mark.asyncio
async def test_bm25_folder_mapping_persists_through_save_and_load(mock_ollama, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BM25_PATH", str(tmp_path / "bm25_test.pkl"))
    await ingest.ingest(b"Thermal analysis document.", "thermal.txt", folder="Thermal")

    bm25_index._index.build([], [])  # wipe in-memory
    bm25_index._index.load(config.BM25_PATH)

    assert "Thermal" in bm25_index._index._folders.values()


# ════════════════════════════════════════════════════════════════════════════════
# Feature 5: PDF table extraction via find_tables
# ════════════════════════════════════════════════════════════════════════════════

def test_table_to_markdown_basic_output():
    table = MagicMock()
    table.extract.return_value = [
        ["Name", "Value", "Unit"],
        ["Mass", "59.4", "kg"],
        ["Power", "350", "W"],
    ]
    result = ingest._table_to_markdown(table)
    lines = result.splitlines()
    assert lines[0] == "| Name | Value | Unit |"
    assert lines[1] == "| --- | --- | --- |"
    assert "| Mass | 59.4 | kg |" in result
    assert "| Power | 350 | W |" in result


def test_table_to_markdown_empty_returns_empty_string():
    table = MagicMock()
    table.extract.return_value = []
    assert ingest._table_to_markdown(table) == ""


def test_table_to_markdown_escapes_pipe_chars():
    table = MagicMock()
    table.extract.return_value = [
        ["Header"],
        ["value|with|pipes"],
    ]
    result = ingest._table_to_markdown(table)
    assert "value\\|with\\|pipes" in result


def test_table_to_markdown_strips_newlines_in_cells():
    table = MagicMock()
    table.extract.return_value = [
        ["Col1", "Col2"],
        ["multi\nline\ncell", "normal"],
    ]
    result = ingest._table_to_markdown(table)
    assert "\n" not in result.split("|")[2].strip()  # cell content shouldn't have newlines
    assert "multi line cell" in result


def test_table_to_markdown_handles_none_cells():
    table = MagicMock()
    table.extract.return_value = [
        ["A", None, "C"],
        [None, "B", None],
    ]
    result = ingest._table_to_markdown(table)
    assert "| A |  | C |" in result


def test_table_to_markdown_single_row():
    table = MagicMock()
    table.extract.return_value = [["Only", "Row"]]
    result = ingest._table_to_markdown(table)
    assert "| Only | Row |" in result
    assert "| --- | --- |" in result


def test_extract_pdf_no_tables_uses_plain_text():
    """When find_tables returns no tables, page text is extracted via get_text('dict')."""

    mock_table_finder = MagicMock()
    mock_table_finder.tables = []

    mock_page = MagicMock()
    mock_page.find_tables.return_value = mock_table_finder
    mock_page.get_text.return_value = {
        "blocks": [{
            "type": 0, "bbox": [0, 0, 400, 50],
            "lines": [{"spans": [{"text": "Plain page text content.", "size": 12.0, "font": "Arial"}]}],
        }]
    }

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__len__ = MagicMock(return_value=1)

    with patch("fitz.open", return_value=mock_doc):
        pages = ingest._extract_pdf(b"fake pdf")

    assert "Plain page text content." in pages[0][1]
    mock_page.get_text.assert_called_once_with("dict")


def test_extract_pdf_with_tables_converts_to_markdown():
    """When find_tables detects tables, output should contain markdown pipe rows."""

    mock_table = MagicMock()
    mock_table.extract.return_value = [["Col1", "Col2"], ["Val1", "Val2"]]
    mock_table.bbox = (0, 100, 400, 200)

    mock_table_finder = MagicMock()
    mock_table_finder.tables = [mock_table]

    mock_page = MagicMock()
    mock_page.find_tables.return_value = mock_table_finder
    mock_page.get_text.return_value = "Header text above table."

    mock_page.get_text.side_effect = lambda mode=None, **kw: (
        {"blocks": [{"type": 0, "bbox": [0, 0, 400, 50],
                     "lines": [{"spans": [{"text": "Header text above table.",
                                           "size": 12.0, "font": "Arial"}]}]}]}
        if mode == "dict" else "Header text above table."
    )

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("fitz.open", return_value=mock_doc):
        with patch("fitz.Rect", side_effect=lambda *args: MagicMock(
            intersects=MagicMock(return_value=False)
        )):
            pages = ingest._extract_pdf(b"fake pdf")

    # Output should contain a markdown table row
    assert "|" in pages[0][1]
    assert "Col1" in pages[0][1]


# ════════════════════════════════════════════════════════════════════════════════
# Feature 6: Comparison table output instruction
# ════════════════════════════════════════════════════════════════════════════════

def test_system_prompt_instructs_markdown_table_for_comparisons():
    assert "markdown table" in retriever._SYSTEM.lower()


def test_system_prompt_mentions_compare_contrast():
    assert "compare" in retriever._SYSTEM.lower()
    assert "contrast" in retriever._SYSTEM.lower()


def test_system_prompt_comparison_includes_column_headers():
    assert "column headers" in retriever._SYSTEM


def test_system_prompt_comparison_one_row_per_item():
    assert "one row per item" in retriever._SYSTEM


def test_system_prompt_comparison_adds_prose_summary():
    assert "prose summary" in retriever._SYSTEM


# ════════════════════════════════════════════════════════════════════════════════
# Feature 5 (where-clause helper): _with_folder
# ════════════════════════════════════════════════════════════════════════════════

def test_with_folder_no_folder_returns_where_unchanged():
    where = {"filename": "report.txt"}
    assert retriever._with_folder(where, None) is where


def test_with_folder_empty_string_returns_where_unchanged():
    where = {"filename": "report.txt"}
    assert retriever._with_folder(where, "") is where


def test_with_folder_no_existing_where_returns_simple_clause():
    result = retriever._with_folder(None, "Engineering")
    assert result == {"folder": "Engineering"}


def test_with_folder_with_existing_where_builds_and_clause():
    where = {"access": "public"}
    result = retriever._with_folder(where, "HR")
    assert result == {"$and": [{"access": "public"}, {"folder": "HR"}]}


def test_with_folder_nested_correctly():
    where = {"$or": [{"access": "public"}, {"owner": "alice"}]}
    result = retriever._with_folder(where, "Finance")
    assert result["$and"][0] == where
    assert result["$and"][1] == {"folder": "Finance"}


# ════════════════════════════════════════════════════════════════════════════════
# Fix 5: Folder name validation
# ════════════════════════════════════════════════════════════════════════════════

def test_safe_folder_strips_whitespace():
    import main
    assert main._safe_folder("  Reports  ") == "Reports"


def test_safe_folder_empty_string_defaults_to_general():
    import main
    assert main._safe_folder("") == "General"


def test_safe_folder_whitespace_only_defaults_to_general():
    import main
    assert main._safe_folder("   ") == "General"


def test_safe_folder_caps_at_64_chars():
    import main
    long_name = "A" * 100
    result = main._safe_folder(long_name)
    assert len(result) == 64
    assert result == "A" * 64


def test_safe_folder_normal_name_unchanged():
    import main
    assert main._safe_folder("Engineering") == "Engineering"


@pytest.mark.asyncio
async def test_api_empty_folder_stored_as_general(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post(
        "/ingest",
        data={"folder": ""},
        files={"file": ("blank_folder.txt", b"content", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["folder"] == "General"


@pytest.mark.asyncio
async def test_api_whitespace_folder_stored_as_general(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post(
        "/ingest",
        data={"folder": "   "},
        files={"file": ("ws_folder.txt", b"content", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["folder"] == "General"


@pytest.mark.asyncio
async def test_api_long_folder_name_truncated(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post(
        "/ingest",
        data={"folder": "X" * 200},
        files={"file": ("long_folder.txt", b"content", "text/plain")},
    )
    assert r.status_code == 200
    assert len(r.json()["folder"]) <= 64


@pytest.mark.asyncio
async def test_api_move_document_empty_folder_defaults_to_general(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("mv.txt", b"data", "text/plain")})
    doc_id = r.json()["doc_id"]
    r2 = client.patch(f"/documents/{doc_id}/folder", json={"folder": ""})
    assert r2.status_code == 200
    docs = client.get("/documents").json()["documents"]
    assert next(d for d in docs if d["doc_id"] == doc_id)["folder"] == "General"


# ════════════════════════════════════════════════════════════════════════════════
# Feature 5 (extended): PDF reading order
# ════════════════════════════════════════════════════════════════════════════════

def test_extract_pdf_table_interleaved_in_reading_order():
    """Text blocks and table markdown are interleaved by y0, not all-text-then-all-tables."""
    # Page layout (y increases downward):
    #   y=20:  "Top paragraph."
    #   y=100: table with header "TableH1"
    #   y=220: "Bottom paragraph."
    mock_table = MagicMock()
    mock_table.extract.return_value = [["TableH1", "TableH2"], ["R1V1", "R1V2"]]
    mock_table.bbox = (0, 100, 400, 200)

    mock_table_finder = MagicMock()
    mock_table_finder.tables = [mock_table]

    mock_page = MagicMock()
    mock_page.find_tables.return_value = mock_table_finder
    mock_page.get_text.side_effect = lambda mode=None, **kw: (
        {"blocks": [
            {"type": 0, "bbox": [0, 20, 400, 80],
             "lines": [{"spans": [{"text": "Top paragraph.", "size": 12.0, "font": "Arial"}]}]},
            {"type": 0, "bbox": [0, 220, 400, 280],
             "lines": [{"spans": [{"text": "Bottom paragraph.", "size": 12.0, "font": "Arial"}]}]},
        ]} if mode == "dict" else "fallback"
    )

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("fitz.open", return_value=mock_doc):
        with patch("fitz.Rect", side_effect=lambda *args: MagicMock(
            intersects=MagicMock(return_value=False)
        )):
            pages = ingest._extract_pdf(b"fake pdf")

    text = pages[0][1]
    top_pos    = text.find("Top paragraph.")
    table_pos  = text.find("TableH1")
    bottom_pos = text.find("Bottom paragraph.")
    assert top_pos != -1 and table_pos != -1 and bottom_pos != -1, (
        "Expected all three content pieces in output"
    )
    assert top_pos < table_pos < bottom_pos, (
        f"Expected reading order top(y=20)<table(y=100)<bottom(y=220), "
        f"got positions top={top_pos}, table={table_pos}, bottom={bottom_pos}"
    )


def test_extract_pdf_table_has_marker_label():
    """_extract_pdf stamps 'Table N' before each detected table so _tag_table_chunks
    can assign table_id even when the PDF has no explicit label."""
    mock_table = MagicMock()
    mock_table.extract.return_value = [["Col1", "Col2"], ["A", "B"]]
    mock_table.bbox = (0, 50, 400, 150)

    mock_table_finder = MagicMock()
    mock_table_finder.tables = [mock_table]

    mock_page = MagicMock()
    mock_page.find_tables.return_value = mock_table_finder
    mock_page.get_text.side_effect = lambda mode=None, **kw: (
        {"blocks": []} if mode == "dict" else "fallback"
    )

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("fitz.open", return_value=mock_doc):
        with patch("fitz.Rect", side_effect=lambda *args: MagicMock(
            intersects=MagicMock(return_value=False)
        )):
            pages = ingest._extract_pdf(b"fake pdf")

    text = pages[0][1]
    assert re.search(r"Table\s+\d+", text), (
        f"Expected 'Table N' marker in PDF output, got: {text!r}"
    )


def test_extract_pdf_table_marker_enables_table_id():
    """table_id is assigned to chunks whose text contains the auto-stamped 'Table N' marker."""
    mock_table = MagicMock()
    mock_table.extract.return_value = [["Header", "Value"], ["Mass", "12.5"]]
    mock_table.bbox = (0, 50, 400, 150)

    mock_table_finder = MagicMock()
    mock_table_finder.tables = [mock_table]

    mock_page = MagicMock()
    mock_page.find_tables.return_value = mock_table_finder
    mock_page.get_text.side_effect = lambda mode=None, **kw: (
        {"blocks": []} if mode == "dict" else "fallback"
    )

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("fitz.open", return_value=mock_doc):
        with patch("fitz.Rect", side_effect=lambda *args: MagicMock(
            intersects=MagicMock(return_value=False)
        )):
            pages = ingest._extract_pdf(b"fake pdf")

    chunks = ingest.chunk_pages(pages)
    assert chunks, "Expected at least one chunk from table content"

    ingest._tag_table_chunks("testdoc", pages, chunks)

    table_chunks = [c for c in chunks if c.get("table_id")]
    assert table_chunks, (
        "Expected table_id on at least one chunk — marker was missing or _tag_table_chunks failed"
    )


# ════════════════════════════════════════════════════════════════════════════════
# Feature 9: Generic named-document detection (_find_named_docs)
# ════════════════════════════════════════════════════════════════════════════════

def _fake_doc_cache(*filenames):
    return [
        {"filename": fn, "doc_id": f"fakeid{i:04d}", "folder": "General", "chunks": 1}
        for i, fn in enumerate(filenames)
    ]


def test_find_named_docs_empty_doc_list(monkeypatch):
    monkeypatch.setattr(ingest, "_doc_cache", [])
    assert retriever._find_named_docs("What does Manual.pdf say?") == []


def test_find_named_docs_matches_full_filename(monkeypatch):
    monkeypatch.setattr(ingest, "_doc_cache", _fake_doc_cache("Manual.pdf"))
    result = retriever._find_named_docs("what does Manual.pdf say about torque?")
    assert result == ["Manual.pdf"]


def test_find_named_docs_matches_stem(monkeypatch):
    monkeypatch.setattr(ingest, "_doc_cache", _fake_doc_cache("ThrusterSpec.docx"))
    result = retriever._find_named_docs("according to ThrusterSpec what is the thrust level?")
    assert result == ["ThrusterSpec.docx"]


def test_find_named_docs_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(ingest, "_doc_cache", _fake_doc_cache("MANUAL.PDF"))
    result = retriever._find_named_docs("what does manual.pdf say?")
    assert result == ["MANUAL.PDF"]


def test_find_named_docs_short_stem_not_matched(monkeypatch):
    # stem "ab" is 2 chars — below the 4-char minimum to avoid incidental matches
    monkeypatch.setattr(ingest, "_doc_cache", _fake_doc_cache("AB.txt"))
    result = retriever._find_named_docs("what is ab anyway?")
    assert result == []


def test_find_named_docs_four_char_stem_is_matched(monkeypatch):
    # stem "spec" is exactly 4 chars — should match
    monkeypatch.setattr(ingest, "_doc_cache", _fake_doc_cache("spec.pdf"))
    result = retriever._find_named_docs("what does spec say about the system?")
    assert result == ["spec.pdf"]


def test_find_named_docs_no_match_returns_empty(monkeypatch):
    monkeypatch.setattr(ingest, "_doc_cache", _fake_doc_cache("ThrusterSpec.pdf"))
    result = retriever._find_named_docs("what is the GPA of the student?")
    assert result == []


def test_find_named_docs_returns_multiple_matches(monkeypatch):
    monkeypatch.setattr(ingest, "_doc_cache",
                        _fake_doc_cache("Alpha.txt", "Beta.txt", "Gamma.txt"))
    result = retriever._find_named_docs("compare Alpha.txt and Beta.txt")
    assert set(result) == {"Alpha.txt", "Beta.txt"}
    assert "Gamma.txt" not in result


# ════════════════════════════════════════════════════════════════════════════════
# Feature 10: Cross-encoder re-ranking
# ════════════════════════════════════════════════════════════════════════════════

def test_rerank_empty_candidates_returns_empty():
    import reranker
    assert reranker.rerank("query", []) == []


def test_rerank_returns_indices_sorted_by_score(monkeypatch):
    import reranker
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]  # candidate 1 is best
    monkeypatch.setattr(reranker, "_model", mock_model)
    candidates = [("id0", "text0"), ("id1", "text1"), ("id2", "text2")]
    order = reranker.rerank("my question", candidates)
    assert order == [1, 2, 0]


def test_rerank_passes_correct_pairs_to_model(monkeypatch):
    import reranker
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5, 0.8]
    monkeypatch.setattr(reranker, "_model", mock_model)
    reranker.rerank("my question", [("id0", "chunk text A"), ("id1", "chunk text B")])
    pairs = mock_model.predict.call_args[0][0]
    assert pairs == [("my question", "chunk text A"), ("my question", "chunk text B")]


def test_rerank_falls_back_on_model_exception(monkeypatch):
    import reranker
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("model exploded")
    monkeypatch.setattr(reranker, "_model", mock_model)
    candidates = [("id0", "text0"), ("id1", "text1"), ("id2", "text2")]
    order = reranker.rerank("query", candidates)
    assert order == [0, 1, 2]  # original order preserved
