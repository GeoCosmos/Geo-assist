"""Comprehensive tests for all new features.

Features covered:
  1. Code file support (.py, .js, .ts, .json, .sql, .md, .sh, .go, .rs, ...)
  2. Russian / Armenian language support (system prompt)
  3. Folder organisation (metadata, list_folders, move_document, BM25 filtering, API)
  4. Audio / video transcription (_extract_audio dispatch, segment grouping, model params)
  5. PDF table extraction (_table_to_markdown format, empty table, reading order)
  6. Comparison table instruction (system prompt)
  7. Multi-user auth (passwords, tokens, user store, API endpoints)
  8. Where-clause helpers (_with_folder, _with_access) for retrieval filtering
  9. Generic named-document detection (_find_named_docs)
 10. Cross-encoder re-ranking (reranker.rerank)
"""
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

import auth
import bm25_index
import config
import ingest
import retriever


# ════════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def auth_store(tmp_path, monkeypatch):
    """Redirect auth file storage to an isolated temp dir and reset secret cache."""
    monkeypatch.setattr(auth, "_USERS_PATH", tmp_path / "users.json")
    monkeypatch.setattr(auth, "_SECRET_PATH", tmp_path / "secret.key")
    monkeypatch.setattr(auth, "_REVOCATIONS_PATH", tmp_path / "revocations.json")
    monkeypatch.setattr(auth, "_secret_cache", None)
    yield tmp_path


@pytest.fixture
def auth_api_client(monkeypatch, auth_store):
    """FastAPI TestClient with AUTH_ENABLED=True and auth injected into main."""
    import main
    from fastapi.testclient import TestClient
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    # auth is only imported in main when AUTH_ENABLED=True at startup; inject here
    monkeypatch.setattr(main, "auth", auth, raising=False)
    return TestClient(main.app)


# ════════════════════════════════════════════════════════════════════════════════
# Feature 1: Code file support
# ════════════════════════════════════════════════════════════════════════════════

def test_python_extracted_as_text():
    pages = ingest.extract_pages(b"def greet():\n    return 'hello'\n", "greet.py")
    assert len(pages) == 1
    assert "def greet" in pages[0][1]


def test_javascript_extracted():
    pages = ingest.extract_pages(b"const pi = Math.PI;", "math.js")
    assert "const pi" in pages[0][1]


def test_typescript_extracted():
    pages = ingest.extract_pages(b"interface User { id: number; }", "user.ts")
    assert "interface User" in pages[0][1]


def test_json_extracted():
    pages = ingest.extract_pages(b'{"env": "prod", "port": 8743}', "config.json")
    assert '"env"' in pages[0][1]


def test_sql_extracted():
    pages = ingest.extract_pages(b"SELECT id, name FROM users WHERE active = 1;", "query.sql")
    assert "SELECT" in pages[0][1]


def test_markdown_extracted():
    pages = ingest.extract_pages(b"# Title\n\nSome content here.", "README.md")
    assert "Title" in pages[0][1]


def test_yaml_extracted():
    pages = ingest.extract_pages(b"version: '3'\nport: 80", "docker.yml")
    assert "version" in pages[0][1]


def test_shell_script_extracted():
    pages = ingest.extract_pages(b"#!/bin/bash\necho 'Deploying...'", "deploy.sh")
    assert "Deploying" in pages[0][1]


def test_go_file_extracted():
    pages = ingest.extract_pages(b"package main\nfunc main() {}", "main.go")
    assert "package main" in pages[0][1]


def test_rust_file_extracted():
    pages = ingest.extract_pages(b"fn main() { println!(\"hi\"); }", "main.rs")
    assert "fn main" in pages[0][1]


def test_csharp_extracted():
    pages = ingest.extract_pages(b"public class Foo { public void Bar() {} }", "Foo.cs")
    assert "class Foo" in pages[0][1]


def test_toml_extracted():
    pages = ingest.extract_pages(b"[package]\nname = \"geo-assist\"\nversion = \"1.0\"", "Cargo.toml")
    assert "geo-assist" in pages[0][1]


def test_ini_extracted():
    pages = ingest.extract_pages(b"[database]\nhost = localhost\nport = 5432", "config.ini")
    assert "localhost" in pages[0][1]


@pytest.mark.asyncio
async def test_python_file_ingest_pipeline(mock_ollama):
    result = await ingest.ingest(b"class Engine:\n    def start(self): pass\n", "engine.py")
    assert result["status"] == "ok"
    assert result["chunks"] >= 1


@pytest.mark.asyncio
async def test_api_accepts_python_file(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("app.py", b"x = 1\n", "text/plain")})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_accepts_typescript_file(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("types.ts", b"type X = string;", "text/plain")})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_api_accepts_json_file(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("cfg.json", b'{"a":1}', "application/json")})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_api_accepts_markdown_file(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("docs.md", b"# Heading", "text/plain")})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_api_accepts_sql_file(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    r = client.post("/ingest", files={"file": ("schema.sql", b"CREATE TABLE t (id INT);", "text/plain")})
    assert r.status_code == 200


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
async def test_list_documents_includes_folder_access_owner(mock_ollama):
    await ingest.ingest(
        b"Sensitive report.",
        "secret.txt",
        folder="Confidential",
        access="private",
        owner="alice",
    )
    docs = ingest.list_documents()
    doc = next(d for d in docs if d["filename"] == "secret.txt")
    assert doc["folder"] == "Confidential"
    assert doc["access"] == "private"
    assert doc["owner"] == "alice"


@pytest.mark.asyncio
async def test_ingest_stores_public_access(mock_ollama):
    result = await ingest.ingest(b"Public content.", "pub.txt", access="public")
    assert result["access"] == "public"


@pytest.mark.asyncio
async def test_ingest_stores_private_access(mock_ollama):
    result = await ingest.ingest(b"Private content.", "priv.txt", access="private", owner="bob")
    assert result["access"] == "private"
    assert result["owner"] == "bob"


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
# Feature 4: Audio / video transcription
# ════════════════════════════════════════════════════════════════════════════════

def test_audio_video_extensions_set_is_complete():
    expected = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mkv", ".mov", ".avi"}
    assert expected.issubset(ingest._AUDIO_VIDEO_EXTS)


def test_extract_pages_dispatches_mp3_to_audio_extractor():
    with patch.object(ingest, "_extract_audio", return_value=[(1, "Transcript.")]) as m:
        pages = ingest.extract_pages(b"audio bytes", "meeting.mp3")
    m.assert_called_once_with(b"audio bytes", "meeting.mp3")
    assert pages == [(1, "Transcript.")]


def test_extract_pages_dispatches_mp4_to_audio_extractor():
    with patch.object(ingest, "_extract_audio", return_value=[(1, "Video text.")]) as m:
        ingest.extract_pages(b"video bytes", "call.mp4")
    m.assert_called_once()


def test_extract_pages_dispatches_wav_to_audio_extractor():
    with patch.object(ingest, "_extract_audio", return_value=[(1, "WAV text.")]) as m:
        ingest.extract_pages(b"wav", "audio.wav")
    m.assert_called_once()


def test_extract_pages_dispatches_m4a_to_audio_extractor():
    with patch.object(ingest, "_extract_audio", return_value=[(1, "m4a text.")]) as m:
        ingest.extract_pages(b"m4a", "voice.m4a")
    m.assert_called_once()


def test_extract_pages_does_not_dispatch_txt_to_audio():
    with patch.object(ingest, "_extract_audio") as m:
        ingest.extract_pages(b"plain text", "notes.txt")
    m.assert_not_called()


def test_extract_audio_segments_grouped_into_pages():
    # The page break fires after appending a segment whose end >= page_start + 30s.
    # So the segment that crosses 30s lands on page 1; the segment AFTER that goes to page 2.
    seg1 = MagicMock(text="First sentence here.", start=0.0, end=5.0)
    seg2 = MagicMock(text="Second sentence.", start=5.0, end=10.0)
    seg3 = MagicMock(text="This one crosses 30s.", start=30.0, end=36.0)  # triggers break
    seg4 = MagicMock(text="New page segment.", start=36.0, end=40.0)       # lands on page 2

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg1, seg2, seg3, seg4], MagicMock())
    mock_fw = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model

    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        pages = ingest._extract_audio(b"audio", "meeting.mp3")

    assert len(pages) >= 2
    assert "First sentence" in pages[0][1]
    assert "Second sentence" in pages[0][1]
    assert "New page segment" in pages[1][1]


def test_extract_audio_short_clip_is_one_page():
    seg = MagicMock(text="Quick note.", start=0.0, end=3.0)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg], MagicMock())
    mock_fw = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model

    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        pages = ingest._extract_audio(b"data", "note.wav")

    assert pages[0][0] == 1
    assert "Quick note" in pages[0][1]


def test_extract_audio_empty_transcription_returns_placeholder():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())
    mock_fw = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model

    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        pages = ingest._extract_audio(b"silence", "silent.wav")

    assert pages == [(1, "")]


def test_extract_audio_uses_tiny_cpu_int8_model():
    """Verifies the correct model variant is used (air-gap safe, CPU-only)."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())
    mock_fw = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model

    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        ingest._extract_audio(b"data", "audio.mp3")

    mock_fw.WhisperModel.assert_called_once_with("tiny", device="cpu", compute_type="int8")


@pytest.mark.asyncio
async def test_api_accepts_audio_extensions(mock_ollama):
    """API endpoint should accept audio/video MIME types and ingest via mock extractor."""
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)

    with patch.object(ingest, "_extract_audio", return_value=[(1, "Meeting transcript.")]):
        r = client.post(
            "/ingest",
            files={"file": ("standup.mp3", b"audio_data", "audio/mpeg")},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_accepts_mp4_video(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)

    with patch.object(ingest, "_extract_audio", return_value=[(1, "Video meeting notes.")]):
        r = client.post(
            "/ingest",
            files={"file": ("weekly.mp4", b"video_data", "video/mp4")},
        )
    assert r.status_code == 200


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
    """When find_tables returns no tables, plain page.get_text() should be used."""

    mock_table_finder = MagicMock()
    mock_table_finder.tables = []

    mock_page = MagicMock()
    mock_page.find_tables.return_value = mock_table_finder
    mock_page.get_text.return_value = "Plain page text content."

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.__len__ = MagicMock(return_value=1)

    with patch("fitz.open", return_value=mock_doc):
        pages = ingest._extract_pdf(b"fake pdf")

    assert pages[0][1] == "Plain page text content."
    mock_page.get_text.assert_called_once()


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

    mock_block = (0, 0, 400, 50, "Header text above table.", 0, 0)
    mock_page.get_text.side_effect = lambda mode=None, **kw: (
        [mock_block] if mode == "blocks" else "Header text above table."
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
# Feature 7a: Where-clause helpers (_with_folder, _with_access)
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


def test_with_access_auth_disabled_is_passthrough(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", False)
    where = {"folder": "HR"}
    user = {"username": "alice", "role": "user"}
    assert retriever._with_access(where, user) is where


def test_with_access_no_current_user_is_passthrough(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    where = {"folder": "HR"}
    assert retriever._with_access(where, None) is where


def test_with_access_admin_user_is_passthrough(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    where = {"folder": "HR"}
    admin = {"username": "admin", "role": "admin"}
    assert retriever._with_access(where, admin) is where


def test_with_access_regular_user_builds_or_clause(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    user = {"username": "alice", "role": "user"}
    result = retriever._with_access(None, user)
    assert result == {"$or": [{"access": "public"}, {"owner": "alice"}]}


def test_with_access_regular_user_with_existing_where_builds_and_clause(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    user = {"username": "bob", "role": "user"}
    where = {"folder": "HR"}
    result = retriever._with_access(where, user)
    assert result["$and"][0] == where
    assert result["$and"][1] == {"$or": [{"access": "public"}, {"owner": "bob"}]}


def test_with_folder_then_with_access_chaining(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    user = {"username": "carol", "role": "user"}
    where = retriever._with_folder(None, "Finance")
    result = retriever._with_access(where, user)
    # Should be: {$and: [{folder: Finance}, {$or: [{access: public}, {owner: carol}]}]}
    assert "$and" in result
    assert {"folder": "Finance"} in result["$and"]
    access_clause = next(x for x in result["$and"] if "$or" in x)
    assert {"access": "public"} in access_clause["$or"]
    assert {"owner": "carol"} in access_clause["$or"]


# ════════════════════════════════════════════════════════════════════════════════
# Feature 7b: Auth module (passwords, tokens, user store)
# ════════════════════════════════════════════════════════════════════════════════

def test_hash_password_and_verify_correct(auth_store):
    h, s = auth.hash_password("correct-horse-battery-staple")
    assert auth.verify_password("correct-horse-battery-staple", h, s)


def test_verify_password_rejects_wrong_password(auth_store):
    h, s = auth.hash_password("rightpassword")
    assert not auth.verify_password("wrongpassword", h, s)


def test_hash_password_produces_different_salts_each_call(auth_store):
    h1, s1 = auth.hash_password("same")
    h2, s2 = auth.hash_password("same")
    assert s1 != s2  # salt is random
    assert h1 != h2  # different salt → different hash


def test_create_token_and_verify_roundtrip(auth_store):
    token = auth.create_token("alice", "user")
    result = auth.verify_token(token)
    assert result["username"] == "alice"
    assert result["role"] == "user"


def test_verify_token_rejects_tampered_signature(auth_store):
    token = auth.create_token("alice", "user")
    p64, sig = token.rsplit(".", 1)
    bad_token = f"{p64}.{'0' * len(sig)}"
    with pytest.raises(ValueError, match="Invalid token signature"):
        auth.verify_token(bad_token)


def test_verify_token_rejects_malformed_token(auth_store):
    with pytest.raises(ValueError, match="Malformed token"):
        auth.verify_token("notavalidtoken")


def test_verify_token_rejects_expired_token(auth_store):
    import time
    # Build a token with exp in the past
    import base64
    import hmac as _hmac
    import hashlib
    import json
    payload = json.dumps({"u": "alice", "r": "user", "exp": int(time.time()) - 1})
    p64 = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = _hmac.new(auth._secret(), p64.encode(), hashlib.sha256).hexdigest()
    expired_token = f"{p64}.{sig}"
    with pytest.raises(ValueError, match="expired"):
        auth.verify_token(expired_token)


def test_verify_token_rejects_bad_payload(auth_store):
    import base64
    import hmac as _hmac
    import hashlib
    p64 = base64.urlsafe_b64encode(b"not json").decode()
    sig = _hmac.new(auth._secret(), p64.encode(), hashlib.sha256).hexdigest()
    bad = f"{p64}.{sig}"
    with pytest.raises(ValueError, match="Malformed token payload"):
        auth.verify_token(bad)


def test_create_user_and_authenticate(auth_store):
    auth.create_user("testuser", "testpass", role="user")
    result = auth.authenticate("testuser", "testpass")
    assert result is not None
    assert result["username"] == "testuser"
    assert result["role"] == "user"


def test_authenticate_wrong_password_returns_none(auth_store):
    auth.create_user("testuser", "rightpass")
    assert auth.authenticate("testuser", "wrongpass") is None


def test_authenticate_nonexistent_user_returns_none(auth_store):
    assert auth.authenticate("nobody", "password") is None


def test_has_users_false_when_empty(auth_store):
    assert not auth.has_users()


def test_has_users_true_after_create(auth_store):
    auth.create_user("first", "pass")
    assert auth.has_users()


def test_create_user_duplicate_raises(auth_store):
    auth.create_user("alice", "pass1")
    with pytest.raises(ValueError, match="already exists"):
        auth.create_user("alice", "pass2")


def test_delete_user_removes_user(auth_store):
    auth.create_user("todelete", "pass")
    assert auth.delete_user("todelete")
    assert auth.authenticate("todelete", "pass") is None


def test_delete_user_nonexistent_returns_false(auth_store):
    assert not auth.delete_user("nobody")


def test_update_role_changes_role(auth_store):
    auth.create_user("promoted", "pass", role="user")
    assert auth.update_role("promoted", "admin")
    result = auth.authenticate("promoted", "pass")
    assert result["role"] == "admin"


def test_update_role_nonexistent_returns_false(auth_store):
    assert not auth.update_role("nobody", "admin")


def test_list_users_returns_all(auth_store):
    auth.create_user("alice", "pass1", role="admin")
    auth.create_user("bob", "pass2", role="user")
    users = auth.list_users()
    names = {u["username"] for u in users}
    assert "alice" in names
    assert "bob" in names


def test_list_users_no_passwords_exposed(auth_store):
    auth.create_user("alice", "secret123")
    users = auth.list_users()
    user = next(u for u in users if u["username"] == "alice")
    assert "hash" not in user
    assert "salt" not in user
    assert "password" not in user
    assert "secret123" not in str(user)


def test_secret_persisted_and_reloaded(auth_store, tmp_path):
    secret1 = auth._secret()
    auth._secret_cache = None  # force re-read from disk
    secret2 = auth._secret()
    assert secret1 == secret2  # same key after reload


# ════════════════════════════════════════════════════════════════════════════════
# Feature 7c: Auth API endpoints
# ════════════════════════════════════════════════════════════════════════════════

def test_auth_setup_creates_first_admin(auth_api_client):
    r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    assert r.status_code == 200
    body = r.json()
    assert "token" in body
    assert body["username"] == "admin"
    assert body["role"] == "admin"


def test_auth_setup_second_call_returns_409(auth_api_client):
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    r = auth_api_client.post("/auth/setup", json={"username": "other", "password": "otherpass"})
    assert r.status_code == 409


def test_auth_login_valid_credentials(auth_api_client):
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    r = auth_api_client.post("/auth/login", json={"username": "admin", "password": "adminpass"})
    assert r.status_code == 200
    assert "token" in r.json()


def test_auth_login_wrong_password_returns_401(auth_api_client):
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    r = auth_api_client.post("/auth/login", json={"username": "admin", "password": "wrongpass"})
    assert r.status_code == 401


def test_auth_login_nonexistent_user_returns_401(auth_api_client):
    r = auth_api_client.post("/auth/login", json={"username": "nobody", "password": "pass"})
    assert r.status_code == 401


def test_auth_me_returns_user_info(auth_api_client):
    r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    token = r.json()["token"]
    r2 = auth_api_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert r2.json()["username"] == "admin"
    assert r2.json()["role"] == "admin"


def test_auth_me_no_token_returns_401(auth_api_client):
    r = auth_api_client.get("/auth/me")
    assert r.status_code == 401


def test_auth_me_bad_token_returns_401(auth_api_client):
    r = auth_api_client.get("/auth/me", headers={"Authorization": "Bearer badtoken"})
    assert r.status_code == 401


def test_auth_list_users_requires_admin(auth_api_client):
    # Setup admin and regular user
    setup_r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    admin_token = setup_r.json()["token"]

    auth.create_user("regularuser", "userpass", role="user")
    user_token = auth.create_token("regularuser", "user")

    # Admin can list users
    r = auth_api_client.get("/auth/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200
    usernames = {u["username"] for u in r.json()["users"]}
    assert "admin" in usernames

    # Regular user cannot list users
    r2 = auth_api_client.get("/auth/users", headers={"Authorization": f"Bearer {user_token}"})
    assert r2.status_code == 403


def test_auth_create_user_endpoint(auth_api_client):
    setup_r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    admin_token = setup_r.json()["token"]

    r = auth_api_client.post(
        "/auth/users",
        json={"username": "newuser", "password": "newpass", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    assert r.json()["username"] == "newuser"

    # Verify they can log in
    login_r = auth_api_client.post("/auth/login", json={"username": "newuser", "password": "newpass"})
    assert login_r.status_code == 200


def test_auth_create_duplicate_user_returns_409(auth_api_client):
    setup_r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    admin_token = setup_r.json()["token"]

    auth_api_client.post(
        "/auth/users",
        json={"username": "dup", "password": "pass", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    r = auth_api_client.post(
        "/auth/users",
        json={"username": "dup", "password": "pass2", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 409


def test_auth_delete_user_endpoint(auth_api_client):
    setup_r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    admin_token = setup_r.json()["token"]

    auth_api_client.post(
        "/auth/users",
        json={"username": "todelete", "password": "pass", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    r = auth_api_client.delete(
        "/auth/users/todelete",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    assert r.json()["deleted"] == "todelete"


def test_auth_delete_nonexistent_user_returns_404(auth_api_client):
    setup_r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    admin_token = setup_r.json()["token"]

    r = auth_api_client.delete(
        "/auth/users/nobody",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 404


def test_auth_update_role_endpoint(auth_api_client):
    setup_r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    admin_token = setup_r.json()["token"]

    auth_api_client.post(
        "/auth/users",
        json={"username": "promoteme", "password": "pass", "role": "user"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    r = auth_api_client.patch(
        "/auth/users/promoteme/role",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "admin"


def test_auth_endpoints_return_404_when_auth_disabled():
    """Auth endpoints must return 404 when GEO_AUTH=0 (default) to avoid confusion."""
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    assert client.post("/auth/setup", json={"username": "a", "password": "b"}).status_code == 404
    assert client.post("/auth/login", json={"username": "a", "password": "b"}).status_code == 404


def test_auth_ingest_sets_owner_from_token(auth_api_client, mock_ollama):
    """When auth is enabled, uploaded doc should be owned by the logged-in user."""
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})

    auth.create_user("uploader", "uploaderpass", role="user")
    login_r = auth_api_client.post("/auth/login", json={"username": "uploader", "password": "uploaderpass"})
    uploader_token = login_r.json()["token"]

    r = auth_api_client.post(
        "/ingest",
        data={"access": "private"},
        files={"file": ("owned.txt", b"Uploader's private document.", "text/plain")},
        headers={"Authorization": f"Bearer {uploader_token}"},
    )
    assert r.status_code == 200

    docs = ingest.list_documents()
    doc = next((d for d in docs if d["filename"] == "owned.txt"), None)
    assert doc is not None
    assert doc["owner"] == "uploader"
    assert doc["access"] == "private"


# ════════════════════════════════════════════════════════════════════════════════
# Fix 1: Graceful audio error (faster-whisper not installed → 501, clear message)
# ════════════════════════════════════════════════════════════════════════════════

def test_extract_audio_raises_runtime_error_when_faster_whisper_missing():
    with patch.dict(sys.modules, {"faster_whisper": None}):
        with pytest.raises(RuntimeError, match="faster-whisper"):
            ingest._extract_audio(b"data", "meeting.mp3")


def test_extract_audio_error_mentions_ffmpeg():
    with patch.dict(sys.modules, {"faster_whisper": None}):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            ingest._extract_audio(b"data", "meeting.mp3")


def test_extract_audio_error_mentions_install_command():
    with patch.dict(sys.modules, {"faster_whisper": None}):
        with pytest.raises(RuntimeError, match="pip install faster-whisper"):
            ingest._extract_audio(b"data", "meeting.mp3")


@pytest.mark.asyncio
async def test_api_returns_501_for_audio_when_faster_whisper_missing(mock_ollama):
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    with patch.dict(sys.modules, {"faster_whisper": None}):
        r = client.post(
            "/ingest",
            files={"file": ("standup.mp3", b"fake_audio", "audio/mpeg")},
        )
    assert r.status_code == 501
    assert "faster-whisper" in r.json()["detail"]


@pytest.mark.asyncio
async def test_api_audio_error_is_not_cryptic_500(mock_ollama):
    """Should be 501 with instructions, not a raw 500 ModuleNotFoundError."""
    from fastapi.testclient import TestClient
    import main
    client = TestClient(main.app)
    with patch.dict(sys.modules, {"faster_whisper": None}):
        r = client.post(
            "/ingest",
            files={"file": ("call.mp4", b"fake_video", "video/mp4")},
        )
    assert r.status_code != 500
    assert "ModuleNotFoundError" not in r.json().get("detail", "")


# ════════════════════════════════════════════════════════════════════════════════
# Fix 2: Login rate limiting
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fresh_login_state(monkeypatch):
    """Isolate _login_failures dict between rate-limit tests."""
    import main
    monkeypatch.setattr(main, "_login_failures", {})


def test_login_rate_limit_triggers_after_max_failures(auth_api_client, fresh_login_state):
    import main
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    auth.create_user("ratelimited", "correctpass", role="user")

    for _ in range(main._MAX_LOGIN_FAILURES):
        r = auth_api_client.post("/auth/login", json={"username": "ratelimited", "password": "wrong"})
        assert r.status_code == 401

    r = auth_api_client.post("/auth/login", json={"username": "ratelimited", "password": "wrong"})
    assert r.status_code == 429
    assert "Too many" in r.json()["detail"]


def test_login_rate_limit_does_not_affect_other_users(auth_api_client, fresh_login_state):
    import main
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    auth.create_user("alice", "alicepass", role="user")
    auth.create_user("bob", "bobpass", role="user")

    for _ in range(main._MAX_LOGIN_FAILURES):
        auth_api_client.post("/auth/login", json={"username": "alice", "password": "wrong"})

    # Alice is locked out
    assert auth_api_client.post("/auth/login", json={"username": "alice", "password": "wrong"}).status_code == 429
    # Bob is not
    assert auth_api_client.post("/auth/login", json={"username": "bob", "password": "wrong"}).status_code == 401


def test_successful_login_clears_failure_count(auth_api_client, fresh_login_state):
    import main
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    auth.create_user("charlie", "correctpass", role="user")

    for _ in range(main._MAX_LOGIN_FAILURES - 1):
        auth_api_client.post("/auth/login", json={"username": "charlie", "password": "wrong"})

    # Correct login clears the counter
    r = auth_api_client.post("/auth/login", json={"username": "charlie", "password": "correctpass"})
    assert r.status_code == 200

    # Counter was cleared — wrong attempts from zero again, not 429
    r2 = auth_api_client.post("/auth/login", json={"username": "charlie", "password": "wrong"})
    assert r2.status_code == 401


def test_login_lockout_message_mentions_retry_time(auth_api_client, fresh_login_state):
    import main
    auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    auth.create_user("locked", "pass", role="user")

    for _ in range(main._MAX_LOGIN_FAILURES):
        auth_api_client.post("/auth/login", json={"username": "locked", "password": "wrong"})

    r = auth_api_client.post("/auth/login", json={"username": "locked", "password": "wrong"})
    assert str(main._LOCKOUT_SECONDS) in r.json()["detail"]


# ════════════════════════════════════════════════════════════════════════════════
# Fix 3: Token revocation
# ════════════════════════════════════════════════════════════════════════════════

def test_deleted_user_token_is_rejected(auth_store):
    auth.create_user("victim", "pass", role="user")
    token = auth.create_token("victim", "user")

    assert auth.verify_token(token)["username"] == "victim"  # valid before deletion

    auth.delete_user("victim")

    with pytest.raises(ValueError, match="revoked"):
        auth.verify_token(token)


def test_recreated_user_old_token_still_rejected(auth_store):
    """A token issued before deletion must not become valid just because the username was re-used."""
    auth.create_user("recycled", "pass1", role="user")
    old_token = auth.create_token("recycled", "user")
    auth.delete_user("recycled")

    # Re-create with same username
    auth.create_user("recycled", "pass2", role="user")

    with pytest.raises(ValueError, match="revoked"):
        auth.verify_token(old_token)


def test_recreated_user_new_token_is_valid(auth_store):
    auth.create_user("recycled2", "pass1", role="user")
    auth.delete_user("recycled2")
    auth.create_user("recycled2", "pass2", role="user")

    new_token = auth.create_token("recycled2", "user")
    result = auth.verify_token(new_token)
    assert result["username"] == "recycled2"


def test_role_update_revokes_old_token(auth_store):
    """After update_role, the old token (with the old role) must be rejected."""
    auth.create_user("promoted", "pass", role="user")
    old_token = auth.create_token("promoted", "user")

    assert auth.verify_token(old_token)["role"] == "user"

    auth.update_role("promoted", "admin")

    with pytest.raises(ValueError, match="revoked"):
        auth.verify_token(old_token)


def test_role_update_new_token_carries_new_role(auth_store):
    auth.create_user("promoted2", "pass", role="user")
    auth.update_role("promoted2", "admin")

    new_token = auth.create_token("promoted2", "admin")
    result = auth.verify_token(new_token)
    assert result["role"] == "admin"


def test_api_deleted_user_token_rejected(auth_api_client):
    setup_r = auth_api_client.post("/auth/setup", json={"username": "admin", "password": "adminpass"})
    admin_token = setup_r.json()["token"]

    auth.create_user("todelete", "pass", role="user")
    user_token = auth.create_token("todelete", "user")

    # Token works before deletion
    r1 = auth_api_client.get("/auth/me", headers={"Authorization": f"Bearer {user_token}"})
    assert r1.status_code == 200

    # Delete the user
    auth_api_client.delete("/auth/users/todelete",
                           headers={"Authorization": f"Bearer {admin_token}"})

    # Token must now be rejected
    r2 = auth_api_client.get("/auth/me", headers={"Authorization": f"Bearer {user_token}"})
    assert r2.status_code == 401


def test_revocation_file_created_on_delete(auth_store):
    auth.create_user("deleteme", "pass", role="user")
    auth.delete_user("deleteme")

    revs = auth._load_revocations()
    assert "deleteme" in revs
    assert isinstance(revs["deleteme"], (int, float))


def test_revocation_cleared_on_user_recreation(auth_store):
    auth.create_user("reborn", "pass1", role="user")
    auth.delete_user("reborn")

    assert "reborn" in auth._load_revocations()

    auth.create_user("reborn", "pass2", role="user")

    assert "reborn" not in auth._load_revocations()


# ════════════════════════════════════════════════════════════════════════════════
# Fix 4: Dead import removed (verified by import)
# ════════════════════════════════════════════════════════════════════════════════

def test_retriever_does_not_import_auth_module():
    """_auth_mod must not exist in retriever's namespace — it was dead code."""
    assert not hasattr(retriever, "_auth_mod"), (
        "retriever still has the unused _auth_mod import; it should have been removed"
    )


def test_retriever_still_works_with_auth_disabled(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", False)
    # _with_access with AUTH_ENABLED=False should pass through unchanged
    where = {"folder": "HR"}
    result = retriever._with_access(where, {"username": "alice", "role": "user"})
    assert result is where


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

    block_top    = (0,  20, 400,  80, "Top paragraph.",    0, 0)
    block_bottom = (0, 220, 400, 280, "Bottom paragraph.", 0, 0)

    mock_page = MagicMock()
    mock_page.find_tables.return_value = mock_table_finder
    mock_page.get_text.side_effect = lambda mode=None, **kw: (
        [block_top, block_bottom] if mode == "blocks" else "fallback"
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
        [] if mode == "blocks" else "fallback"
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
        [] if mode == "blocks" else "fallback"
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
        {"filename": fn, "doc_id": f"fakeid{i:04d}", "folder": "General",
         "access": "public", "owner": "", "chunks": 1}
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
