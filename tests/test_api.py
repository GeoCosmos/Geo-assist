"""Tests for FastAPI endpoints."""
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import config
import ingest_nas
import main

client = TestClient(main.app)


def test_health():
    with patch("llm.reachable", new_callable=AsyncMock, return_value=True):
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "chat_model" in body
    assert "embed_model" in body
    assert "chunks_stored" in body


def test_health_ollama_down():
    with patch("llm.reachable", new_callable=AsyncMock, return_value=False):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_index_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert b"Geo-Assist" in r.content


def test_ingest_txt(mock_ollama):
    r = client.post(
        "/ingest",
        files={"file": ("test.txt", b"The GPA is 3.85", "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] >= 1
    assert body["filename"] == "test.txt"


def test_ingest_unsupported_type(mock_ollama):
    r = client.post(
        "/ingest",
        files={"file": ("photo.png", b"\x89PNG...", "image/png")},
    )
    assert r.status_code == 400


def test_ingest_empty_file(mock_ollama):
    r = client.post(
        "/ingest",
        files={"file": ("empty.txt", b"   ", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "empty"


def test_list_documents_empty():
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json()["documents"] == []


def test_list_documents_after_ingest(mock_ollama):
    client.post("/ingest", files={"file": ("doc.txt", b"Some content here.", "text/plain")})
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert any(d["filename"] == "doc.txt" for d in docs)


def test_delete_document(mock_ollama):
    r = client.post("/ingest", files={"file": ("todel.txt", b"Delete me.", "text/plain")})
    doc_id = r.json()["doc_id"]

    r = client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["removed_chunks"] > 0

    docs = client.get("/documents").json()["documents"]
    assert not any(d["doc_id"] == doc_id for d in docs)


def test_delete_nonexistent_document():
    r = client.delete("/documents/deadbeef00000000")
    assert r.status_code == 404


def test_clear_all_documents_empty():
    r = client.delete("/documents")
    assert r.status_code == 200
    assert r.json()["removed_chunks"] == 0


def test_clear_all_documents(mock_ollama):
    client.post("/ingest", files={"file": ("a.txt", b"Document A content.", "text/plain")})
    client.post("/ingest", files={"file": ("b.txt", b"Document B content.", "text/plain")})
    assert len(client.get("/documents").json()["documents"]) == 2

    r = client.delete("/documents")
    assert r.status_code == 200
    assert r.json()["removed_chunks"] > 0

    assert client.get("/documents").json()["documents"] == []


def test_chat_empty_question():
    r = client.post("/chat/stream", json={"question": "  "})
    assert r.status_code == 400


def test_chat_no_docs(mock_ollama):
    r = client.post("/chat/stream", json={"question": "What is the GPA?"})
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
    payloads = [json.loads(ln[len("data: "):]) for ln in lines]
    assert any("token" in p or "sources" in p for p in payloads)


def test_chat_returns_answer_and_sources(mock_ollama):
    client.post("/ingest", files={"file": ("t.txt", b"Cumulative GPA: 3.78", "text/plain")})

    async def fake_stream(question, history=None, folder_filter=None, procedure=None):
        yield {"token": "The GPA is 3.78"}
        yield {"sources": [{"filename": "t.txt", "page": 1}], "done": True}

    with patch("retriever.answer_stream", side_effect=fake_stream):
        r = client.post("/chat/stream", json={"question": "What is the GPA?"})

    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
    payloads = [json.loads(ln[len("data: "):]) for ln in lines]
    tokens = "".join(p["token"] for p in payloads if "token" in p)
    sources = next(p["sources"] for p in payloads if "sources" in p)
    assert "3.78" in tokens
    assert sources[0]["filename"] == "t.txt"


def test_ingest_bulk(mock_ollama):
    r = client.post(
        "/ingest/bulk",
        files=[
            ("files", ("a.txt", b"Content of alpha.", "text/plain")),
            ("files", ("b.txt", b"Content of beta.", "text/plain")),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["total"] == 2
    assert body["skipped"] == []


def test_ingest_bulk_skips_invalid_type(mock_ollama):
    r = client.post(
        "/ingest/bulk",
        files=[
            ("files", ("good.txt", b"Valid content.", "text/plain")),
            ("files", ("bad.exe", b"not allowed", "application/octet-stream")),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert any("bad.exe" in e for e in body["skipped"])


def test_ingest_bulk_all_invalid():
    r = client.post(
        "/ingest/bulk",
        files=[("files", ("photo.png", b"\x89PNG", "image/png"))],
    )
    assert r.status_code == 400


def test_ingest_bulk_returns_job_id(mock_ollama):
    r = client.post(
        "/ingest/bulk",
        files=[("files", ("a.txt", b"Content alpha.", "text/plain"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str)
    assert body["total"] == 1


def test_ingest_status_unknown_job():
    r = client.get("/ingest/status/doesnotexist")
    assert r.status_code == 404


def test_ingest_status_shape(mock_ollama):
    r = client.post(
        "/ingest/bulk",
        files=[("files", ("b.txt", b"Content beta.", "text/plain"))],
    )
    job_id = r.json()["job_id"]
    r2 = client.get(f"/ingest/status/{job_id}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["job_id"] == job_id
    assert body["status"] in {"running", "done", "failed"}
    assert "total" in body
    assert "prepared" in body
    assert "results" in body
    assert "errors" in body


# ── procedure endpoints ───────────────────────────────────────────────────────

def test_procedure_start_unknown_doc():
    r = client.post("/procedure/session/sess_x/start", json={"doc_id": "doesnotexist"})
    assert r.status_code == 404


def test_procedure_start_and_navigate(mock_ollama):
    """Upload a doc, start a procedure session, navigate forward and back."""
    doc_text = b"Valve maintenance specification. Nominal pressure 25 bar."
    ri = client.post("/ingest", files={"file": ("proc.txt", doc_text, "text/plain")})
    assert ri.status_code == 200
    doc_id = ri.json()["doc_id"]

    async def fake_gen(chunks):
        return ["1. Open the valve", "2. Record the reading", "3. Close the valve"]

    with patch("retriever._generate_procedure_steps", side_effect=fake_gen):
        r = client.post("/procedure/session/sess_proc/start", json={"doc_id": doc_id})

    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "proc.txt"
    assert body["step_num"] == 1
    assert body["total_steps"] == 3
    assert body["step_text"] == "1. Open the valve"

    r2 = client.post("/procedure/session/sess_proc/navigate", json={"direction": "next"})
    assert r2.status_code == 200
    assert r2.json()["step_num"] == 2

    r3 = client.post("/procedure/session/sess_proc/navigate", json={"direction": "prev"})
    assert r3.status_code == 200
    assert r3.json()["step_num"] == 1


def test_procedure_navigate_unknown_session():
    r = client.post("/procedure/session/sess_none/navigate", json={"direction": "next"})
    assert r.status_code == 404


def test_procedure_end(mock_ollama):
    """DELETE clears the procedure session; subsequent navigate returns 404."""
    doc_text = b"Valve specification document."
    ri = client.post("/ingest", files={"file": ("p2.txt", doc_text, "text/plain")})
    doc_id = ri.json()["doc_id"]

    async def fake_gen(chunks):
        return ["1. Step one", "2. Step two"]

    with patch("retriever._generate_procedure_steps", side_effect=fake_gen):
        client.post("/procedure/session/sess_end/start", json={"doc_id": doc_id})

    r = client.delete("/procedure/session/sess_end")
    assert r.status_code == 200

    r2 = client.post("/procedure/session/sess_end/navigate", json={"direction": "next"})
    assert r2.status_code == 404


def test_procedure_navigate_clamps_to_bounds(mock_ollama):
    """Navigating past the first/last step stays clamped; no error."""
    doc_text = b"Single step specification."
    ri = client.post("/ingest", files={"file": ("single.txt", doc_text, "text/plain")})
    doc_id = ri.json()["doc_id"]

    async def fake_gen(chunks):
        return ["1. Only step"]

    with patch("retriever._generate_procedure_steps", side_effect=fake_gen):
        client.post("/procedure/session/sess_clamp/start", json={"doc_id": doc_id})

    r_prev = client.post("/procedure/session/sess_clamp/navigate", json={"direction": "prev"})
    assert r_prev.status_code == 200
    assert r_prev.json()["step_num"] == 1

    r_next = client.post("/procedure/session/sess_clamp/navigate", json={"direction": "next"})
    assert r_next.status_code == 200
    assert r_next.json()["step_num"] == r_next.json()["total_steps"]


# ── NAS ingestion endpoints ───────────────────────────────────────────────────

@pytest.fixture
def nas_api_tree(tmp_path, monkeypatch):
    root = tmp_path / "documents"
    (root / "Manuals").mkdir(parents=True)
    (root / "Manuals" / "a.txt").write_bytes(b"alpha body")
    (root / "b.txt").write_bytes(b"beta body")
    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    monkeypatch.setattr(config, "NAS_MANIFEST_PATH", str(tmp_path / "m.db"))
    # The manifest is a module-level singleton; drop it so it reopens against
    # this test's monkeypatched path rather than a previous test's database.
    ingest_nas._manifest_singleton = None
    yield root
    ingest_nas._manifest_singleton = None


def test_nas_health_reports_available_root(nas_api_tree):
    r = client.get("/ingest/nas/health")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["subfolders"] == ["Manuals"]


def test_nas_health_reports_missing_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NAS_ROOT", str(tmp_path / "absent"))
    body = client.get("/ingest/nas/health").json()
    assert body["available"] is False
    assert body["reason"] == "missing"


def test_nas_health_distinguishes_permission_denied(tmp_path, monkeypatch):
    """EACCES and an empty share look identical without this — and a UID mismatch
    between the container and the CIFS mount is the likeliest deploy failure."""
    root = tmp_path / "locked"
    root.mkdir()
    monkeypatch.setattr(config, "NAS_ROOT", str(root))
    monkeypatch.setattr(os, "access", lambda p, m: False)
    body = client.get("/ingest/nas/health").json()
    assert body["available"] is False
    assert body["reason"] == "unreadable"


def test_nas_preview_returns_counts(nas_api_tree, mock_ollama):
    r = client.post("/ingest/nas/preview", json={"subpath": ""})
    assert r.status_code == 200
    assert r.json()["new"] == 2


def test_nas_preview_rejects_escape(nas_api_tree):
    r = client.post("/ingest/nas/preview", json={"subpath": "../.."})
    assert r.status_code == 403


def test_nas_preview_rejects_absolute_path(nas_api_tree):
    r = client.post("/ingest/nas/preview", json={"subpath": "/etc"})
    assert r.status_code == 403


def test_nas_scan_returns_job_id(nas_api_tree, mock_ollama):
    r = client.post("/ingest/nas/scan", json={"subpath": ""})
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_nas_scan_rejects_escape(nas_api_tree):
    r = client.post("/ingest/nas/scan", json={"subpath": "../../etc"})
    assert r.status_code == 403


# ── source-file availability ──────────────────────────────────────────────────

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
    assert client.get("/documents/nothex/file").status_code == 404


# ── figure serving ────────────────────────────────────────────────────────────

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


def test_figure_route_rejects_non_integer_page(tmp_path, monkeypatch):
    """FastAPI's int converter rejects these before any path is constructed."""
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    assert client.get(f"/documents/{'a' * 16}/figures/notanint/0").status_code == 422
