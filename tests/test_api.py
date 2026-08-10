"""Tests for FastAPI endpoints."""
import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

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
