"""Tests for the offline vision indexer (vision_index.py)."""
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
import ingest
import llm
import vision_index


# ── helpers ───────────────────────────────────────────────────────────────────

def _doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


# ── parse_args ────────────────────────────────────────────────────────────────

def test_parse_args_defaults(tmp_path):
    args = vision_index.parse_args(["--dir", str(tmp_path)])
    assert args.model == "moondream"
    assert args.skip_existing is False
    assert args.limit == 0


def test_parse_args_custom(tmp_path):
    args = vision_index.parse_args([
        "--dir", str(tmp_path),
        "--model", "llava:7b",
        "--skip-existing",
        "--limit", "5",
    ])
    assert args.model == "llava:7b"
    assert args.skip_existing is True
    assert args.limit == 5


# ── process_file: file not ingested ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_file_not_ingested(tmp_path):
    """Returns not_ingested when the file has no matching doc_id in ChromaDB."""
    f = tmp_path / "mystery.pdf"
    f.write_bytes(b"not a real pdf")
    result = await vision_index.process_file(f)
    assert result["status"] == "not_ingested"
    assert result["images"] == 0


# ── process_file: no images ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_file_no_images(tmp_path, mock_ollama, monkeypatch):
    """Returns no_images when the file has no qualifying images."""
    data = b"Text-only document with no images."
    f = tmp_path / "text.pdf"
    f.write_bytes(data)

    monkeypatch.setattr(ingest, "extract_pages",
                        lambda d, fn: [(1, "Text-only document with no images.")])
    await ingest.ingest(data, "text.pdf")

    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    monkeypatch.setattr(ingest, "extract_images", lambda d, fn: [])

    result = await vision_index.process_file(f)
    assert result["status"] == "no_images"
    assert result["images"] == 0


# ── process_file: skip_existing ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_file_skip_existing(tmp_path, mock_ollama, monkeypatch):
    """Returns skipped_existing when the doc already has image chunks."""
    data = b"Document with existing image analysis."
    f = tmp_path / "doc.pdf"
    f.write_bytes(data)

    monkeypatch.setattr(ingest, "extract_pages",
                        lambda d, fn: [(1, "Document with existing image analysis.")])
    await ingest.ingest(data, "doc.pdf")

    # Manually plant a fake image chunk for this doc
    doc_id = _doc_id(data)
    col = ingest._db()
    col.add(
        ids=[f"{doc_id}_1_-100"],
        embeddings=[[0.1] * 768],
        documents=["[doc.pdf] [Figure on page 1]: A diagram."],
        metadatas=[{"doc_id": doc_id, "filename": "doc.pdf",
                    "page": 1, "folder": "General",
                    "access": "public", "owner": "", "chunk_type": "image"}],
    )

    result = await vision_index.process_file(f, skip_existing=True)
    assert result["status"] == "skipped_existing"
    assert result["images"] == 1


# ── process_file: analysis failed ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_file_analysis_failed(tmp_path, mock_ollama, monkeypatch):
    """Returns analysis_failed when vision model returns no descriptions."""
    data = b"PDF with images that fail analysis."
    f = tmp_path / "fail.pdf"
    f.write_bytes(data)

    monkeypatch.setattr(ingest, "extract_pages",
                        lambda d, fn: [(1, "PDF with images that fail analysis.")])
    await ingest.ingest(data, "fail.pdf")

    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    monkeypatch.setattr(ingest, "extract_images",
                        lambda d, fn: [(1, 0, b"x" * config.MIN_IMAGE_BYTES)])
    monkeypatch.setattr(ingest, "_analyze_images",
                        AsyncMock(return_value=[]))

    result = await vision_index.process_file(f)
    assert result["status"] == "analysis_failed"
    assert result["images"] == 0


# ── process_file: happy path ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_file_stores_image_chunks(tmp_path, mock_ollama, monkeypatch):
    """Image chunks are stored in ChromaDB after successful analysis."""
    data = b"Engineering document with figures."
    f = tmp_path / "eng.pdf"
    f.write_bytes(data)

    monkeypatch.setattr(ingest, "extract_pages",
                        lambda d, fn: [(1, "Engineering document with figures.")])
    await ingest.ingest(data, "eng.pdf")

    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    monkeypatch.setattr(ingest, "extract_images",
                        lambda d, fn: [(1, 0, b"x" * config.MIN_IMAGE_BYTES)])
    monkeypatch.setattr(ingest, "_analyze_images",
                        AsyncMock(return_value=[(1, 0, "A block diagram of the thruster.")]))

    result = await vision_index.process_file(f)
    assert result["status"] == "ok"
    assert result["images"] == 1

    col = ingest._db()
    all_docs = col.get(include=["documents", "metadatas"])
    image_docs = [d for d in all_docs["documents"] if "[Figure on page" in d]
    assert image_docs
    assert any("block diagram" in d for d in image_docs)


@pytest.mark.asyncio
async def test_process_file_replaces_stale_image_chunks(tmp_path, mock_ollama, monkeypatch):
    """Re-running without skip_existing replaces old image chunks with fresh ones."""
    data = b"Document to re-analyze."
    f = tmp_path / "rerun.pdf"
    f.write_bytes(data)

    monkeypatch.setattr(ingest, "extract_pages",
                        lambda d, fn: [(1, "Document to re-analyze.")])
    await ingest.ingest(data, "rerun.pdf")

    doc_id = _doc_id(data)
    col = ingest._db()
    col.add(
        ids=[f"{doc_id}_1_-100"],
        embeddings=[[0.1] * 768],
        documents=["[rerun.pdf] [Figure on page 1]: Old description."],
        metadatas=[{"doc_id": doc_id, "filename": "rerun.pdf",
                    "page": 1, "folder": "General",
                    "access": "public", "owner": "", "chunk_type": "image"}],
    )

    monkeypatch.setattr(config, "VISION_MODEL", "moondream")
    monkeypatch.setattr(ingest, "extract_images",
                        lambda d, fn: [(1, 0, b"x" * config.MIN_IMAGE_BYTES)])
    monkeypatch.setattr(ingest, "_analyze_images",
                        AsyncMock(return_value=[(1, 0, "New updated description.")]))

    result = await vision_index.process_file(f, skip_existing=False)
    assert result["status"] == "ok"

    all_docs = col.get(include=["documents"])
    image_docs = [d for d in all_docs["documents"] if "[Figure on page" in d]
    assert not any("Old description" in d for d in image_docs)
    assert any("New updated description" in d for d in image_docs)


# ── run ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_empty_dir(tmp_path, capsys):
    """run() prints a message and returns [] when no supported files are found."""
    results = await vision_index.run(tmp_path, "moondream", False, 0)
    assert results == []
    assert "No PDF/PPTX/DOCX files found" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_respects_limit(tmp_path, mock_ollama, monkeypatch):
    """run() processes at most --limit files."""
    for i in range(5):
        (tmp_path / f"doc{i}.pdf").write_bytes(b"content")

    monkeypatch.setattr(vision_index, "process_file",
                        AsyncMock(return_value={"file": "x", "status": "no_images", "images": 0}))

    results = await vision_index.run(tmp_path, "moondream", False, limit=2)
    assert len(results) == 2
