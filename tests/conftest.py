"""
Shared fixtures. Ollama is mocked so tests run without a real GPU/server.

Each test gets its own Qdrant store in a unique tmp_path. Tests use the Qdrant
client's *local* mode (`path=...`), which needs no running server. That mode is
brute-force and capped at roughly 20k points, which is unsuitable for the real
real corpus — production uses the server, see store.py — but it is exactly
right for a test fixture holding a handful of documents, and it exercises the same
client API surface.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pytest_asyncio
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

import bm25_index
import config
import ingest
import store


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    """Give every test an isolated Qdrant store, BM25 index, and data directories."""
    monkeypatch.setattr(config, "BM25_PATH", str(tmp_path / "bm25_index.pkl"))
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "originals"))
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    # Parsing in a subprocess pool would not see these monkeypatched paths, and
    # spawning processes per test is slow. Run parse work inline instead.
    monkeypatch.setattr(ingest, "_pool", lambda: None)

    qdrant = QdrantDocumentStore(
        path=str(tmp_path / "qdrant"),
        index="test_docs",
        embedding_dim=768,
        similarity="cosine",
        recreate_index=True,
        progress_bar=False,
    )
    store.reset_store(qdrant)
    ingest._doc_cache = None
    bm25_index._index.build([], [])   # reset BM25 between tests
    yield qdrant
    store.reset_store(None)
    ingest._doc_cache = None
    bm25_index._index.build([], [])


def make_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    import hashlib
    import random
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x ** 2 for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest.fixture
def mock_ollama(monkeypatch):
    """Patch llm.embed and llm.chat with deterministic fakes."""
    import llm

    async def fake_embed(texts, prefix="search_document"):
        return [make_embedding(f"{prefix}: {t}") for t in texts]

    async def fake_chat(system, user, model=None, history=None):
        return f"MOCK_ANSWER for: {user}"

    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(llm, "chat", fake_chat)


@pytest_asyncio.fixture
async def ingested_transcript(mock_ollama):
    """Ingest a known transcript document before each retriever test."""
    transcript = """\
Student Academic Transcript
Name: Alex Student  Student ID: 12345678

Semester 1:
  MATH 101 - Calculus I          Grade: A    Credits: 3
  PHYS 101 - Physics I           Grade: B+   Credits: 4
  ENGL 101 - English Composition Grade: A-   Credits: 3

Semester 2:
  MATH 201 - Calculus II         Grade: A    Credits: 3
  PHYS 201 - Physics II          Grade: A-   Credits: 4
  COMP 101 - Intro to CS         Grade: A    Credits: 3

Cumulative GPA: 3.78
Total Credits Earned: 20
"""
    await ingest.ingest(transcript.encode(), "transcript.txt")
