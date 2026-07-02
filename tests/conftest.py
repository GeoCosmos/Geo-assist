"""
Shared fixtures. Ollama is mocked so tests run without a real GPU/server.
Each test gets its own PersistentClient in a unique tmp_path for true isolation
(EphemeralClient shares in-process state across tests).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
import pytest
import pytest_asyncio
import bm25_index
import config
import ingest


@pytest.fixture(autouse=True)
def fresh_chroma(tmp_path, monkeypatch):
    """Give every test its own isolated ChromaDB and BM25 path via a temp directory."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
    monkeypatch.setattr(ingest, "_chroma_client", client)
    monkeypatch.setattr(ingest, "_chroma", col)
    monkeypatch.setattr(ingest, "_doc_cache", None)
    monkeypatch.setattr(config, "BM25_PATH", str(tmp_path / "bm25_index.pkl"))
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "originals"))
    bm25_index._index.build([], [])   # reset BM25 between tests
    yield col
    ingest._chroma_client = None
    ingest._chroma = None
    ingest._doc_cache = None
    bm25_index._index.build([], [])
    client._system.stop()
    client.clear_system_cache()


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
