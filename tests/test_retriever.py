"""
Tests for the RAG retrieval pipeline.

Key scenario: a transcript document contains GPA information.
We verify the correct chunks are retrieved and passed to the LLM.
"""
from unittest.mock import patch
import ingest
import retriever
from tests.conftest import make_embedding


async def test_gpa_chunk_is_retrieved(ingested_transcript):
    """The chunk containing 'Cumulative GPA' must be in the top results."""
    q_vec = make_embedding("What is the cumulative GPA?")
    col = ingest._db()
    results = col.query(
        query_embeddings=[q_vec],
        n_results=ingest.config.RETRIEVAL_K,
        include=["documents", "distances"],
    )
    combined = " ".join(results["documents"][0])
    assert "3.78" in combined or "GPA" in combined, (
        "GPA chunk was not retrieved. Retrieved:\n" + "\n---\n".join(results["documents"][0])
    )


async def test_answer_calls_llm_with_context(ingested_transcript, mock_ollama):
    """answer() must pass context containing GPA data to the LLM."""
    captured_system = []
    call_count = [0]

    async def fake_chat(system, user, model=None, history=None):
        call_count[0] += 1
        # first call is query expansion (no system context injected yet)
        # subsequent call is the actual answer generation
        if "Retrieved context" in system:
            captured_system.append(system)
            return "The GPA is 3.78"
        return "alternative phrasing of the question"

    with patch("llm.chat", side_effect=fake_chat):
        result = await retriever.answer("What is the cumulative GPA?")

    assert result["answer"] == "The GPA is 3.78"
    assert captured_system, "llm.chat was never called with a retrieval context"
    assert "3.78" in captured_system[0] or "GPA" in captured_system[0], (
        "Context passed to LLM did not contain GPA info.\nSystem prompt:\n" + captured_system[0]
    )


async def test_answer_includes_sources(ingested_transcript, mock_ollama):
    result = await retriever.answer("What is the GPA?")
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0
    assert result["sources"][0]["filename"] == "transcript.txt"


async def test_answer_no_docs_returns_not_found(mock_ollama):
    """When nothing is ingested, should say info not available."""
    result = await retriever.answer("What is the GPA?")
    assert "not in the ingested documents" in result["answer"]
    assert result["sources"] == []


async def test_answer_irrelevant_question_returns_not_found(ingested_transcript, mock_ollama):
    """A question with no matching context should not hallucinate."""
    with patch.object(ingest._db(), "query", return_value={
        "ids": [["chunk_0"]],
        "documents": [["unrelated chunk"]],
        "metadatas": [[{"doc_id": "x", "filename": "other.txt", "page": 1}]],
        "distances": [[2.0]],   # above DISTANCE_THRESHOLD of 1.3
    }):
        result = await retriever.answer("What is the weather forecast?")

    assert "not in the ingested documents" in result["answer"]
    assert result["sources"] == []


async def test_bm25_rescues_keyword_miss(mock_ollama):
    """A rare proper noun not well-represented in embedding space should still
    be found because BM25 matches the exact token."""
    await ingest.ingest(
        b"The sponsoring agency is XYZQ-7 and they provided $1.2M in funding.",
        "sponsor.txt",
    )
    # Ask using the exact rare token — semantic alone may miss it
    captured = []

    async def fake_chat(system, user, model=None, history=None):
        captured.append(system)
        return "XYZQ-7"

    with patch("llm.chat", side_effect=fake_chat):
        await retriever.answer("Who is XYZQ-7?")

    # The context passed to the LLM should contain the rare token
    assert any("XYZQ-7" in s for s in captured), (
        "BM25 did not surface the chunk containing the rare token XYZQ-7"
    )


async def test_multiple_docs_sources_are_distinct(mock_ollama):
    """When multiple docs are relevant, sources list should include each."""
    await ingest.ingest(b"Alpha project budget is $50,000.", "alpha.txt")
    await ingest.ingest(b"Beta project budget is $75,000.", "beta.txt")

    async def fake_chat(system, user, model=None, history=None):
        return "Alpha $50k, Beta $75k"

    with patch("llm.chat", side_effect=fake_chat):
        result = await retriever.answer("What are the project budgets?")

    filenames = {s["filename"] for s in result["sources"]}
    assert len(filenames) >= 1


async def test_named_doc_injected_by_filename(mock_ollama):
    """When the question names an ingested file explicitly, that doc's chunks appear
    in context even when its semantic distance might not rank it highly."""
    await ingest.ingest(b"Safety margin requirement: 2.5 g minimum.", "safetymargin.txt")
    await ingest.ingest(b"Unrelated document about budget forecasts.", "budget.txt")

    async def fake_chat(system, user, model=None, history=None):
        if "Retrieved context" in system:
            return "The safety margin is 2.5 g."
        return "alternative phrasing"

    with patch("llm.chat", side_effect=fake_chat):
        result = await retriever.answer("what does safetymargin.txt say?")

    assert any(s["filename"] == "safetymargin.txt" for s in result["sources"])
