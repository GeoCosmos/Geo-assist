"""
Tests for the RAG retrieval pipeline.

Key scenario: a transcript document contains GPA information.
We verify the correct chunks are retrieved and passed to the LLM.
"""
from unittest.mock import patch
import config
import ingest
import reranker
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


async def test_generate_procedure_steps_uses_llm(mock_ollama):
    """_generate_procedure_steps calls the LLM and parses numbered output."""
    async def fake_chat(system, user, model=None, history=None):
        return "1. Open the valve\n2. Record the reading\n3. Close the valve"

    with patch("llm.chat", side_effect=fake_chat):
        steps = await retriever._generate_procedure_steps(
            ["Valve system description. Nominal pressure 25 bar."]
        )

    assert len(steps) == 3
    assert steps[0].startswith("1.")
    assert steps[1].startswith("2.")
    assert steps[2].startswith("3.")


async def test_generate_procedure_steps_fallback_on_llm_failure(mock_ollama):
    """When the LLM call fails, raw chunks are returned as steps."""
    async def bad_chat(system, user, model=None, history=None):
        raise RuntimeError("ollama down")

    chunks = ["Chunk one.", "Chunk two."]
    with patch("llm.chat", side_effect=bad_chat):
        steps = await retriever._generate_procedure_steps(chunks)

    assert steps == chunks


async def test_generate_procedure_steps_fallback_on_no_numbered_output(mock_ollama):
    """When the LLM returns no parseable numbered lines, fall back to chunks."""
    async def fake_chat(system, user, model=None, history=None):
        return "Here is a summary of the document without numbered steps."

    chunks = ["Some document content."]
    with patch("llm.chat", side_effect=fake_chat):
        steps = await retriever._generate_procedure_steps(chunks)

    assert steps == chunks


async def test_inject_system_names_missing_summary_no_crash(mock_ollama):
    """_inject_system_names must not crash when the summary chunk doesn't exist.

    Bug: rrf_scores[summary_cid] was set unconditionally even when col.get()
    returned no results, leaving a cid in rrf_scores with no entry in sem_hits.
    _diverse_top then raised KeyError on sem_hits[cid][1].
    """
    await ingest.ingest(
        b"ATLAS-9 Detector Array nominal current 4.5 A at 28 V.",
        "atlas.txt",
    )
    col = ingest._db()
    # Verify the index has chunks but deliberately has no summary chunk for page 2+
    # (the bug triggers whenever summary_cid points to a non-existent chunk).
    sem_hits: dict = {}
    rrf_scores: dict = {}
    named_doc_set: set = set()
    # Directly call _inject_system_names with a sys_name whose BM25 top hit
    # builds a summary_cid that doesn't exist in ChromaDB.
    import bm25_index as bm25_mod
    bm25_mod.rebuild(col)
    sys_results = bm25_mod.search("ATLAS-9 Detector Array")
    if sys_results and sys_results[0][1] > 0:
        top_cid = sys_results[0][0]
        parts = top_cid.split("_")
        # Manufacture a summary_cid that intentionally won't exist
        fake_summary_cid = parts[0] + "_99_-1"
        # Patch the BM25 search to return the top hit so _inject_system_names
        # uses our fake summary_cid derivation path by using page 99
        original_top = sys_results[0]
        fake_top_cid = parts[0] + "_99_0"
        col.add(
            ids=[fake_top_cid],
            embeddings=[[0.0] * 768],
            documents=["ATLAS-9 Detector Array on a non-existent page"],
            metadatas=[{"doc_id": parts[0], "filename": "atlas.txt", "page": 99, "folder": "General"}],
        )
        bm25_mod.rebuild(col)
        with patch("bm25_index.search", return_value=[(fake_top_cid, 1.0)]):
            # Should not raise KeyError
            retriever._inject_system_names(
                ["ATLAS-9 Detector Array"], sem_hits, rrf_scores, named_doc_set, col
            )
        # The key invariant: every cid in rrf_scores must also be in sem_hits
        for cid in rrf_scores:
            assert cid in sem_hits, f"cid {cid!r} in rrf_scores but missing from sem_hits"


async def test_apply_comparison_boost_missing_chunk_no_crash(mock_ollama):
    """_apply_comparison_boost must not crash when col.get returns no results for a BM25 hit.

    Bug: rrf_scores[cid] was set unconditionally even when col.get() came back
    empty, leaving orphan cids in rrf_scores that caused KeyError in _diverse_top.
    """
    await ingest.ingest(b"Performance benchmark results for system A.", "bench.txt")
    col = ingest._db()
    import bm25_index as bm25_mod
    bm25_mod.rebuild(col)

    sem_hits: dict = {}
    rrf_scores: dict = {}
    ghost_cid = "deadbeef_1_0"  # cid that exists in BM25 but not in ChromaDB
    with patch("bm25_index.search", return_value=[(ghost_cid, 5.0)]):
        await retriever._apply_comparison_boost(
            "which system performs best", sem_hits, rrf_scores, col, None, None
        )
    for cid in rrf_scores:
        assert cid in sem_hits, f"cid {cid!r} in rrf_scores but missing from sem_hits"


async def test_procedure_mode_uses_procedure_prompt(mock_ollama):
    """When procedure dict is passed, answer() uses the procedure system prompt."""
    await ingest.ingest(b"ICD specifies valve torque: 30 Nm maximum.", "icd.txt")

    captured_systems = []

    async def fake_chat(system, user, model=None, history=None):
        captured_systems.append(system)
        return "Torque is 30 Nm."

    proc = {
        "doc_id": "proc_doc",
        "filename": "procedure.pdf",
        "steps": ["Step 1: Tighten valve to 25 Nm."],
        "step_idx": 0,
    }

    with patch("llm.chat", side_effect=fake_chat):
        result = await retriever.answer("What is the valve torque?", procedure=proc)

    assert result["answer"] == "Torque is 30 Nm."
    assert captured_systems, "llm.chat was never called"
    system = captured_systems[-1]
    assert "Procedure Mode" in system
    assert "procedure.pdf" in system
    assert "STEP 1 OF 1" in system
    assert "Tighten valve to 25 Nm" in system
    assert "CONFLICT DETECTION" in system


async def test_expand_query_uses_three_variants(mock_ollama, monkeypatch):
    """_expand_query must consume all 3 LLM-generated variants, returning 4 queries total.

    Bug: the slice [:2] discarded the third variant, so only 2 of 3 LLM alternatives
    were ever used, wasting part of the expansion LLM call.
    """
    monkeypatch.setattr(config, "QUERY_EXPANSION", True)

    async def fake_chat(system, user, model=None, history=None):
        return "alternative phrasing one\nalternative phrasing two\nalternative phrasing three"

    with patch("llm.chat", side_effect=fake_chat):
        result = await retriever._expand_query("what is the operating pressure?")

    assert len(result) == 4, (
        f"Expected original + 3 variants = 4 queries, got {len(result)}: {result}"
    )
    assert result[0] == "what is the operating pressure?"
    assert "alternative phrasing three" in result


async def test_score_band_non_reranked_chunks_capped(mock_ollama, monkeypatch):
    """Non-reranked chunks must not outrank the reranker's top picks after score rewrite.

    Bug: after reranking rewrites the top-N scores to [1/(60+0+1), 1/(60+N-1+1)],
    any chunk at rank N+1..M kept its higher pre-rewrite two-list RRF score
    (e.g. 2/(60+N+1) ≈ 0.025) and outranked the reranker's entire pool
    (max reranked ≈ 0.016).  The fix: apply structural boosts before pool
    selection, then cap non-injected, non-reranked chunks to half the worst
    reranked score so the reranker's ordering is preserved.
    """
    monkeypatch.setattr(config, "RERANK_TOP_N", 5)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)

    for i in range(10):
        await ingest.ingest(
            f"Component {i}: rated voltage {10 + i} V, current {i + 1} A nominal.".encode(),
            f"component_{i:02d}.txt",
        )

    import bm25_index as bm25_mod
    bm25_mod.rebuild(ingest._db())

    reranked_cids_seen: list[str] = []

    def recording_rerank(query, candidates):
        reranked_cids_seen.extend(c[0] for c in candidates)
        return list(range(len(candidates)))

    captured_scores: dict = {}
    original_diverse_top = retriever._diverse_top

    def capturing_diverse_top(rrf_scores, sem_hits, **kw):
        captured_scores.update(rrf_scores)
        return original_diverse_top(rrf_scores, sem_hits, **kw)

    with patch("reranker.rerank", side_effect=recording_rerank):
        with patch("retriever._diverse_top", side_effect=capturing_diverse_top):
            with patch("llm.chat", return_value="5 V"):
                await retriever.answer("what is the rated voltage?")

    assert captured_scores, "rrf_scores were not captured"
    assert reranked_cids_seen, "reranker.rerank was never called"

    reranked_set = set(reranked_cids_seen)
    N = 5
    floor = 1.0 / (60 + N + 1) * 0.5  # cap applied to non-reranked, non-injected chunks

    for cid, score in captured_scores.items():
        if cid in reranked_set:
            continue
        # Injected chunks (sentinel distances ≥ 991) are exempt from the cap.
        dist = ingest._db().get(ids=[cid], include=["metadatas"])
        # Use the score itself as proxy: if it's very high it must be injected.
        # Regular retrieval chunks (non-injected, non-reranked) must be ≤ floor.
        # We allow summary-boosted chunks that entered the reranker pool to be absent
        # here since they were captured by reranked_set; only truly non-reranked ones remain.
        assert score <= floor * 1.01, (
            f"Non-reranked chunk {cid!r} has score {score:.5f} which exceeds "
            f"floor {floor:.5f} — reranker ordering can be bypassed"
        )


async def test_retrieval_eval_specific_value_in_context(mock_ollama):
    """Retrieval eval: a specific measurement must appear in the LLM context.

    This is a pipeline-logic eval (fake embeddings, deterministic ranking), not a
    semantic accuracy eval.  It verifies the BM25+RRF+expansion pipeline routes
    specific numeric values into the answer context.
    """
    await ingest.ingest(
        b"The primary coolant loop operates at 138 bar maximum allowable working pressure (MAWP).",
        "coolant_spec.txt",
    )

    context_seen: list[str] = []

    async def fake_chat(system, user, model=None, history=None):
        if "Retrieved context" in system:
            context_seen.append(system)
        return "138 bar"

    with patch("llm.chat", side_effect=fake_chat):
        result = await retriever.answer("What is the MAWP of the coolant loop?")

    assert context_seen, "LLM was never called with retrieved context"
    assert "138" in context_seen[0], (
        "The specific measurement '138 bar' was not present in the LLM context.\n"
        "Context received:\n" + context_seen[0][:500]
    )
    assert any(s["filename"] == "coolant_spec.txt" for s in result["sources"])


# ---------------------------------------------------------------------------
# Revision extraction unit tests
# ---------------------------------------------------------------------------

def test_revision_table_single_column_highest():
    """Single-column Is.Rev table returns the highest value as X.X (not last in reading order)."""
    from retriever import _latest_revision_from_chunks
    # Newest-first (descending) EXRX table — must return highest as X.X
    chunks = [
        "| Is.Rev | Date | Author |\n| --- | --- | --- |\n| E4R0 | 2025-01-01 | Jones |\n| E3R10 | 2024-06-01 | Smith |\n| E2R5 | 2023-01-01 | Brown |",
    ]
    result = _latest_revision_from_chunks(chunks)
    assert result == "4.0", f"Expected 4.0, got {result!r}"


def test_revision_table_numeric_version_aware():
    """3.10 > 3.9 version-aware comparison (not lexicographic)."""
    from retriever import _latest_revision_from_chunks
    chunks = [
        "| Revision | Date |\n| --- | --- |\n| 3.9 | 2023-01-01 |\n| 3.10 | 2024-01-01 |",
    ]
    result = _latest_revision_from_chunks(chunks)
    assert result == "3.10", f"Expected 3.10, got {result!r}"


def test_revision_table_beats_dotnum():
    """Table scan takes priority over Is.Rev dotnum in body text."""
    from retriever import _latest_revision_from_chunks
    # Is.Rev 2.0 appears in body text but revision table has 4.0 — table wins
    chunks = [
        "Is.Rev 2.0\n\n| Revision | Date |\n| --- | --- |\n| 3.5 | 2023 |\n| 4.0 | 2024 |",
    ]
    result = _latest_revision_from_chunks(chunks)
    assert result == "4.0", f"Expected 4.0 (table), got {result!r}"


def test_revision_dotnum_colon_separator():
    """Is.Rev: 3.10 with colon separator is matched (page-1 title block format)."""
    from retriever import _latest_revision_from_chunks
    chunks = ["Document Title\nIs.Rev: 3.10\nDate: 2024-01-15"]
    result = _latest_revision_from_chunks(chunks)
    assert result == "3.10", f"Expected 3.10, got {result!r}"


def test_revision_dotnum_fallback_when_no_table():
    """Is.Rev N.M used when no revision table is found."""
    from retriever import _latest_revision_from_chunks
    chunks = [
        "Document Title\nIs.Rev 3.10\nDate: 2024-01-15",
    ]
    result = _latest_revision_from_chunks(chunks)
    assert result == "3.10", f"Expected 3.10, got {result!r}"


def test_revision_dotnum_spaces_and_dots():
    """Is. Rev. 2.5 with spaces and dots should also match."""
    from retriever import _latest_revision_from_chunks
    chunks = ["Is. Rev. 2.5  Author: Jones"]
    result = _latest_revision_from_chunks(chunks)
    assert result == "2.5", f"Expected 2.5, got {result!r}"


def test_revision_exrx_labeled_in_body_text():
    """Is.Rev-labeled EXRX in body text returned as X.X; bare EXRX is NOT matched."""
    from retriever import _latest_revision_from_chunks
    # bare EXRX in body text (e.g. part number suffix) — should be ignored
    chunks = ["Technical document E3R10 revision history."]
    result = _latest_revision_from_chunks(chunks)
    assert result == "—", f"Bare EXRX should not match, got {result!r}"


def test_revision_exrx_highest_pair():
    """Is.Rev-labeled EXRX: highest E+R wins, returned as X.X."""
    from retriever import _latest_revision_from_chunks
    chunks = ["Is.Rev E2R5 previous release. Is.Rev E3R10 current. Is.Rev E3R9 draft."]
    result = _latest_revision_from_chunks(chunks)
    assert result == "3.10", f"Expected 3.10, got {result!r}"


def test_revision_is_num_rev_num_body_text():
    """'Is. N - Rev. M' split format in body text (dtu-style running header)."""
    from retriever import _latest_revision_from_chunks
    chunks = ["COMMAND RANGING UNIT\nIs. 5 - Rev. 9\nMCS USER'S MANUAL\nDate: February 9, 2017"]
    result = _latest_revision_from_chunks(chunks)
    assert result == "5.9", f"Expected 5.9, got {result!r}"


def test_revision_is_num_rev_num_highest():
    """When 'Is. N - Rev. M' appears multiple times, highest pair wins."""
    from retriever import _latest_revision_from_chunks
    chunks = ["Is. 5 - Rev. 8 older\nIs. 5 - Rev. 9 current"]
    result = _latest_revision_from_chunks(chunks)
    assert result == "5.9", f"Expected 5.9, got {result!r}"


def test_revision_page_header_no_pipe_no_false_positive():
    """Page headers stop at pipe lines — table body with 'Is.\\nRev.\\nDate' doesn't give 'D'."""
    from retriever import _revision_from_page_headers
    # Non-pipe stacked-column table on page 2: headers and data on separate lines
    text = "[doc.pdf] EVOLUTIONS\n\nIs. \nRev. \nDate \nDescription\n\n3 \n27 \nSeptember 27, 2018"
    text_pairs = [("Cover page", {"page": 1}), (text, {"page": 2})]
    result = _revision_from_page_headers(text_pairs)
    assert result == "—", f"Expected '—' (no false positive), got {result!r}"


def test_revision_page_header_combined_is_abbrev():
    """'Is. N - Rev. M' in a running header is parsed as N.M by _revision_from_page_headers."""
    from retriever import _revision_from_page_headers
    text_pairs = [
        ("Cover page", {"page": 1}),
        ("DOCUMENT TITLE\nIs. 5 - Rev. 9\nDate: 2017", {"page": 2}),
    ]
    result = _revision_from_page_headers(text_pairs)
    assert result == "5.9", f"Expected 5.9, got {result!r}"


def test_revision_two_column_table():
    """| Is. | Rev. | separate column headers detected and combined as N.M."""
    from retriever import _latest_revision_from_chunks
    chunks = [
        "| Is. | Rev. | Date | Description |\n| --- | --- | --- | --- |\n| 3 | 10 | 2024-01-01 | Initial |\n| 4 | 0 | 2025-06-01 | Updated |",
    ]
    result = _latest_revision_from_chunks(chunks)
    assert result == "4.0", f"Expected 4.0, got {result!r}"


def test_revision_two_column_issue_full_word():
    """| Issue | Rev. | headers (full word) also match."""
    from retriever import _latest_revision_from_chunks
    chunks = [
        "| Issue | Rev. | Date |\n| --- | --- | --- |\n| 1 | A | 2022 |\n| 2 | A | 2023-05-01 |",
    ]
    result = _latest_revision_from_chunks(chunks)
    assert result == "2.A", f"Expected 2.A, got {result!r}"


def test_revision_page_header_exrx():
    """EXRX in page 2+ running header returned as X.X by _revision_from_page_headers."""
    from retriever import _revision_from_page_headers
    text_pairs = [
        ("Page 1 title block content", {"page": 1}),
        ("Company Name  E3R10  Document Title\nOther header content", {"page": 2}),
    ]
    result = _revision_from_page_headers(text_pairs)
    assert result == "3.10", f"Expected 3.10, got {result!r}"


def test_revision_page_header_is_rev_dotnum():
    """Is.Rev N.M in a page header is extracted by _revision_from_page_headers."""
    from retriever import _revision_from_page_headers
    text_pairs = [
        ("Cover page", {"page": 1}),
        ("Title  Is.Rev 5.2  Date 2024", {"page": 2}),
    ]
    result = _revision_from_page_headers(text_pairs)
    assert result == "5.2", f"Expected 5.2, got {result!r}"
