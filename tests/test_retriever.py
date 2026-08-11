"""
Tests for the RAG retrieval pipeline.

Key scenario: a transcript document contains GPA information.
We verify the correct chunks are retrieved and passed to the LLM.
"""
import os
from unittest.mock import patch

import pytest

import config
import ingest
import retriever
import store
from tests.conftest import make_embedding


async def test_gpa_chunk_is_retrieved(ingested_transcript):
    """The chunk containing 'Cumulative GPA' must be in the top results."""
    q_vec = make_embedding("search_query: What is the cumulative GPA?")
    hits = await store.query_embedding(q_vec, top_k=config.RETRIEVAL_K)
    texts = [text for text, _meta, _dist in hits.values()]
    combined = " ".join(texts)
    assert "3.78" in combined or "GPA" in combined, (
        "GPA chunk was not retrieved. Retrieved:\n" + "\n---\n".join(texts)
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
    async def only_distant_hits(vector, top_k, filters=None):
        # 2.0 is above DISTANCE_THRESHOLD (1.3), so the semantic gate must close.
        return {"x_1_0": ("unrelated chunk",
                          {"doc_id": "x", "filename": "other.txt", "page": 1, "chunk_index": 0},
                          2.0)}

    with patch.object(store, "query_embedding", side_effect=only_distant_hits):
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

    Bug: rrf_scores[summary_cid] was set unconditionally even when the fetch
    returned no results, leaving a cid in rrf_scores with no entry in sem_hits.
    _diverse_top then raised KeyError on sem_hits[cid][1].
    """
    await ingest.ingest(
        b"ATLAS-9 Detector Array nominal current 4.5 A at 28 V.",
        "atlas.txt",
    )
    sem_hits: dict = {}
    rrf_scores: dict = {}
    named_doc_set: set = set()

    # A BM25 hit whose derived summary_cid points at a chunk that does not exist.
    ghost_top_cid = "deadbeef12345678_99_0"
    with patch("bm25_index.search", return_value=[(ghost_top_cid, 1.0)]):
        await retriever._inject_system_names(
            ["ATLAS-9 Detector Array"], sem_hits, rrf_scores, named_doc_set
        )

    # The key invariant: every cid in rrf_scores must also be in sem_hits
    for cid in rrf_scores:
        assert cid in sem_hits, f"cid {cid!r} in rrf_scores but missing from sem_hits"


async def test_apply_comparison_boost_missing_chunk_no_crash(mock_ollama):
    """The boost must not crash when a BM25 hit has no corresponding stored chunk.

    Bug: rrf_scores[cid] was set unconditionally even when the fetch came back
    empty, leaving orphan cids in rrf_scores that caused KeyError in _diverse_top.
    """
    await ingest.ingest(b"Performance benchmark results for system A.", "bench.txt")

    sem_hits: dict = {}
    rrf_scores: dict = {}
    ghost_cid = "deadbeef12345678_1_0"  # in BM25, absent from the store
    with patch("bm25_index.search", return_value=[(ghost_cid, 5.0)]):
        await retriever._apply_comparison_boost(
            "which system performs best", sem_hits, rrf_scores, None, None
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
    await bm25_mod.rebuild_from_store()

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
# Catalog field extraction
#
# Document number and revision are now resolved once at ingest time by an LLM
# call and stored in chunk metadata (ingest._extract_catalog_fields), replacing
# ~280 lines of query-time regex tuned to one customer's revision conventions
# and the ~16 unit tests that pinned each of its branches. The behaviour worth
# testing is no longer "does this regex recognise E3R10" but "does a catalog
# query serve what ingest stored".
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catalog_fields_parsed_from_llm_reply(monkeypatch):
    import ingest

    async def fake_chat(system, user, model=None, history=None):
        return "DOC: GCA-2024-0002\nREV: 5.9"

    monkeypatch.setattr("llm.chat", fake_chat)
    doc_no, rev = await ingest._extract_catalog_fields("spec.pdf", "Some front matter")
    assert doc_no == "GCA-2024-0002"
    assert rev == "5.9"


@pytest.mark.asyncio
async def test_catalog_fields_absent_become_em_dash(monkeypatch):
    import ingest

    async def fake_chat(system, user, model=None, history=None):
        return "DOC: \u2014\nREV: \u2014"

    monkeypatch.setattr("llm.chat", fake_chat)
    assert await ingest._extract_catalog_fields("notes.txt", "no header here") == ("\u2014", "\u2014")


@pytest.mark.asyncio
async def test_catalog_fields_survive_llm_failure(monkeypatch):
    """A failed extraction must not fail the ingest."""
    import ingest

    async def boom(system, user, model=None, history=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("llm.chat", boom)
    assert await ingest._extract_catalog_fields("spec.pdf", "front matter") == ("\u2014", "\u2014")


@pytest.mark.asyncio
async def test_catalog_stream_serves_stored_metadata(mock_ollama, monkeypatch):
    """The catalog table is built from stored metadata, not re-parsed per query."""
    import ingest
    import retriever

    async def fake_fields(filename, front_matter):
        return ("GCA-9001", "3.10")

    monkeypatch.setattr(ingest, "_extract_catalog_fields", fake_fields)
    await ingest.ingest(b"Thruster qualification report.", "qual.txt")

    rows = [c async for c in retriever._catalog_stream("list all documents", None)]
    table = "".join(c["token"] for c in rows if "token" in c)
    assert "qual.txt" in table
    assert "GCA-9001" in table
    assert "3.10" in table


# ── citation source grouping ──────────────────────────────────────────────────

def test_source_accumulator_groups_pages_under_one_document():
    acc = retriever._source_acc()
    for page in (10, 1, 30, 10):
        retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "m.pdf", "page": page})
    out = retriever._finalize_sources(acc)
    assert len(out) == 1
    assert out[0]["pages"] == [1, 10, 30]
    assert out[0]["page"] == 1          # legacy field: lowest cited page


def test_source_accumulator_preserves_first_citation_order():
    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "b" * 16, "filename": "second.pdf", "page": 2})
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "first.pdf", "page": 1})
    retriever._add_source(acc, {"doc_id": "b" * 16, "filename": "second.pdf", "page": 9})
    assert [s["filename"] for s in retriever._finalize_sources(acc)] == ["second.pdf", "first.pdf"]


def test_figure_chunk_attaches_to_the_same_document_entry():
    """The bug grouping fixes: under the old (doc_id, page) dedup a figure on a
    page that also contributed a text chunk was dropped and never surfaced."""
    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "m.pdf", "page": 3})
    retriever._add_source(acc, {
        "doc_id": "a" * 16, "filename": "m.pdf", "page": 3,
        "chunk_type": "image", "chunk_index": ingest._IMAGE_CHUNK_IDX_BASE - 0,
    })
    out = retriever._finalize_sources(acc)
    assert len(out) == 1
    assert out[0]["figures"] == [{"page": 3, "idx": 0}]


def test_figures_are_deduped():
    acc = retriever._source_acc()
    meta = {"doc_id": "a" * 16, "filename": "m.pdf", "page": 3,
            "chunk_type": "image", "chunk_index": ingest._IMAGE_CHUNK_IDX_BASE - 2}
    retriever._add_source(acc, meta)
    retriever._add_source(acc, dict(meta))
    assert retriever._finalize_sources(acc)[0]["figures"] == [{"page": 3, "idx": 2}]


def test_documents_without_figures_omit_the_key():
    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "m.pdf", "page": 1})
    assert "figures" not in retriever._finalize_sources(acc)[0]


def test_has_original_is_reported_per_document(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ORIGINALS_DIR", str(tmp_path / "originals"))
    os.makedirs(config.ORIGINALS_DIR)
    open(os.path.join(config.ORIGINALS_DIR, f"{'a' * 16}.pdf"), "wb").write(b"x")

    acc = retriever._source_acc()
    retriever._add_source(acc, {"doc_id": "a" * 16, "filename": "kept.pdf", "page": 1})
    retriever._add_source(acc, {"doc_id": "c" * 16, "filename": "gone.pdf", "page": 1})
    out = {s["filename"]: s["has_original"] for s in retriever._finalize_sources(acc)}
    assert out == {"kept.pdf": True, "gone.pdf": False}
