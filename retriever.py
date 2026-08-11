import asyncio
import logging
import re

import bm25_index
import config
import ingest
import llm
import reranker
import store

# Matches named subsystems like "PULSE-3 Detector Array" or "ZENITH-3 Focal Plane Module".
# Requires at least one Title-Case word after the prefix-number so bare doc prefixes and
# standalone acronyms don't match.
_SYS_REF = re.compile(r'\b([A-Z]+-\d+(?:\s+[A-Z][A-Za-z]*){1,3})\b')

# Queries seeking a specific measured or specified value — triggers BM25 boosting
# so exact token matching dominates over semantic similarity.
_VALUE_QUERY_RE = re.compile(
    r'\b('
    r'pressure|temperature|voltage|current|power|watt|mass|weight|'
    r'torque|force|flow|frequency|speed|velocity|thrust|impulse|acceleration|'
    r'dimension|thickness|diameter|radius|height|width|length|depth|area|volume|'
    r'resistance|impedance|capacitance|inductance|gain|loss|efficiency|'
    r'spec(?:ification)?s?|parameter|rating|rated|nominal|'
    r'maximum|minimum|max\b|min\b|tolerance|limit|threshold|range|margin|'
    r'measurement|reading|value\s+of|how\s+(?:much|many)'
    r')\b',
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r'\b(top.perform|best|highest|most efficient|lowest|least massive|'
    r'rank|leader|which .{0,20}(best|highest|top)|among .{0,40}documents?)\b',
    re.IGNORECASE,
)

# Queries that ask for a cross-document table / inventory of all files.
# Normal RRF (k=8, max_per_file=2) can only surface 4 files at once — catalog
# mode bypasses that cap and fetches one representative chunk per document.
_CATALOG_RE = re.compile(
    r'(?:table|list|overview|summary|inventory|index)\s+of\s+(?:all\s+)?(?:files?|documents?)|'
    r'\b(?:create|make|generate|show|give|build|display|produce)\b.{0,40}\b(?:table|list)\b.{0,50}\b(?:documents?|files?|revisions?)\b|'
    r'\b(?:all|each|every)\s+(?:files?|documents?)\b|'
    r'\bfile\s*names?\s*(?:and|with|,)|'
    r'\bdocument\s+(?:inventory|index|catalog|register)\b|'
    r'\b(?:all|each|every|list)\b.{0,50}\b(?:revision|version|rev)\b',
    re.IGNORECASE,
)

# ── catalog / inventory mode ──────────────────────────────────────────────────
#
# Document number and revision are resolved once at ingest time (see
# ingest._extract_catalog_fields) and stored in chunk metadata. Catalog queries
# are therefore a metadata read.
#
# This replaces ~280 lines of regex that tried to recognise revision conventions
# at query time — "Is. 3 - Rev. 2", "Is.Rev 3.10", "E3R10", French "Éd./Rév.",
# two-column pipe tables, running page headers — with priority ordering, tight
# and loose modes, and a version-aware sort to pick the highest. It was tuned to
# one customer's document set, could not be validated by inspection, ran on every
# catalog query for every document, and silently produced an em dash for any
# convention it had not been taught. Extraction now happens once per document,
# is fixable by re-ingesting a single file, and costs nothing at query time.


async def _catalog_stream(question: str, folder_filter: str | None):
    """Stream a File / Document No / Revision table over the ingested corpus."""
    docs = await ingest.list_documents()
    if folder_filter:
        docs = [d for d in docs if d.get("folder") == folder_filter]
    if not docs:
        yield {"token": _NOT_FOUND}
        yield {"sources": [], "done": True}
        return

    yield {"token": "| File | Document No | Revision |\n| --- | --- | --- |\n"}

    source_acc = _source_acc()
    for doc in sorted(docs, key=lambda d: d["filename"].lower()):
        yield {
            "token": "| {} | {} | {} |\n".format(
                doc["filename"],
                doc.get("doc_number") or "—",
                doc.get("revision") or "—",
            )
        }
        _add_source(source_acc, {"filename": doc["filename"], "page": 1, "doc_id": doc["doc_id"]})

    yield {"sources": _finalize_sources(source_acc), "done": True}

log = logging.getLogger(__name__)


def _source_acc() -> dict:
    """Accumulator for citation sources, keyed by doc_id.

    One entry per document rather than per (doc_id, page). A document cited on
    three pages previously produced three separate chips, and — more importantly
    — a figure on a page that had also contributed a text chunk was dropped by
    the old dedup, so it could never be displayed.

    Python dicts preserve insertion order, so first-citation order (most
    relevant document first) is retained for free.
    """
    return {}


def _add_source(acc: dict, meta: dict) -> None:
    """Record one retrieved chunk's provenance against its document."""
    doc_id = meta["doc_id"]
    entry = acc.get(doc_id)
    if entry is None:
        entry = {"filename": meta["filename"], "doc_id": doc_id, "pages": [], "figures": []}
        acc[doc_id] = entry

    page = meta.get("page", 1)
    if page not in entry["pages"]:
        entry["pages"].append(page)

    if meta.get("chunk_type") == "image":
        ref = {"page": page, "idx": ingest.figure_index(meta.get("chunk_index", 0))}
        if ref not in entry["figures"]:
            entry["figures"].append(ref)


def _finalize_sources(acc: dict) -> list[dict]:
    """Turn the accumulator into the payload sent to the client.

    Reads the originals directory once for the whole answer. The catalog path
    calls this with every document in the corpus, so a per-document glob would
    be O(docs x files) of synchronous stat work on the event loop, mid-stream.
    """
    originals = ingest.originals_index()
    out = []
    for entry in acc.values():
        pages = sorted(entry["pages"])
        source = {
            "filename": entry["filename"],
            "doc_id": entry["doc_id"],
            "pages": pages,
            # Retained so anything reading the pre-grouping field keeps working,
            # including conversations replayed from localStorage.
            "page": pages[0] if pages else 1,
            "has_original": entry["doc_id"] in originals,
        }
        if entry["figures"]:
            source["figures"] = entry["figures"]
        out.append(source)
    return out

_RRF_K = 60
_EXPAND_WINDOW = 2  # neighbor chunks fetched on each side for parent-chunk retrieval

# Table completeness bounds. A table_id covering more chunks than _TABLE_MAX_CHUNKS
# is a table-detection false positive, not a table — DOCX/PDF extraction happily
# tags long runs of prose. Expanding one drags a large slice of the document into
# the prompt: measured live at ~150 context chunks and 13,480 prompt tokens, which
# cost 573s of prompt evaluation on CPU before the model emitted a single token.
# _TABLE_EXPANSION_BUDGET bounds the total across all tables in one query, since
# several individually-plausible tables can still add up.
_TABLE_MAX_CHUNKS = 24
_TABLE_EXPANSION_BUDGET = 32


_NAMED_DOC_MIN_STEM = 6


async def _find_named_docs(question: str) -> list[str]:
    """Return filenames of ingested documents named verbatim in the question.

    Uses the in-memory doc cache — no extra DB round-trip per query.

    Matching is word-bounded and requires a longer stem than it used to. The old
    rule was a bare substring test with a 4-character floor, which meant a file
    called `data.pdf` or `test.docx` matched any question containing "data" or
    "test" — and a matched document is then granted *every* context slot by
    `_diverse_top`. One generically named file could hijack the entire context
    window for unrelated questions.
    """
    q_lower = question.lower()
    named = []
    for doc in await ingest.list_documents():
        fn = doc["filename"]
        fn_lower = fn.lower()
        stem_lower = fn_lower.rsplit(".", 1)[0] if "." in fn_lower else fn_lower
        if fn_lower in q_lower or len(stem_lower) >= _NAMED_DOC_MIN_STEM and re.search(
            rf"(?<!\w){re.escape(stem_lower)}(?!\w)", q_lower
        ):
            named.append(fn)
    return named


_SUMMARY_BOOST = 3.0
# Sentinel distance values assigned by injection helpers to force-added chunks.
# These chunks must survive the post-rerank cap — they carry pipeline guarantees
# (named-doc, sys-name, comparison-boost) that must not be silenced by the reranker.
_INJECTED_SENTINELS = frozenset({991.0, 994.0, 997.0, 998.0})

_PROCEDURE_EXTRACT_SYSTEM = """\
You are a technical procedure specialist. Given the content of an engineering document, \
extract or synthesize a clear numbered step-by-step procedure that an engineer can execute \
in the field.

Rules:
- Number steps starting at 1
- Each step is one concrete, executable action
- Include specific values, tolerances, part numbers, or conditions from the document
- If the document already has explicit numbered steps, preserve and clarify them
- If the document is a specification, manual, or description without explicit steps, \
  infer the correct execution sequence from the content — think about what an engineer \
  would actually do in order
- Do not invent steps or values not supported by the document
- Format each step as: "N. [action]" — one step per line, no sub-bullets, no headings
- Aim for 5–20 steps; combine trivially short steps and split compound actions
- Respond with ONLY the numbered steps — no preamble, no closing remarks
"""

_STEP_LINE_RE = re.compile(r'^\d+\.\s+.+$')


async def _generate_procedure_steps(chunks: list[str]) -> list[str]:
    """Call the LLM to extract or synthesize a numbered step list from document chunks.

    Falls back to raw chunks if the LLM call fails or returns no parseable steps.
    Caps input at 20 chunks to stay within the model's context window.
    """
    content = "\n\n".join(chunks[:20])
    try:
        raw = await llm.chat(
            system=_PROCEDURE_EXTRACT_SYSTEM,
            user=f"Document content:\n\n{content}",
        )
        steps = [ln.strip() for ln in raw.splitlines() if _STEP_LINE_RE.match(ln.strip())]
        if steps:
            return steps
    except Exception:
        log.warning("procedure step generation failed", exc_info=True)
    return [c.strip() for c in chunks if c.strip()]

# Included in the system prompt only when OCR is enabled, so the LLM is not
# primed to look for figure annotations that will never appear.
_FIGURE_NOTE = (
    "Some blocks begin with \"[Figure on page N]:\" — these are OCR-extracted text "
    "from images found on that document page. Treat them as you would textual content: "
    "cite the source file and page when referencing them, and flag any uncertainty if the "
    "extracted text seems garbled or incomplete.\n\n"
) if config.OCR_ENABLED else ""

_SYSTEM = """\
You are Geo-Assist, a technical document Q&A assistant in a secure, air-gapped
environment. You answer only from the retrieved document excerpts at the bottom of
this prompt. An engineer may act on your answer, so a wrong number is worse than
no number.

═══ GROUNDING (highest priority) ═══
G1. Never produce a filename, document number, reference number, part number,
    revision, or numeric value that does not appear verbatim in the retrieved
    context. No approximation, no rounding, no extrapolation, no training
    knowledge.
G2. You may cite only the documents listed under "Available sources" below. Any
    other document name does not exist in this system.
G3. Attribute every claim to the block it came from. Each block is labelled with
    its source file and page — that label, not proximity or plausibility, decides
    which document the text belongs to. Never move a fact from one block's
    document to another's.
G4. If the retrieved context does not support an answer, reply exactly:
    "That information is not in the ingested documents."
    Use this exact sentence every time — for missing documents, missing values,
    and missing sections alike. Do not soften it, and do not append a guess.
G5. Numbers and units are copied, never converted or localized. Keep the original
    decimal separator, unit, and precision as written (do not turn 1.5 mm into
    1,5 mm or 0.059 in; do not write 1200 as 1.2k).

═══ BEFORE YOU ANSWER ═══
Begin every response with a short "Sources used:" line listing the source file and
page label of each block you will draw on — copied from the block labels, not
recalled. If no block is relevant, that line is empty and your answer is the
sentence in G4. Then write the answer. Nothing may appear in the answer that is not
traceable to a block you listed.

═══ CONFLICTS ═══
C1. If two or more blocks give different values for what appears to be the same
    fact, never average, blend, or silently pick one. State that the value differs
    by document, list each document with its value, and say which one (if any)
    applies to the product the question is about.
C2. This applies within a single document as well as across documents.
C3. Boilerplate specs — weight limits, temperature ranges, voltages, torques —
    are the highest-risk case, because near-identical manuals cover different
    product variants. Check these against every retrieved block before answering.
C4. When in doubt about whether two statements conflict, flag it. A false flag
    costs the engineer thirty seconds; a missed one costs more.

═══ ANSWERING ═══
A1. Cite the source file and page in parentheses after each key factual claim.
A2. For a specific value, name, or number: quote or closely paraphrase the passage
    that supports it, so the engineer can find it in the document.
A3. If you reason beyond what a block states, label it: "Inference:" followed by
    the stated facts it rests on. Never let an inference read as a quotation.
A4. Nominal value plus tolerance: give both.
A5. Procedures: include every explicit timing value, in order.
A6. Totals or combined values: show the arithmetic step by step, naming the source
    document of each input number.
A7. Recommendations: ground them in the measurements, test results, margins, and
    findings in the retrieved blocks. If the context contains a "Recommendations"
    section, you may cite it, but do not simply restate it — say what evidence
    supports it, and flag it if the measurements do not.
A8. When the question names a specific document, use only blocks labelled with
    that document. If none were retrieved, reply with G4 rather than answering
    from a different document.

═══ FORMAT ═══
F1. Comparisons, contrasts, rankings, and tabulations: use a markdown table, one
    row per item, one column per attribute, with a short prose summary after it if
    the comparison needs context. Everything else: plain prose.
F2. Answer in the language of the question — Russian question, Russian answer;
    Armenian question, Armenian answer. Retrieved context may be in English;
    translate the prose. Give uncertain technical terms with the English term in
    parentheses. When answering in another language, use that language's direct
    equivalent of the G4 sentence.
F3. Per G5, numbers, units, and identifiers stay exactly as written in the source,
    in every language.

Available sources (the ONLY documents you may cite):
{source_list}

{figure_note}Retrieved context:
{context}

═══ REMEMBER ═══
Two rules override any other consideration, including helpfulness and completeness:
never write a number or identifier you cannot point to in a block above, and never
resolve a contradiction silently. If you cannot ground the answer, the correct
answer is "That information is not in the ingested documents."
"""
_PROCEDURE_SYSTEM = """\
You are Geo-Assist in Procedure Mode. An engineer is executing a procedure step by
step, in the field, and may ask for clarification or ask you to verify the step
against reference documents. Assume they will act on what you say.

CURRENT PROCEDURE: {filename}
STEP {step_num} OF {total_steps}
──────────────────────────────────────
{step_text}
──────────────────────────────────────

═══ GROUNDING (highest priority) ═══
G1. Never produce a filename, document number, part number, revision, or numeric
    value that is not present verbatim in the step text above or in the retrieved
    context below.
G2. You may cite only the documents listed under "Available sources".
G3. Cite the source file and page in parentheses after each factual claim. Facts
    from the step itself are cited as (procedure, step {step_num}).
G4. If the retrieved context does not contain what is needed, reply exactly:
    "That information is not in the ingested documents."
G5. Copy numbers and units exactly as written — no conversion, no rounding, no
    localization.

═══ CONFLICT DETECTION (run this first, every turn) ═══
Before answering, take every numeric value in the current step — pressure, torque,
temperature, voltage, timing, clearance, quantity — and look for the same parameter
in the retrieved context. If a reference document states a different value, or an
instruction that contradicts the step, emit this immediately, before your answer:

⚠️ CONFLICT: Procedure says [procedure claim] but [reference source] (page N) says
[reference claim]. Verify before proceeding.

One line per conflict. Flag borderline cases too — if you are unsure whether two
statements genuinely conflict, emit the warning and say what is ambiguous. If the
retrieved context contradicts itself, flag each contradiction the same way.
Never resolve a conflict on the engineer's behalf by choosing a value.

═══ ANSWERING ═══
A1. Help the engineer understand and execute the current step, using the step text
    and the retrieved references.
A2. Include every explicit timing value in the step, in order.
A3. Nominal value plus tolerance: give both.
A4. Label reasoning that goes beyond the text as "Inference:" and name the facts
    it rests on.
A5. Answer in the language the engineer used; keep numbers, units, and identifiers
    unchanged.

Available sources (the ONLY documents you may cite):
{source_list}

{figure_note}Retrieved reference documents:
{context}

═══ REMEMBER ═══
Never invent a value. Never suppress a possible conflict. If the answer is not in
the step text or the retrieved context, say "That information is not in the
ingested documents."
"""

_EXPAND_SYSTEM = """\
You are a search query optimizer for a technical document retrieval system serving
an air-gapped engineering corpus.

Given a question, produce 3 alternative search queries that would retrieve the most
relevant passages. Make each meaningfully different:

- Query 1 — literal: the exact technical terms, values, units, parameter names, and
  table headers likely to appear verbatim in the documents.
- Query 2 — evaluative: the trade-offs, comparisons, margins, methodology, test
  results, or design rationale that would surface analytical passages.
- Query 3 — lexical variant: the same intent in different vocabulary — synonyms,
  domain phrasing, or the wording a different section would use (summary vs.
  detail, spec table vs. narrative).

Rules:
- Copy part numbers, document numbers, model designations, units, and numeric
  values verbatim into every query where they are relevant. Never paraphrase,
  expand, normalize, or reformat an identifier — "P/N 4471-B" stays "P/N 4471-B".
- Do not introduce technical terms, part numbers, or values that are not in the
  question or a plain synonym of something in it.
- Keep each query under about 20 words.

Output only the 3 queries, one per line. No numbering, no explanation.
"""


async def _expand_query(question: str) -> list[str]:
    try:
        raw = await llm.chat(system=_EXPAND_SYSTEM, user=question, model=config.EXPAND_MODEL)
        variants = [q.strip() for q in raw.splitlines() if q.strip()][:3]
    except Exception:
        log.warning("query expansion failed", exc_info=True)
        variants = []
    return [question] + variants


async def _semantic_search(query: str, k: int, where: dict | None = None,
                           q_vec: list[float] | None = None) -> dict[str, store.Hit]:
    """Single semantic query → {chunk_id: (doc, meta, distance)}."""
    if q_vec is None:
        [q_vec] = await llm.embed([query], prefix="search_query")
    return await store.query_embedding(q_vec, top_k=k, filters=where)


def _rrf(ranked_lists: list[list[str]], weights: list[float] | None = None,
         k: int = _RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk IDs.

    weights: per-list multipliers (default 1.0 each). Pass [1.0, 2.0] to give
    the second list (BM25) twice the influence of the first (semantic).
    """
    scores: dict[str, float] = {}
    for i, ranked in enumerate(ranked_lists):
        w = weights[i] if weights else 1.0
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank + 1)
    return scores


def _diverse_top(
    rrf_scores: dict[str, float],
    sem_hits: dict[str, tuple[str, dict, float]],
    k: int = 8,
    max_per_file: int = 2,
    named_doc_set: set[str] | None = None,
) -> list[str]:
    """Top-k by RRF score, capped at max_per_file chunks per source document.

    max_per_file=2 lets the summary chunk AND the recommendations/detail chunk
    from the same document both make it through, while still preventing any
    single file from consuming all slots.

    Files in named_doc_set get proportionally more slots: a single named doc gets
    all k slots (so every procedural step or spec table row is reachable); two named
    docs split them evenly at k//2 each; more docs fall back to 4.
    """
    named_doc_set = named_doc_set or set()
    # Fewer named docs → more slots each, so deep sections (timing steps, spec rows) surface.
    named_limit = min(k, max(4, k // max(1, len(named_doc_set)))) if named_doc_set else 0
    ranked = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    selected: list[str] = []
    file_counts: dict[str, int] = {}
    for cid in ranked:
        if len(selected) >= k:
            break
        fname = sem_hits[cid][1].get("filename", "")
        limit = named_limit if fname in named_doc_set else max_per_file
        if file_counts.get(fname, 0) < limit:
            selected.append(cid)
            file_counts[fname] = file_counts.get(fname, 0) + 1
    return selected


_NOT_FOUND = "That information is not in the ingested documents."


async def _inject_named_docs(
    named_docs: list[str],
    sem_hits: dict,
    rrf_scores: dict,
    folder_filter: str | None,
    all_bm25: dict[str, float],
) -> None:
    """Guarantee top BM25-ranked chunks from explicitly named documents appear in context."""
    slots_per_doc = max(2, 8 // len(named_docs))
    for named in named_docs:
        doc_hits = await store.get_by_filter(
            store.with_folder(store.eq("filename", named), folder_filter), distance=998.0
        )
        doc_ids = list(doc_hits.keys())
        doc_ranked = sorted(doc_ids, key=lambda x: -all_bm25.get(x, 0.0))
        for cid in doc_ranked[:slots_per_doc]:
            if cid not in sem_hits:
                sem_hits[cid] = doc_hits[cid]
        for rank, cid in enumerate(doc_ranked[:slots_per_doc]):
            rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + rank + 1) * 2)
        # Always guarantee the spec-table summary chunk is in context — BM25 length
        # normalisation often ranks it below body chunks so it may miss the top slots.
        for cid in doc_ids:
            if cid.endswith("_1_-1"):
                if cid not in sem_hits:
                    text, meta, _ = doc_hits[cid]
                    sem_hits[cid] = (text, meta, 994.0)
                rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + 1) * 3)
                break


async def _inject_system_names(
    sys_names: list[str],
    sem_hits: dict,
    rrf_scores: dict,
    named_doc_set: set[str],
) -> None:
    """Inject summary chunks for subsystems named explicitly in the query.

    Runs a per-name BM25 search to avoid token-overlap confusing the global ranking
    (e.g. "PULSE-3 Detector Array" vs "PULSE-3 Propulsion Module" in the same query).
    """
    for sys_name in sys_names:
        sys_results = bm25_index.search(sys_name, top_k=1)
        if not sys_results or sys_results[0][1] <= 0:
            continue
        top_cid = sys_results[0][0]
        summary_cid = "_".join(top_cid.split("_")[:-1]) + "_-1"
        if summary_cid not in sem_hits:
            fetched = await store.get_by_ids([summary_cid], distance=997.0)
            if summary_cid in fetched:
                sem_hits[summary_cid] = fetched[summary_cid]
                named_doc_set.add(fetched[summary_cid][1].get("filename", ""))
        else:
            named_doc_set.add(sem_hits[summary_cid][1].get("filename", ""))
        if summary_cid in sem_hits:
            rrf_scores[summary_cid] = max(rrf_scores.get(summary_cid, 0.0), 1.0 / (_RRF_K + 1) * 2)


async def _apply_comparison_boost(
    question: str,
    sem_hits: dict,
    rrf_scores: dict,
    folder_filter: str | None,
    sem_where: dict | None,
) -> None:
    """Surface aggregate/digest documents for comparison and ranking queries.

    Two-pass: augmented BM25 lifts digest docs by keyword; a framed semantic search
    catches aggregate docs whose wording differs from individual spec vocabulary.
    """
    aug_query = "domain performance digest top performer " + question
    aug_results = bm25_index.search(aug_query, folder=folder_filter, top_k=config.RETRIEVAL_K)
    scored = [(cid, s) for cid, s in aug_results if s > 0]
    # One batched fetch instead of a round-trip per candidate.
    missing = [cid for cid, _ in scored if cid not in sem_hits]
    if missing:
        sem_hits.update(await store.get_by_ids(missing, distance=991.0))
    for rank, (cid, _score) in enumerate(scored):
        if cid in sem_hits:
            rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + rank + 1) * 3)
    try:
        comp_sem_query = f"domain performance digest top performers highest {question}"
        comp_hits = await _semantic_search(comp_sem_query, config.RETRIEVAL_K, where=sem_where)
        comp_ranked = sorted(comp_hits.items(), key=lambda x: x[1][2])
        for rank, (cid, data) in enumerate(comp_ranked):
            if cid not in sem_hits or data[2] < sem_hits[cid][2]:
                sem_hits[cid] = data
            rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + rank + 1) * 3)
    except Exception:  # a failed boost must not fail the query
        log.warning("comparison semantic search failed", exc_info=True)
    log.info("comparison boost: augmented BM25 + semantic ran for ranking query")


# Folder-filter helper now lives in store (Haystack filter syntax, not Chroma's).
_with_folder = store.with_folder


async def _catalog_context(
    question: str,
    folder_filter: str | None,
) -> tuple[list[str], list[dict]] | None:
    """Catalog/inventory mode: first page chunks from every ingested document.

    For structured documents (SOPs, specs, manuals) the header block — document
    number, revision, date, author — always lives on page 1. Fetching by position
    (first 3 chunks of page 1) is more reliable than BM25 scoring, which drifts
    toward wherever the query term appears most frequently in the body.

    Capped at 15 documents to keep context within a range the LLM can synthesise.
    """
    docs = await ingest.list_documents()
    if folder_filter:
        docs = [d for d in docs if d.get("folder") == folder_filter]
    if not docs:
        return None

    capped = docs[:15]
    # Constrain to page 1 in the query rather than fetching every chunk of every
    # document and filtering in Python — on a large corpus that pulled the whole
    # store into memory to build a 15-row table.
    batch_where = store.and_(
        store.in_("doc_id", [d["doc_id"] for d in capped]),
        store.eq("page", 1),
        store.eq("folder", folder_filter) if folder_filter else None,
    )
    all_fetched = await store.get_by_filter(batch_where)

    by_doc: dict[str, list[tuple]] = {}
    for cid, (text, meta, _dist) in all_fetched.items():
        by_doc.setdefault(meta["doc_id"], []).append((cid, text, meta))

    context_parts: list[str] = []
    source_acc = _source_acc()

    for doc in capped:
        items = by_doc.get(doc["doc_id"], [])
        if not items:
            continue

        text_chunks = [
            (cid, text, meta)
            for cid, text, meta in items
            if meta.get("chunk_type") != "image" and text and text.strip()
        ]
        if not text_chunks:
            continue

        # First 3 chunks from page 1, sorted by chunk_index — the document header
        # (revision, doc number, date) is always in this region for SOPs and specs.
        # chunk_index is now persisted, so the `>= 0` guard actually excludes the
        # synthesised summary chunk; previously it defaulted to 0 and never did.
        page1 = sorted(
            [c for c in text_chunks if c[2].get("chunk_index", 0) >= 0],
            key=lambda x: x[2].get("chunk_index", 0),
        )[:3]
        chosen = page1 if page1 else text_chunks[:2]

        for _, text, meta in chosen:
            context_parts.append(f"[{meta['filename']}, page {meta['page']}]\n{text}")
            _add_source(source_acc, meta)

    return (context_parts, _finalize_sources(source_acc)) if context_parts else None


async def _retrieve(question: str, folder_filter: str | None = None) -> tuple[str, list] | None:
    """Run the full retrieval pipeline. Returns (system_prompt, sources) or None."""
    if bm25_index.size() == 0:
        await bm25_index.load_or_rebuild()

    if _CATALOG_RE.search(question):
        result = await _catalog_context(question, folder_filter)
        if result:
            log.info("catalog mode: returning one chunk per document (%d docs)", len(result[1]))
            return result

    sem_where = _with_folder(None, folder_filter)

    if config.QUERY_EXPANSION:
        # Start embedding the original query immediately so it overlaps with the LLM
        # expansion call, then batch-embed all variants in a single HTTP request.
        orig_embed_task = asyncio.create_task(llm.embed([question], prefix="search_query"))
        variants = await _expand_query(question)
        log.info("queries: %s", variants)
        extra_vecs = await llm.embed(variants[1:], prefix="search_query") if variants[1:] else []
        [orig_vec] = await orig_embed_task
        sem_results = list(await asyncio.gather(
            _semantic_search(question, config.RETRIEVAL_K, where=sem_where, q_vec=orig_vec),
            *[_semantic_search(q, config.RETRIEVAL_K, where=sem_where, q_vec=v)
              for q, v in zip(variants[1:], extra_vecs)]
        ))
    else:
        log.info("queries: %s (expansion disabled)", [question])
        sem_results = [await _semantic_search(question, config.RETRIEVAL_K, where=sem_where)]

    sem_hits: dict[str, store.Hit] = {}
    for hits in sem_results:
        for cid, data in hits.items():
            if cid not in sem_hits or data[2] < sem_hits[cid][2]:
                sem_hits[cid] = data

    named_docs = await _find_named_docs(question)

    # Bypass the semantic gate when the question explicitly names an ingested document —
    # BM25 + named-doc injection below will still populate context even when
    # semantic distance exceeds the threshold (e.g. very specific procedural queries).
    has_semantic = any(d <= config.DISTANCE_THRESHOLD for _, _, d in sem_hits.values())
    if not has_semantic and not named_docs:
        return None

    is_value_query = bool(_VALUE_QUERY_RE.search(question))

    # Value queries cast a wider BM25 net — the chunk with the exact number may
    # rank lower than conceptual chunks in semantic search, so more BM25 candidates
    # increases the chance of pulling it into context.
    bm25_k = config.RETRIEVAL_K * 2 if is_value_query else config.RETRIEVAL_K
    all_bm25, bm25_filtered = bm25_index.search_with_folder(
        question, folder=folder_filter, top_k=bm25_k
    )
    bm25_top = [cid for cid, score in bm25_filtered if score > 0]

    missing = [cid for cid in bm25_top if cid not in sem_hits]
    if missing:
        sem_hits.update(await store.get_by_ids(missing, distance=999.0))

    sem_ranked = [cid for cid, _ in sorted(sem_hits.items(), key=lambda x: x[1][2])]
    # For value queries, weight BM25 at 2× semantic so exact token hits dominate.
    rrf_weights = [1.0, 2.0] if is_value_query else None
    rrf_scores = _rrf([sem_ranked, bm25_top], weights=rrf_weights)

    named_doc_set: set[str] = set(named_docs)

    if named_docs:
        await _inject_named_docs(named_docs, sem_hits, rrf_scores, folder_filter, all_bm25)

    sys_names = _SYS_REF.findall(question)
    if sys_names:
        await _inject_system_names(sys_names, sem_hits, rrf_scores, named_doc_set)

    if _COMPARISON_RE.search(question):
        await _apply_comparison_boost(question, sem_hits, rrf_scores, folder_filter, sem_where)

    # Structural boosts applied before pool selection so boosted chunks compete
    # fairly for the top-N reranker slots. A summary or table chunk that would
    # rank #21 without a boost might rank #8 with it, changing what the reranker sees.
    for cid in rrf_scores:
        if cid.endswith("_1_-1"):
            rrf_scores[cid] *= _SUMMARY_BOOST

    if is_value_query:
        for cid in rrf_scores:
            if cid in sem_hits and sem_hits[cid][1].get("table_id"):
                rrf_scores[cid] *= 1.5

    if config.RERANK_ENABLED:
        top_n = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:config.RERANK_TOP_N]
        candidates = [(cid, sem_hits[cid][0]) for cid in top_n if cid in sem_hits]
        # The cross-encoder is a synchronous torch forward pass over 20 pairs —
        # a few hundred milliseconds during which the event loop must not be
        # blocked, or every concurrent request stalls behind this one.
        order = await asyncio.get_running_loop().run_in_executor(
            None, reranker.rerank, question, candidates
        )
        reranked_cids = {candidates[i][0] for i in range(len(candidates))}
        for new_rank, orig_idx in enumerate(order):
            rrf_scores[candidates[orig_idx][0]] = 1.0 / (_RRF_K + new_rank + 1)
        # Cap non-reranked chunks strictly below the worst reranked score so chunks
        # the reranker never evaluated cannot outrank the reranker's top picks.
        # Exempt force-injected chunks (sentinel distances) — they carry hard pipeline
        # guarantees (named-doc, sys-name, comparison-boost) that must be preserved.
        if candidates:
            floor = 1.0 / (_RRF_K + len(candidates) + 1) * 0.5
            for cid in rrf_scores:
                if cid not in reranked_cids:
                    dist = sem_hits.get(cid, (None, None, 0.0))[2]
                    if dist not in _INJECTED_SENTINELS:
                        rrf_scores[cid] = min(rrf_scores[cid], floor)

    top_ids = _diverse_top(rrf_scores, sem_hits, named_doc_set=named_doc_set)

    # Product/document identity guarantee: a chunk deep in a manual (e.g. a page-24
    # safety warning) often never restates the document's title or product name, so
    # the model can't verify which product a fact belongs to and will (correctly,
    # per its own grounding rules) refuse to attribute it. Inject each document's
    # synthesized summary chunk (always {doc_id}_1_-1, see ingest.py) for every
    # document represented in context, not just explicitly-named ones — generalizes
    # the guarantee _inject_named_docs already gives named documents.
    docs_in_context = {sem_hits[cid][1]["doc_id"] for cid in top_ids}
    docs_with_summary = {sem_hits[cid][1]["doc_id"] for cid in top_ids if cid.endswith("_1_-1")}
    missing_summary_docs = docs_in_context - docs_with_summary
    if missing_summary_docs:
        summary_ids = [f"{did}_1_-1" for did in missing_summary_docs]
        for cid, hit in (await store.get_by_ids(summary_ids, distance=993.0)).items():
            if cid not in sem_hits:
                sem_hits[cid] = hit
            if cid not in top_ids:
                top_ids.append(cid)

    # Table completeness: if any top chunk is tagged as part of a table, pull in
    # every chunk from that same table so the model always sees the full table.
    table_ids_in_context: set[str] = set()
    for cid in top_ids:
        tid = sem_hits[cid][1].get("table_id", "")
        if tid:
            table_ids_in_context.add(tid)

    if table_ids_in_context:
        t_where = _with_folder(store.in_("table_id", list(table_ids_in_context)), folder_filter)
        t_hits = await store.get_by_filter(t_where, distance=995.0)
        # Drop tables too large to be tables before anything is added to context.
        per_table: dict[str, list] = {}
        for cid, hit in t_hits.items():
            per_table.setdefault(hit[1].get("table_id", ""), []).append((cid, hit))
        plausible: list = []
        for tid, items in per_table.items():
            if len(items) > _TABLE_MAX_CHUNKS:
                log.info("table %s spans %d chunks — treating as a parse artifact, "
                         "skipping expansion", tid, len(items))
                continue
            plausible.extend(items)
        paired = sorted(
            plausible,
            key=lambda kv: (kv[1][1]["page"], kv[1][1].get("chunk_index", 0)),
        )[:_TABLE_EXPANSION_BUDGET]
        table_cids_ordered: list[str] = []
        for cid, hit in paired:
            if cid not in sem_hits:
                sem_hits[cid] = hit
            if cid not in top_ids:
                table_cids_ordered.append(cid)
        # Group table chunks first so they appear together at the top of context
        top_ids = table_cids_ordered + [c for c in top_ids if c not in set(table_cids_ordered)]

    for cid in top_ids:
        doc, meta, dist = sem_hits[cid]
        log.info("chunk rrf=%.4f dist=%.3f file=%s page=%s text=%r",
                 rrf_scores.get(cid, 0.0), dist, meta["filename"], meta["page"], doc[:80])

    # Parent-chunk retrieval: for each selected chunk, fetch ±_EXPAND_WINDOW neighbors
    # on the same page so the LLM sees the full surrounding paragraph, not just a
    # 512-char fragment that may start or end mid-sentence.
    neighbor_pool: set[str] = set()
    parsed_top: dict[str, tuple[str, int, int]] = {}  # cid → (doc_id, page, chunk_index)
    for cid in top_ids:
        parts = cid.split("_")
        if len(parts) != 3:
            continue
        try:
            doc_id_p, page_p, idx_p = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if idx_p < 0:
            continue  # summary (−1) and image (<−1) chunks: no expansion
        parsed_top[cid] = (doc_id_p, page_p, idx_p)
        for d in range(-_EXPAND_WINDOW, _EXPAND_WINDOW + 1):
            ni = idx_p + d
            if ni >= 0:
                neighbor_pool.add(f"{doc_id_p}_{page_p}_{ni}")

    missing_neighbors = neighbor_pool - set(sem_hits.keys())
    if missing_neighbors:
        sem_hits.update(await store.get_by_ids(list(missing_neighbors), distance=996.0))

    context_parts, source_acc = [], _source_acc()
    context_cids_used: set[str] = set()
    for cid in top_ids:
        _, meta, _ = sem_hits[cid]
        if cid in parsed_top:
            doc_id_p, page_p, idx_p = parsed_top[cid]
            neighbor_cids: list[str] = []
            for d in range(-_EXPAND_WINDOW, _EXPAND_WINDOW + 1):
                ni = idx_p + d
                if ni >= 0:
                    ncid = f"{doc_id_p}_{page_p}_{ni}"
                    if ncid in sem_hits:
                        neighbor_cids.append(ncid)
            fresh = [ncid for ncid in neighbor_cids if ncid not in context_cids_used]
            if not fresh:
                continue  # entire neighborhood already shown in a prior block
            text = "\n".join(sem_hits[ncid][0] for ncid in neighbor_cids)
            context_cids_used.update(neighbor_cids)
        else:
            if cid in context_cids_used:
                continue
            text = sem_hits[cid][0]
            context_cids_used.add(cid)

        context_parts.append(f"[{meta['filename']}, page {meta['page']}]\n{text}")
        _add_source(source_acc, meta)

    return context_parts, _finalize_sources(source_acc)


def _retrieval_query(question: str, history: list[dict] | None) -> str:
    """Augment the retrieval query with the last user turn from history.

    Follow-up questions ("explain that", "list the steps", "give me more detail")
    carry no subject on their own. Prepending the previous user message gives the
    retrieval pipeline the same context the LLM already receives via history.
    Only the immediately preceding user turn is used — older context adds noise.
    """
    if not history:
        return question
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), None
    )
    if not last_user or last_user == question:
        return question
    return f"{last_user} {question}"


def _build_system(context_parts: list[str], procedure: dict | None) -> str:
    context = "\n\n---\n\n".join(context_parts)

    # Extract unique filenames from context block headers so the model has an
    # explicit list of what's real — prevents it from inventing document names.
    seen: set[str] = set()
    filenames: list[str] = []
    for part in context_parts:
        first_line = part.split("\n", 1)[0]  # "[filename, page N]"
        if first_line.startswith("[") and "," in first_line:
            fname = first_line[1 : first_line.rfind(",")].strip()
            if fname and fname not in seen:
                filenames.append(fname)
                seen.add(fname)
    source_list = "\n".join(f"  • {f}" for f in filenames) if filenames else "  (none)"

    if procedure:
        steps = procedure["steps"]
        idx = procedure["step_idx"]
        return _PROCEDURE_SYSTEM.format(
            filename=procedure["filename"],
            step_num=idx + 1,
            total_steps=len(steps),
            step_text=steps[idx] if steps else "(no steps found)",
            figure_note=_FIGURE_NOTE,
            source_list=source_list,
            context=context,
        )
    return _SYSTEM.format(context=context, figure_note=_FIGURE_NOTE, source_list=source_list)


async def answer(question: str, history: list[dict] | None = None,
                 folder_filter: str | None = None, procedure: dict | None = None) -> dict:
    if await store.count() == 0:
        return {"answer": _NOT_FOUND, "sources": []}
    result = await _retrieve(_retrieval_query(question, history), folder_filter=folder_filter)
    if result is None:
        return {"answer": _NOT_FOUND, "sources": []}
    context_parts, sources = result
    reply = await llm.chat(system=_build_system(context_parts, procedure),
                           user=question, history=history)
    return {"answer": reply, "sources": sources}


async def answer_stream(question: str, history: list[dict] | None = None,
                        folder_filter: str | None = None, procedure: dict | None = None):
    """Async generator: yields {token} dicts then a final {sources, done} dict."""
    if await store.count() == 0:
        yield {"token": _NOT_FOUND}
        yield {"sources": [], "done": True}
        return

    # Catalog queries bypass normal RAG — read per-document metadata and stream rows.
    if _CATALOG_RE.search(question) and not procedure:
        async for chunk in _catalog_stream(question, folder_filter):
            yield chunk
        return

    result = await _retrieve(_retrieval_query(question, history), folder_filter=folder_filter)
    if result is None:
        yield {"token": _NOT_FOUND}
        yield {"sources": [], "done": True}
        return
    context_parts, sources = result
    async for token in llm.chat_stream(system=_build_system(context_parts, procedure),
                                       user=question, history=history):
        yield {"token": token}
    yield {"sources": sources, "done": True}
