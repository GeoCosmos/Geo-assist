import asyncio
import logging
import re

import bm25_index
import config
import ingest
import llm
import reranker

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
    r'(?:table|list|overview|summary)\s+of\s+(?:all\s+)?(?:files?|documents?)|'
    r'\b(?:all|each|every)\s+(?:files?|documents?)\b|'
    r'\bfile\s*names?\s*(?:and|with|,)|'
    r'\b(?:all|each|every|list)\b.{0,40}\brevision\b',
    re.IGNORECASE,
)

# ── per-document catalog extraction ───────────────────────────────────────────

_REV_TABLE_HEADER = re.compile(
    r'\|\s*(?:is\.?\s*rev|ed\.?\s*r[eé]v|revision|rev(?:ision)?\.?\s*no\.?|version)\b',
    re.IGNORECASE,
)
_TABLE_ROW = re.compile(r'^\|\s*([^|]+?)\s*\|')
_DOCNUM_FROM_FILENAME = re.compile(r'^([A-Za-z]{1,5}\d{2,}[A-Za-z0-9]*|[A-Z]{2}-\d+)', re.IGNORECASE)
_COPYRIGHT_EDITION = re.compile(r'[©\-]\s*\w[\w\s]+[–\-]\s*(\S+)\s+(\S+)\s*$', re.MULTILINE)
_DATE_RE = re.compile(
    r'\b(?:date|issued|approved)\s*[:#.]?\s*'
    r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{2}[/\-]\d{2}|\w+\.?\s+\d{1,2},?\s+\d{4})',
    re.IGNORECASE,
)


def _doc_number_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    m = _DOCNUM_FROM_FILENAME.match(stem)
    return m.group(1).upper() if m else "—"


_CHUNK_TITLE_PREFIX = re.compile(r'^\[[^\]]{1,120}\]\s*')
_PAGE_HEADER_COMBINED = re.compile(
    # "Issue 3  Rev 2" / "Ed. 3  Rév. 2" / "Is. 3  Rev. 2" — issue + revision as separate fields
    r'(?:iss?ue|ed(?:ition)?\.?|éd(?:ition)?\.?)\s*[:\.]?\s*(\d+)\s{0,15}'
    r'(?:rev(?:ision)?\.?|rév\.?)\s*[:\.]?\s*(\d[\w\.]*)',
    re.IGNORECASE,
)
_PAGE_HEADER_REV = re.compile(
    # Standalone: must have explicit label + colon/dot to avoid matching section/table numbers
    r'(?:is\.?\s*rev(?:ision)?|rev(?:ision)?\.?|rév(?:ision)?\.?)\s*[:\.]\s*(\d[\w\.]*)',
    re.IGNORECASE,
)


def _revision_from_page_headers(text_pairs: list[tuple[str, dict]]) -> str:
    """Extract revision from running page headers (top of page 2+).

    SAFRAN manuals stamp the current revision on every page header. We look at
    the first chunk of each page beyond page 1. Try combined Issue+Rev first
    (e.g. "Issue 3 Rev 2" → "3/2"), then standalone Rev label.
    """
    pages_seen: set[int] = set()
    for text, meta in text_pairs:
        page = meta.get("page", 1)
        if page <= 1 or page in pages_seen:
            continue
        pages_seen.add(page)
        top = "\n".join(text.splitlines()[:10])
        m = _PAGE_HEADER_COMBINED.search(top)
        if m:
            return f"{m.group(1)}/{m.group(2).strip()}"
        m = _PAGE_HEADER_REV.search(top)
        if m:
            return m.group(1).strip()
    return "—"


def _latest_revision_from_chunks(chunks: list[str]) -> str:
    """Scan all page-1 chunk text for a revision history table and return the
    last non-empty revision value.

    Each chunk stored in ChromaDB starts with a [filename] prefix line. That
    non-pipe line would normally reset in_rev_table=False mid-table when the
    same table spans multiple chunks. We skip those prefix lines so the table
    state carries across chunk boundaries correctly.
    """
    last_rev = None
    in_rev_table = False

    for text in chunks:
        for line in text.splitlines():
            stripped = line.strip()
            # Strip [filename] prefix. When in a rev table the overlap region
            # may embed a new row mid-line: "| B.Author | | 2.0 | date |..."
            # so scan the remainder for "| | value |" inline row boundaries.
            prefix_m = _CHUNK_TITLE_PREFIX.match(stripped)
            if prefix_m:
                if in_rev_table:
                    remainder = stripped[prefix_m.end():]
                    for row_m in re.finditer(r'\|\s*\|\s*([^|]{1,80}?)\s*\|', remainder):
                        val = row_m.group(1).strip()
                        if val and not re.match(r'^[-\s]+$', val):
                            last_rev = val
                continue
            if not stripped.startswith("|"):
                if in_rev_table and stripped:
                    in_rev_table = False
                continue
            if _REV_TABLE_HEADER.search(stripped):
                in_rev_table = True
                continue
            if "---" in stripped:
                continue
            if in_rev_table:
                m = _TABLE_ROW.match(stripped)
                if m:
                    val = m.group(1).strip()
                    if val:
                        last_rev = val

    return last_rev or "—"


def _date_from_text(text: str) -> str:
    m = _DATE_RE.search(text)
    return m.group(1).strip() if m else "—"


_CATALOG_EXTRACT_SYSTEM = """\
You are extracting one specific piece of information from a single document.
Reply with ONLY the value — no labels, no explanation, no punctuation.
If the information is not present reply with exactly: —\
"""


async def _llm_extract_field(filename: str, text: str, field: str) -> str:
    """Single-field LLM extraction for one document. Used only when code-based
    extraction returns nothing."""
    try:
        raw = await llm.chat(
            system=_CATALOG_EXTRACT_SYSTEM,
            user=f"Document: {filename}\n\nText:\n{text[:1500]}\n\nExtract: {field}",
        )
        val = raw.strip().splitlines()[0].strip(" \t:\"'")
        return val if val and val != "—" else "—"
    except Exception:
        return "—"


async def _catalog_stream(
    question: str,
    col,
    folder_filter: str | None,
    current_user: dict | None,
):
    """Per-document extraction loop for catalog queries.

    For each document: extracts doc number from the filename pattern, finds the
    latest revision by scanning all page-1 chunks for a revision history table,
    and falls back to a focused single-document LLM call only for fields that
    code-based extraction cannot find. Streams each row as it completes.
    """
    docs = ingest.list_documents()
    if folder_filter:
        docs = [d for d in docs if d.get("folder") == folder_filter]
    if not docs:
        yield {"token": _NOT_FOUND}
        yield {"sources": [], "done": True}
        return

    yield {"token": "| File | Document No | Revision |\n| --- | --- | --- |\n"}

    # Batch-fetch all chunks for all documents in one ChromaDB round-trip
    batch_where = _with_access(
        _with_folder({"doc_id": {"$in": [d["doc_id"] for d in docs]}}, folder_filter),
        current_user,
    )
    all_fetched = col.get(where=batch_where, include=["documents", "metadatas"])
    by_doc: dict[str, list[tuple[str, dict]]] = {}
    for text, meta in zip(all_fetched["documents"], all_fetched["metadatas"]):
        did = meta["doc_id"]
        if did not in by_doc:
            by_doc[did] = []
        by_doc[did].append((text, meta))

    sources = []
    for doc in docs:
        filename = doc["filename"]
        items = by_doc.get(doc["doc_id"], [])
        if not items:
            continue

        pairs = sorted(items, key=lambda x: (x[1].get("page", 1), x[1].get("chunk_index", 0)))
        text_pairs = [(t, m) for t, m in pairs if m.get("chunk_type") != "image" and t]

        def _chunks_for_page(page_num: int) -> list[str]:
            return [t for t, m in text_pairs if m.get("page") == page_num]

        page1_chunks = _chunks_for_page(1)
        combined = "\n".join(page1_chunks)

        doc_no   = _doc_number_from_filename(filename)
        # Running page header is most reliable — every page 2+ stamps the current revision
        revision = _revision_from_page_headers(text_pairs)
        # Fallback: revision history table on page 1 or 2
        if revision == "—":
            revision = _latest_revision_from_chunks(page1_chunks)
        if revision == "—":
            page2_chunks = _chunks_for_page(2)
            revision = _latest_revision_from_chunks(page2_chunks)

        # LLM fallback only for fields still missing after all code-based attempts
        if revision == "—":
            revision = await _llm_extract_field(filename, combined, "the latest revision or version number")
        if doc_no == "—":
            doc_no = await _llm_extract_field(filename, combined, "the document or reference number")

        yield {"token": f"| {filename} | {doc_no} | {revision} |\n"}
        sources.append({"filename": filename, "page": 1})

    yield {"sources": sources, "done": True}

log = logging.getLogger(__name__)

_RRF_K = 60
_EXPAND_WINDOW = 2  # neighbor chunks fetched on each side for parent-chunk retrieval


def _find_named_docs(question: str) -> list[str]:
    """Return filenames of ingested documents whose name or stem appears verbatim in the question.

    Uses the in-memory doc cache — no extra DB round-trip per query.
    The 4-char minimum on stems avoids short words like 'log' or 'map' matching incidentally.
    """
    q_lower = question.lower()
    named = []
    for doc in ingest.list_documents():
        fn = doc["filename"]
        fn_lower = fn.lower()
        stem_lower = fn_lower.rsplit(".", 1)[0] if "." in fn_lower else fn_lower
        if fn_lower in q_lower or (len(stem_lower) >= 4 and stem_lower in q_lower):
            named.append(fn)
    return named


_SUMMARY_BOOST = 3.0

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

# Included in the system prompt only when vision analysis is enabled, so the LLM
# is not primed to look for figure annotations that will never appear.
_FIGURE_NOTE = (
    "Some blocks begin with \"[Figure on page N]:\" — these are AI-generated descriptions "
    "of images extracted from that document page. Treat them as you would textual content: "
    "cite the source file and page when referencing them, and flag any uncertainty if the "
    "description seems ambiguous.\n\n"
) if config.VISION_MODEL else ""

_SYSTEM = """\
You are Geo-Assist, a technical document Q&A assistant operating in a secure, \
air-gapped environment. You answer questions strictly from the retrieved document \
excerpts shown at the bottom of this prompt.

━━━ HARD CONSTRAINTS — these override everything else ━━━
1. NEVER invent, guess, or produce a filename, document name, document number, \
   reference number, part number, revision number, or numeric value that is not \
   explicitly present in the retrieved context below. \
   If you find yourself writing a name or number that you cannot point to in the \
   retrieved text, stop and say "That information is not in the ingested documents."
2. You may ONLY reference documents whose filenames appear in the source list \
   below. Any other document name does not exist in this system — do not produce it.
3. Every fact, figure, and technical claim must be traceable to a specific passage \
   in the retrieved context. Never use knowledge from your training data for \
   factual assertions.
4. If the retrieved context does not contain what is needed, say exactly: \
   "That information is not in the ingested documents." \
   Do not approximate, extrapolate, or fill gaps from training knowledge.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Documents available in this session (the ONLY sources you may cite):
{source_list}

Read all retrieved context blocks before writing a single word. Each block is \
labelled with its source file and page — that label determines which document the \
text belongs to. Never attribute content from one block to a different document.

{figure_note}When the question names a specific document, use only blocks labelled \
with that document.

For specific values, names, or numbers: quote or paraphrase the exact passage that \
supports the claim. If you cannot point to that passage, say "I cannot find that in \
the retrieved text." Never fill the gap with a guess.

If the retrieved text contains contradictions within the same document, flag them \
explicitly rather than picking one silently.

When asked for a recommendation, derive it from measurements, test results, margins, \
and findings in the retrieved blocks — not from a section labelled "Recommendations".

When asked for a combined or total value, show arithmetic step by step, stating \
which document each number came from.

When context gives a nominal value and a tolerance range, state both.

When walking through a procedure, include every explicit timing value.

When asked to compare, contrast, rank, or tabulate multiple items, format the \
response as a markdown table with clear column headers. \
Use one row per item and one column per attribute being compared. \
Add a brief prose summary after the table if the comparison needs context.

After each key factual claim, cite the source file and page in parentheses. \
If you draw a logical inference from stated facts, say so explicitly.

Always respond in the same language the user used in their question. \
If the user writes in Russian, respond in Russian. \
If the user writes in Armenian, respond in Armenian. \
The retrieved context may be in English — translate as needed. \
If unsure of a technical term translation, include the English term in parentheses.

Retrieved context:
{context}
"""

_PROCEDURE_SYSTEM = """\
You are Geo-Assist in Procedure Mode. An engineer is executing a procedure step by \
step and may ask questions, request clarification, or want you to verify specifications \
against reference documents.

CURRENT PROCEDURE: {filename}
STEP {step_num} OF {total_steps}
──────────────────────────────────────
{step_text}
──────────────────────────────────────

Your responsibilities:
1. Help the engineer understand and execute the current step.
2. Answer questions using the retrieved reference documents shown below.
3. CONFLICT DETECTION — if any retrieved reference document contains a specification, \
   value, or instruction that contradicts what the current step says, output a warning \
   immediately before your answer, in this format:
   ⚠️ CONFLICT: Procedure says [procedure claim] but [reference source] \
   (page N) says [reference claim]. Verify before proceeding.
4. After each factual claim, cite the source file and page in parentheses.
5. If the step mentions a numeric value (pressure, torque, temperature, voltage, timing), \
   always cross-check the retrieved context for the same parameter and flag any discrepancy.

HARD CONSTRAINTS:
- NEVER invent filenames, document numbers, part numbers, or numeric values not \
  present in the procedure step or the retrieved context below.
- Every fact must be traceable to the procedure step text or the retrieved context.
- If the retrieved context does not contain what is needed, say exactly: \
  "That information is not in the ingested documents."
- If retrieved context contradicts itself, flag all contradictions explicitly.

Available reference documents (the ONLY sources you may cite):
{source_list}

{figure_note}Retrieved reference documents:
{context}
"""

_EXPAND_SYSTEM = """\
You are a search query optimizer for a technical document retrieval system.
Given a question, produce 3 alternative search queries to retrieve the most relevant \
passages from a technical document corpus. Make each query meaningfully different:
- Query 1: specific technical terms, exact values, or table headers likely to appear \
  verbatim in documents (part numbers, units, spec values, parameter names)
- Query 2: higher-level concepts such as trade-offs, comparisons, recommendations, \
  methodology, or design rationale that would surface evaluative passages
- Query 3: different vocabulary covering the same intent — synonyms, domain-specific \
  phrasing, or the perspective of a different document section (e.g. summary vs. detail)
Output only the 3 queries, one per line, no numbering or explanation.\
"""


async def _expand_query(question: str) -> list[str]:
    try:
        raw = await llm.chat(system=_EXPAND_SYSTEM, user=question, model=config.EXPAND_MODEL)
        variants = [q.strip() for q in raw.splitlines() if q.strip()][:2]
    except Exception:
        log.warning("query expansion failed", exc_info=True)
        variants = []
    return [question] + variants


async def _semantic_search(col, query: str, k: int, where: dict | None = None,
                           q_vec: list[float] | None = None,
                           total: int | None = None) -> dict[str, tuple[str, dict, float]]:
    """Single semantic query → {chunk_id: (doc, meta, distance)}."""
    if q_vec is None:
        [q_vec] = await llm.embed([query], prefix="search_query")
    n = min(k, total if total is not None else col.count())
    query_kwargs: dict = {
        "query_embeddings": [q_vec],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    results = col.query(**query_kwargs)
    hits = {}
    for cid, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if cid not in hits or dist < hits[cid][2]:
            hits[cid] = (doc, meta, dist)
    return hits


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


def _inject_named_docs(
    named_docs: list[str],
    sem_hits: dict,
    rrf_scores: dict,
    col,
    folder_filter: str | None,
    all_bm25: dict[str, float],
) -> None:
    """Guarantee top BM25-ranked chunks from explicitly named documents appear in context."""
    slots_per_doc = max(2, 8 // len(named_docs))
    for named in named_docs:
        doc_result = col.get(
            where=_with_folder({"filename": named}, folder_filter),
            include=["documents", "metadatas"],
        )
        doc_ids = doc_result.get("ids") or []
        doc_ranked = sorted(doc_ids, key=lambda x: -all_bm25.get(x, 0.0))
        # Batch-fetch all missing chunks in one call instead of one per chunk.
        missing_cids = [cid for cid in doc_ranked[:slots_per_doc] if cid not in sem_hits]
        if missing_cids:
            fetched = col.get(ids=missing_cids, include=["documents", "metadatas"])
            for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
                sem_hits[cid] = (doc, meta, 998.0)
        for rank, cid in enumerate(doc_ranked[:slots_per_doc]):
            rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + rank + 1) * 2)
        # Always guarantee the spec-table summary chunk is in context — BM25 length
        # normalisation often ranks it below body chunks so it may miss the top slots.
        for idx, cid in enumerate(doc_ids):
            if cid.endswith("_1_-1"):
                if cid not in sem_hits:
                    sem_hits[cid] = (doc_result["documents"][idx], doc_result["metadatas"][idx], 994.0)
                rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + 1) * 3)
                break


def _inject_system_names(
    sys_names: list[str],
    sem_hits: dict,
    rrf_scores: dict,
    named_doc_set: set[str],
    col,
) -> None:
    """Inject summary chunks for subsystems named explicitly in the query.

    Runs a per-name BM25 search to avoid token-overlap confusing the global ranking
    (e.g. "PULSE-3 Detector Array" vs "PULSE-3 Propulsion Module" in the same query).
    """
    for sys_name in sys_names:
        sys_results = bm25_index.search(sys_name)
        if not sys_results or sys_results[0][1] <= 0:
            continue
        top_cid = sys_results[0][0]
        parts = top_cid.split("_")
        summary_cid = "_".join(parts[:-1]) + "_-1"
        if summary_cid not in sem_hits:
            fetched = col.get(ids=[summary_cid], include=["documents", "metadatas"])
            if fetched["ids"]:
                sem_hits[summary_cid] = (fetched["documents"][0], fetched["metadatas"][0], 997.0)
                named_doc_set.add(fetched["metadatas"][0].get("filename", ""))
        else:
            named_doc_set.add(sem_hits[summary_cid][1].get("filename", ""))
        rrf_scores[summary_cid] = max(rrf_scores.get(summary_cid, 0.0), 1.0 / (_RRF_K + 1) * 2)


async def _apply_comparison_boost(
    question: str,
    sem_hits: dict,
    rrf_scores: dict,
    col,
    folder_filter: str | None,
    sem_where: dict | None,
) -> None:
    """Surface aggregate/digest documents for comparison and ranking queries.

    Two-pass: augmented BM25 lifts digest docs by keyword; a framed semantic search
    catches aggregate docs whose wording differs from individual spec vocabulary.
    """
    aug_query = "domain performance digest top performer " + question
    aug_results = bm25_index.search(aug_query, folder=folder_filter)
    for rank, (cid, score) in enumerate(aug_results[:config.RETRIEVAL_K]):
        if score <= 0:
            continue
        if cid not in sem_hits:
            fetched = col.get(ids=[cid], include=["documents", "metadatas"])
            if fetched["ids"]:
                sem_hits[cid] = (fetched["documents"][0], fetched["metadatas"][0], 991.0)
        rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + rank + 1) * 3)
    try:
        comp_sem_query = f"domain performance digest top performers highest {question}"
        comp_hits = await _semantic_search(col, comp_sem_query, config.RETRIEVAL_K, where=sem_where)
        comp_ranked = sorted(comp_hits.items(), key=lambda x: x[1][2])
        for rank, (cid, data) in enumerate(comp_ranked):
            if cid not in sem_hits or data[2] < sem_hits[cid][2]:
                sem_hits[cid] = data
            rrf_scores[cid] = max(rrf_scores.get(cid, 0.0), 1.0 / (_RRF_K + rank + 1) * 3)
    except Exception:
        log.warning("comparison semantic search failed", exc_info=True)
    log.info("comparison boost: augmented BM25 + semantic ran for ranking query")


def _with_folder(where: dict | None, folder: str | None) -> dict | None:
    """Intersect a ChromaDB where clause with an optional folder filter."""
    if not folder:
        return where
    folder_clause = {"folder": folder}
    if not where:
        return folder_clause
    return {"$and": [where, folder_clause]}


def _with_access(where: dict | None, current_user: dict | None) -> dict | None:
    """Add access-control filter for non-admin users when auth is enabled."""
    if not config.AUTH_ENABLED or not current_user:
        return where
    if current_user.get("role") == "admin":
        return where
    access_clause = {"$or": [{"access": "public"}, {"owner": current_user["username"]}]}
    if not where:
        return access_clause
    return {"$and": [where, access_clause]}


def _catalog_context(
    question: str,
    col,
    folder_filter: str | None,
    current_user: dict | None,
) -> tuple[list[str], list[dict]] | None:
    """Catalog/inventory mode: first page chunks from every ingested document.

    For structured documents (SOPs, specs, manuals) the header block — document
    number, revision, date, author — always lives on page 1. Fetching by position
    (first 3 chunks of page 1) is more reliable than BM25 scoring, which drifts
    toward wherever the query term appears most frequently in the body.

    Capped at 15 documents to keep context within a range the LLM can synthesise.
    """
    docs = ingest.list_documents()
    if folder_filter:
        docs = [d for d in docs if d.get("folder") == folder_filter]
    if not docs:
        return None

    capped = docs[:15]
    batch_where = _with_access(
        _with_folder({"doc_id": {"$in": [d["doc_id"] for d in capped]}}, folder_filter),
        current_user,
    )
    all_fetched = col.get(where=batch_where, include=["documents", "metadatas"])

    by_doc: dict[str, list[tuple]] = {}
    for cid, text, meta in zip(all_fetched["ids"], all_fetched["documents"], all_fetched["metadatas"]):
        did = meta["doc_id"]
        if did not in by_doc:
            by_doc[did] = []
        by_doc[did].append((cid, text, meta))

    context_parts: list[str] = []
    sources: list[dict] = []
    seen: set = set()

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
        page1 = sorted(
            [c for c in text_chunks if c[2]["page"] == 1 and c[2].get("chunk_index", 0) >= 0],
            key=lambda x: x[2].get("chunk_index", 0),
        )[:3]
        chosen = page1 if page1 else text_chunks[:2]

        for _, text, meta in chosen:
            context_parts.append(f"[{meta['filename']}, page {meta['page']}]\n{text}")
            key = (meta["doc_id"], meta["page"])
            if key not in seen:
                seen.add(key)
                sources.append({"filename": meta["filename"], "page": meta["page"]})

    return (context_parts, sources) if context_parts else None


async def _retrieve(question: str, col, folder_filter: str | None = None,
                    current_user: dict | None = None) -> tuple[str, list] | None:
    """Run the full retrieval pipeline. Returns (system_prompt, sources) or None."""
    if bm25_index.size() == 0:
        bm25_index.load_or_rebuild(col)

    if _CATALOG_RE.search(question):
        result = _catalog_context(question, col, folder_filter, current_user)
        if result:
            log.info("catalog mode: returning one chunk per document (%d docs)", len(result[1]))
            return result

    sem_where = _with_access(_with_folder(None, folder_filter), current_user)
    total_chunks = col.count()

    if config.QUERY_EXPANSION:
        # Start embedding the original query immediately so it overlaps with the LLM
        # expansion call, then batch-embed all variants in a single HTTP request.
        orig_embed_task = asyncio.create_task(llm.embed([question], prefix="search_query"))
        variants = await _expand_query(question)
        log.info("queries: %s", variants)
        extra_vecs = await llm.embed(variants[1:], prefix="search_query") if variants[1:] else []
        [orig_vec] = await orig_embed_task
        sem_results = list(await asyncio.gather(
            _semantic_search(col, question, config.RETRIEVAL_K, where=sem_where, q_vec=orig_vec, total=total_chunks),
            *[_semantic_search(col, q, config.RETRIEVAL_K, where=sem_where, q_vec=v, total=total_chunks)
              for q, v in zip(variants[1:], extra_vecs)]
        ))
    else:
        log.info("queries: %s (expansion disabled)", [question])
        sem_results = [await _semantic_search(col, question, config.RETRIEVAL_K, where=sem_where, total=total_chunks)]

    sem_hits: dict[str, tuple[str, dict, float]] = {}
    for hits in sem_results:
        for cid, data in hits.items():
            if cid not in sem_hits or data[2] < sem_hits[cid][2]:
                sem_hits[cid] = data

    named_docs = _find_named_docs(question)

    # Bypass the semantic gate when the question explicitly names an ingested document —
    # BM25 + named-doc injection below will still populate context even when
    # semantic distance exceeds the threshold (e.g. very specific procedural queries).
    has_semantic = any(d <= config.DISTANCE_THRESHOLD for _, _, d in sem_hits.values())
    if not has_semantic and not named_docs:
        return None

    is_value_query = bool(_VALUE_QUERY_RE.search(question))

    all_bm25_sorted, bm25_filtered = bm25_index.search_with_folder(question, folder=folder_filter)
    all_bm25 = dict(all_bm25_sorted)
    # Value queries cast a wider BM25 net — the chunk with the exact number may
    # rank lower than conceptual chunks in semantic search, so more BM25 candidates
    # increases the chance of pulling it into context.
    bm25_k = config.RETRIEVAL_K * 2 if is_value_query else config.RETRIEVAL_K
    bm25_top = [cid for cid, score in bm25_filtered[:bm25_k] if score > 0]

    missing = [cid for cid in bm25_top if cid not in sem_hits]
    if missing:
        fetched = col.get(ids=missing, include=["documents", "metadatas"])
        for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
            sem_hits[cid] = (doc, meta, 999.0)

    sem_ranked = [cid for cid, _ in sorted(sem_hits.items(), key=lambda x: x[1][2])]
    # For value queries, weight BM25 at 2× semantic so exact token hits dominate.
    rrf_weights = [1.0, 2.0] if is_value_query else None
    rrf_scores = _rrf([sem_ranked, bm25_top], weights=rrf_weights)

    named_doc_set: set[str] = set(named_docs)

    if named_docs:
        _inject_named_docs(named_docs, sem_hits, rrf_scores, col, folder_filter, all_bm25)

    sys_names = _SYS_REF.findall(question)
    if sys_names:
        _inject_system_names(sys_names, sem_hits, rrf_scores, named_doc_set, col)

    for cid in rrf_scores:
        if cid.endswith("_1_-1"):
            rrf_scores[cid] *= _SUMMARY_BOOST

    if is_value_query:
        for cid in rrf_scores:
            if cid in sem_hits and sem_hits[cid][1].get("table_id"):
                rrf_scores[cid] *= 1.5

    if _COMPARISON_RE.search(question):
        await _apply_comparison_boost(question, sem_hits, rrf_scores, col, folder_filter, sem_where)

    if config.RERANK_ENABLED:
        top_n = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:config.RERANK_TOP_N]
        candidates = [(cid, sem_hits[cid][0]) for cid in top_n if cid in sem_hits]
        order = reranker.rerank(question, candidates)
        for new_rank, orig_idx in enumerate(order):
            rrf_scores[candidates[orig_idx][0]] = 1.0 / (_RRF_K + new_rank + 1)

    top_ids = _diverse_top(rrf_scores, sem_hits, named_doc_set=named_doc_set)

    # Table completeness: if any top chunk is tagged as part of a table, pull in
    # every chunk from that same table so the model always sees the full table.
    table_ids_in_context: set[str] = set()
    for cid in top_ids:
        tid = sem_hits[cid][1].get("table_id", "")
        if tid:
            table_ids_in_context.add(tid)

    if table_ids_in_context:
        t_ids = list(table_ids_in_context)
        t_where = _with_folder(
            {"table_id": t_ids[0]} if len(t_ids) == 1 else {"table_id": {"$in": t_ids}},
            folder_filter,
        )
        t_result = col.get(where=t_where, include=["documents", "metadatas"])
        paired = sorted(
            zip(t_result["ids"], t_result["documents"], t_result["metadatas"]),
            key=lambda x: (x[2]["page"], x[2].get("chunk_index", 0)),
        )
        table_cids_ordered: list[str] = []
        for cid, doc, meta in paired:
            if cid not in sem_hits:
                sem_hits[cid] = (doc, meta, 995.0)
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
        nfetch = col.get(ids=list(missing_neighbors), include=["documents", "metadatas"])
        for ncid, ndoc, nmeta in zip(nfetch["ids"], nfetch["documents"], nfetch["metadatas"]):
            sem_hits[ncid] = (ndoc, nmeta, 996.0)

    context_parts, sources, seen = [], [], set()
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
        key = (meta["doc_id"], meta["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"filename": meta["filename"], "page": meta["page"]})

    return context_parts, sources


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
                 folder_filter: str | None = None, current_user: dict | None = None,
                 procedure: dict | None = None) -> dict:
    col = ingest._db()
    if col.count() == 0:
        return {"answer": _NOT_FOUND, "sources": []}
    result = await _retrieve(_retrieval_query(question, history), col,
                             folder_filter=folder_filter, current_user=current_user)
    if result is None:
        return {"answer": _NOT_FOUND, "sources": []}
    context_parts, sources = result
    reply = await llm.chat(system=_build_system(context_parts, procedure),
                           user=question, history=history)
    return {"answer": reply, "sources": sources}


async def answer_stream(question: str, history: list[dict] | None = None,
                        folder_filter: str | None = None, current_user: dict | None = None,
                        procedure: dict | None = None):
    """Async generator: yields {token} dicts then a final {sources, done} dict."""
    col = ingest._db()
    if col.count() == 0:
        yield {"token": _NOT_FOUND}
        yield {"sources": [], "done": True}
        return

    # Catalog queries bypass normal RAG — extract per-document and stream row by row.
    if _CATALOG_RE.search(question) and not procedure:
        async for chunk in _catalog_stream(question, col, folder_filter, current_user):
            yield chunk
        return

    result = await _retrieve(_retrieval_query(question, history), col,
                             folder_filter=folder_filter, current_user=current_user)
    if result is None:
        yield {"token": _NOT_FOUND}
        yield {"sources": [], "done": True}
        return
    context_parts, sources = result
    async for token in llm.chat_stream(system=_build_system(context_parts, procedure),
                                       user=question, history=history):
        yield {"token": token}
    yield {"sources": sources, "done": True}
