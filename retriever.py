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

# Regex fast-path for common document header fields.
_FIELD_RES = {
    "Revision":      re.compile(r'\b(?:rev(?:ision)?\.?\s*(?:no\.?|#)?|version|ver\.?)\s*[:#.]?\s*([A-Z]?\d+(?:[.\-]\d+)*[A-Za-z]?|[A-Z](?:[.\-]\d+)*)\b', re.IGNORECASE),
    "Document No":   re.compile(r'\b(?:doc(?:ument)?\.?\s*(?:no\.?|num(?:ber)?|#)|ref(?:erence)?\.?\s*(?:no\.?|#)?)\s*[:#.]?\s*([A-Z0-9][A-Z0-9\-/._]{1,40})', re.IGNORECASE),
    "Date":          re.compile(r'\b(?:date|effective\s+date|issued|approved)\s*[:#.]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{2}[/\-]\d{2}|\w+\.?\s+\d{1,2},?\s+\d{4})', re.IGNORECASE),
    "Software Ver":  re.compile(r'\b(?:software\s+(?:version|ver\.?|release)|sw\s+(?:version|ver\.?))\s*[:#.]?\s*([A-Z0-9][A-Z0-9\-._]{0,30})', re.IGNORECASE),
    "Hardware":      re.compile(r'\b(?:hardware\s+(?:version|ver\.?|revision|rev\.?)|hw\s+(?:version|ver\.?))\s*[:#.]?\s*([A-Z0-9][A-Z0-9\-._]{0,30})', re.IGNORECASE),
}

_CATALOG_EXTRACT_SYSTEM = """\
You are extracting specific information from a single document excerpt.
Return ONLY the values requested, one per line, in exactly this format:
  Field: value
If a field is not present in the text write "Field: —".
No other text, no explanations.\
"""


def _regex_extract(text: str) -> dict[str, str]:
    """Pull common header fields from text via regex. Returns only matched fields."""
    found = {}
    for label, pattern in _FIELD_RES.items():
        m = pattern.search(text)
        if m:
            found[label] = m.group(1).strip()
    return found


async def _llm_extract(filename: str, text: str, fields_needed: list[str]) -> dict[str, str]:
    """Ask the LLM to extract specific fields from a single document's text."""
    field_list = "\n".join(f"  {f}:" for f in fields_needed)
    prompt = (
        f"Document: {filename}\n\n"
        f"Text:\n{text[:1200]}\n\n"
        f"Extract these fields:\n{field_list}"
    )
    try:
        raw = await llm.chat(system=_CATALOG_EXTRACT_SYSTEM, user=prompt)
        result = {}
        for line in raw.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if key in fields_needed and val and val != "—":
                    result[key] = val
        return result
    except Exception:
        return {}


async def _catalog_stream(
    question: str,
    col,
    folder_filter: str | None,
    current_user: dict | None,
):
    """Per-document extraction loop for catalog queries.

    Tries regex first (instant). Falls back to a single focused LLM call per
    document for fields not found by regex. Streams each table row as it
    completes so the user sees progress rather than a blank screen for 3 minutes.
    """
    docs = ingest.list_documents()
    if folder_filter:
        docs = [d for d in docs if d.get("folder") == folder_filter]
    if not docs:
        yield {"token": _NOT_FOUND}
        yield {"sources": [], "done": True}
        return

    # Ask the LLM once to decide which columns to build from the question.
    try:
        col_raw = await llm.chat(
            system="Reply with ONLY a comma-separated list of column names to extract from documents based on the user's request. Maximum 5 columns. No explanations.",
            user=question,
        )
        columns = [c.strip().title() for c in col_raw.split(",") if c.strip()][:5]
    except Exception:
        columns = ["Revision", "Document No", "Date"]

    # Always include filename as first column
    header = "| File | " + " | ".join(columns) + " |"
    sep    = "| --- | " + " | ".join(["---"] * len(columns)) + " |"
    yield {"token": header + "\n" + sep + "\n"}

    sources = []
    for doc in docs:
        filename = doc["doc_id"] and doc["filename"]
        where = _with_access(_with_folder({"doc_id": doc["doc_id"]}, folder_filter), current_user)
        fetched = col.get(where=where, include=["documents", "metadatas"])
        if not fetched["ids"]:
            continue

        # First 3 chunks from page 1
        page1 = sorted(
            [(t, m) for t, m in zip(fetched["documents"], fetched["metadatas"])
             if m.get("chunk_type") != "image" and m.get("page") == 1 and m.get("chunk_index", 0) >= 0],
            key=lambda x: x[1].get("chunk_index", 0),
        )[:3]
        combined = "\n".join(t for t, _ in page1) if page1 else ""
        if not combined:
            combined = "\n".join(t for t in fetched["documents"][:2] if t)

        # Regex first — instant for common fields
        found = _regex_extract(combined)

        # LLM fallback for anything regex missed
        missing = [c for c in columns if c not in found]
        if missing:
            llm_found = await _llm_extract(filename, combined, missing)
            found.update(llm_found)

        row_cells = [found.get(c, "—") for c in columns]
        row = f"| {filename} | " + " | ".join(row_cells) + " |"
        yield {"token": row + "\n"}
        sources.append({"filename": filename, "page": 1})

    yield {"sources": sources, "done": True}

log = logging.getLogger(__name__)

_RRF_K = 60


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
You are Geo-Assist, a senior technical analyst operating in a secure, air-gapped \
environment. Your role is not to summarise documents — it is to REASON across them \
and give the user insights, comparisons, and recommendations they could not easily \
extract themselves.

HARD CONSTRAINTS — never break these:
- Every fact, figure, and technical claim must be traceable to the retrieved context \
  below. Never use knowledge from your training data for factual assertions.
- If the context does not contain what is needed, say exactly: \
  "That information is not in the ingested documents." Do not guess or approximate numbers.

Read all retrieved context blocks before writing a single word. Each block is \
labelled with its source file and page at the top — that label determines which \
document the text belongs to. Never attribute content from one labelled block to \
a different document.

{figure_note}When the question names a specific document, use only the blocks labelled with \
that document. Ignore blocks from other documents for that answer.

For specific values, names, or numbers: quote or paraphrase the exact text that \
supports the claim. If you cannot point to a specific passage, say "I cannot find \
that in the retrieved text" — never fill the gap with a guess or a value from \
your training knowledge.

If the retrieved text contains contradictory information within the same document \
(e.g. one section says one thing, a table says another), point out the contradiction \
explicitly rather than picking one silently.

When the question asks for a recommendation or next action, derive it from the \
technical data in the retrieved blocks: measurements, test results, margins, \
findings. Do not look for a section explicitly labelled "Recommendations" — reason \
from the content itself.

When asked for a combined or total value, perform the arithmetic and show your \
working step by step, stating which document each number came from. For mass or \
weight, use the primary total system mass value, not partial masses such as \
structural mass, propellant mass, or mass flow rate.

When context gives both a nominal value and a tolerance range for the same \
parameter, state the nominal first and the range second.

When walking through a procedure, include every explicit timing value — do not \
omit wait or hold durations.

When the user asks to compare, contrast, rank, or tabulate multiple items \
(e.g. "compare X and Y", "list the specs of A, B, C", "what are the differences between…"), \
format the response as a markdown table with clear column headers. \
Use one row per item and one column per attribute being compared. \
Add a brief prose summary after the table if the comparison needs context.

After each key factual claim, cite the source file and page in parentheses. \
If you draw a logical inference from stated facts, say so explicitly before stating it. \
If the retrieved context does not contain what is needed to answer, say so — \
do not fabricate numbers or conclusions.

Always respond in the same language the user used in their question. \
If the user writes in Russian, respond in Russian. \
If the user writes in Armenian, respond in Armenian. \
The retrieved context may be in English — that is fine; translate as needed in your response. \
If you are not confident in a translation of a technical term, include the English term in parentheses.

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
- Every fact must be traceable to the procedure step text or the retrieved context below.
- Never fabricate values, steps, or specifications.
- If the retrieved context does not contain what is needed, say exactly: \
  "That information is not in the ingested documents."
- If retrieved context contradicts itself, flag all contradictions explicitly.

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
                           q_vec: list[float] | None = None) -> dict[str, tuple[str, dict, float]]:
    """Single semantic query → {chunk_id: (doc, meta, distance)}."""
    if q_vec is None:
        [q_vec] = await llm.embed([query], prefix="search_query")
    n = min(k, col.count())
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


def _rrf(ranked_lists: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk IDs."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
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
    question: str,
    sem_hits: dict,
    rrf_scores: dict,
    col,
    folder_filter: str | None,
) -> dict[str, float]:
    """Guarantee top BM25-ranked chunks from explicitly named documents appear in context.

    Returns the full BM25 score map (reused by system-name injection if called next).
    """
    all_bm25 = {cid: score for cid, score in bm25_index.search(question)}
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
    return all_bm25


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

    context_parts: list[str] = []
    sources: list[dict] = []
    seen: set = set()

    for doc in docs[:15]:
        doc_id = doc["doc_id"]
        where = _with_access(_with_folder({"doc_id": doc_id}, folder_filter), current_user)
        fetched = col.get(where=where, include=["documents", "metadatas"])
        if not fetched["ids"]:
            continue

        text_chunks = [
            (cid, text, meta)
            for cid, text, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
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

    if config.QUERY_EXPANSION:
        # Start embedding the original query immediately so it overlaps with the LLM
        # expansion call, then batch-embed all variants in a single HTTP request.
        orig_embed_task = asyncio.create_task(llm.embed([question], prefix="search_query"))
        variants = await _expand_query(question)
        log.info("queries: %s", variants)
        extra_vecs = await llm.embed(variants[1:], prefix="search_query") if variants[1:] else []
        [orig_vec] = await orig_embed_task
        sem_results = list(await asyncio.gather(
            _semantic_search(col, question, config.RETRIEVAL_K, where=sem_where, q_vec=orig_vec),
            *[_semantic_search(col, q, config.RETRIEVAL_K, where=sem_where, q_vec=v)
              for q, v in zip(variants[1:], extra_vecs)]
        ))
    else:
        log.info("queries: %s (expansion disabled)", [question])
        sem_results = [await _semantic_search(col, question, config.RETRIEVAL_K, where=sem_where)]

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

    bm25_results = bm25_index.search(question, folder=folder_filter)
    bm25_top = [cid for cid, score in bm25_results[:config.RETRIEVAL_K] if score > 0]

    missing = [cid for cid in bm25_top if cid not in sem_hits]
    if missing:
        fetched = col.get(ids=missing, include=["documents", "metadatas"])
        for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
            sem_hits[cid] = (doc, meta, 999.0)

    sem_ranked = [cid for cid, _ in sorted(sem_hits.items(), key=lambda x: x[1][2])]
    rrf_scores = _rrf([sem_ranked, bm25_top])

    named_doc_set: set[str] = set(named_docs)

    all_bm25: dict[str, float] = {}
    if named_docs:
        all_bm25 = _inject_named_docs(named_docs, question, sem_hits, rrf_scores, col, folder_filter)

    sys_names = _SYS_REF.findall(question)
    if sys_names:
        if not all_bm25:
            all_bm25 = {cid: score for cid, score in bm25_index.search(question)}
        _inject_system_names(sys_names, sem_hits, rrf_scores, named_doc_set, col)

    for cid in rrf_scores:
        if cid.endswith("_1_-1"):
            rrf_scores[cid] *= _SUMMARY_BOOST

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
        table_cids_ordered: list[str] = []
        for tid in table_ids_in_context:
            t_result = col.get(where=_with_folder({"table_id": tid}, folder_filter), include=["documents", "metadatas"])
            paired = sorted(
                zip(t_result["ids"], t_result["documents"], t_result["metadatas"]),
                key=lambda x: (x[2]["page"], x[2].get("chunk_index", 0)),
            )
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

    context_parts, sources, seen = [], [], set()
    for cid in top_ids:
        doc, meta, _ = sem_hits[cid]
        context_parts.append(f"[{meta['filename']}, page {meta['page']}]\n{doc}")
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
    if procedure:
        steps = procedure["steps"]
        idx = procedure["step_idx"]
        return _PROCEDURE_SYSTEM.format(
            filename=procedure["filename"],
            step_num=idx + 1,
            total_steps=len(steps),
            step_text=steps[idx] if steps else "(no steps found)",
            figure_note=_FIGURE_NOTE,
            context=context,
        )
    return _SYSTEM.format(context=context, figure_note=_FIGURE_NOTE)


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
