"""In-memory BM25 index over all ingested chunks, persisted to disk between restarts.

BM25 (Best Match 25) ranks documents by keyword relevance using two improvements
over TF-IDF:
  - Term frequency saturation: a word appearing 100x is not 100x more relevant
    than one appearing once — relevance grows but plateaus (controlled by k1).
  - Document length normalisation: longer docs naturally have more occurrences;
    BM25 divides by a length factor so short, keyword-dense chunks score higher
    (controlled by b, default 0.75).

Used alongside semantic search because embeddings miss exact tokens (part numbers,
acronyms, version strings). RRF fuses both rankings without score normalisation.

Why not Qdrant sparse vectors
─────────────────────────────
Qdrant supports sparse vectors natively and Haystack exposes a sparse retriever
for them, which would fold keyword search into the same store. It is deliberately
not used: the sparse encoders (FastEmbed BM25/BM42) require downloading an ONNX
model on first use. That is the same "needs internet once" caveat that already
makes OCR and the cross-encoder awkward to deploy, and keyword search is the one
part of retrieval that must never silently degrade on an air-gapped machine.
`rank_bm25` is pure Python with no model to fetch.

Incrementality
──────────────
The previous implementation re-fetched every chunk from the vector store, re-ran
the Snowball stemmer over the whole corpus, and rebuilt the index from scratch on
*every single ingest and delete*. Ingesting 500 files one at a time meant 500 full
rebuilds — O(n×batches), hours on a 148k-chunk corpus.

Now the tokenised corpus is held in memory and persisted alongside the index, so
adding or removing a document only tokenises the chunks that actually changed.
Re-instantiating BM25Okapi from cached token lists is still O(n), but it is a
cheap array pass compared to re-stemming millions of words. `rebuild_from_store`
remains as a cold-start fallback.
"""
import heapq
import logging
import os
import pickle
import re

from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi

import config

log = logging.getLogger(__name__)

_stemmer = SnowballStemmer("english")
# Match decimal numbers (e.g. "3.7", "59.473") as a single token before falling
# back to regular word tokens. Without this, "3.7" splits into ["3", "7"] and
# exact numeric matching in BM25 becomes useless.
_TOKEN_RE = re.compile(r"\d+\.\d+|\b\w+\b")

# Bumped whenever the on-disk layout changes so a stale pickle is discarded
# rather than silently loaded into a mismatched structure.
_FORMAT_VERSION = 2


def _tokenize(text: str) -> list[str]:
    return [
        t if t[0].isdigit() else (_stemmer.stem(t) if t.isascii() else t)
        for t in _TOKEN_RE.findall(text.lower())
    ]


class BM25Index:
    """Keyword index over chunk IDs.

    Holds three parallel structures keyed by position: `_ids`, `_corpus` (token
    lists) and `_folders`. Deletions mark positions as tombstones rather than
    compacting immediately, so removing one document out of thousands does not
    reshuffle every list.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._folders: dict[str, str] = {}   # chunk_id → folder name
        self._pos: dict[str, int] = {}       # chunk_id → index into _ids/_corpus
        self._live_ids: list[str] = []       # chunk IDs in BM25Okapi row order
        self._dirty = False

    # ── construction ──────────────────────────────────────────────────────────

    def build(self, ids: list[str], texts: list[str], folders: dict[str, str] | None = None) -> None:
        """Replace the whole index. Used on cold start and by tests."""
        self._ids = list(ids)
        self._corpus = [_tokenize(t) for t in texts]
        self._folders = dict(folders or {})
        self._pos = {cid: i for i, cid in enumerate(self._ids)}
        self._refit()

    def _refit(self) -> None:
        live = [toks for toks in self._corpus if toks is not None]
        self._bm25 = BM25Okapi(live) if live else None
        # BM25Okapi indexes only live rows, so map its positions back to chunk IDs.
        self._live_ids = [cid for cid, toks in zip(self._ids, self._corpus) if toks is not None]
        self._dirty = False

    def add(self, ids: list[str], texts: list[str], folders: dict[str, str] | None = None) -> None:
        """Add or replace chunks. Existing IDs are updated in place."""
        folders = folders or {}
        for cid, text in zip(ids, texts):
            toks = _tokenize(text)
            existing = self._pos.get(cid)
            if existing is not None:
                self._corpus[existing] = toks
            else:
                self._pos[cid] = len(self._ids)
                self._ids.append(cid)
                self._corpus.append(toks)
            if cid in folders:
                self._folders[cid] = folders[cid]
        self._dirty = True

    def remove(self, ids: list[str]) -> None:
        """Tombstone chunks. Positions are reclaimed on the next compaction."""
        for cid in ids:
            pos = self._pos.pop(cid, None)
            if pos is not None:
                self._corpus[pos] = None  # type: ignore[call-overload]
            self._folders.pop(cid, None)
        self._dirty = True

    def remove_prefix(self, prefix: str) -> int:
        """Remove every chunk whose ID starts with `prefix` (i.e. one document)."""
        doomed = [cid for cid in self._pos if cid.startswith(prefix)]
        self.remove(doomed)
        return len(doomed)

    def commit(self) -> None:
        """Re-fit BM25 if anything changed. Call once after a batch of writes,
        never once per file — that is the O(n×batches) trap this module exists
        to avoid."""
        if self._dirty:
            self._compact()
            self._refit()

    def _compact(self) -> None:
        if not any(toks is None for toks in self._corpus):
            return
        kept = [(cid, toks) for cid, toks in zip(self._ids, self._corpus) if toks is not None]
        self._ids = [cid for cid, _ in kept]
        self._corpus = [toks for _, toks in kept]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        # Write-then-rename: a crash mid-write leaves the previous index intact
        # instead of a truncated pickle that fails to load on next boot.
        with open(tmp, "wb") as f:
            pickle.dump(
                {
                    "version": _FORMAT_VERSION,
                    "ids": self._ids,
                    "corpus": self._corpus,
                    "folders": self._folders,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, path)
        log.debug("BM25 index saved (%d chunks) → %s", len(self._ids), path)

    def load(self, path: str) -> bool:
        """Load from disk. Returns True on success, False if missing or stale."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("version") != _FORMAT_VERSION:
                log.info("BM25 index format changed — rebuilding")
                return False
            self._ids = data["ids"]
            self._corpus = data["corpus"]
            self._folders = data.get("folders", {})
            self._pos = {cid: i for i, cid in enumerate(self._ids)}
            self._refit()
            log.info("BM25 index loaded (%d chunks) ← %s", len(self._ids), path)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:  # corrupt pickle must not be fatal
            log.warning("BM25 index load failed (%s); will rebuild", exc)
            return False

    # ── search ────────────────────────────────────────────────────────────────

    def _scored(self, query: str) -> list[tuple[str, float]] | None:
        if not self._bm25 or not self._ids:
            return None
        tokens = _tokenize(query)
        if not tokens:
            return None
        scores = self._bm25.get_scores(tokens)
        # Skip rows tombstoned since the last commit — searching between a delete
        # and the next commit must not resurrect removed chunks.
        return [
            (cid, s) for cid, s in zip(self._live_ids, scores) if cid in self._pos
        ]

    def search(self, query: str, folder: str | None = None,
               top_k: int | None = None) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) best-first, optionally folder-filtered.

        `top_k` uses a heap instead of sorting the full corpus. At 148k chunks a
        full sort per query costs ~100 ms for a top-30 result that needs none of it.
        """
        pairs = self._scored(query)
        if pairs is None:
            return []
        if folder:
            pairs = [(cid, s) for cid, s in pairs if self._folders.get(cid) == folder]
        if top_k is not None:
            return heapq.nlargest(top_k, pairs, key=lambda x: x[1])
        return sorted(pairs, key=lambda x: -x[1])

    def search_with_folder(
        self, query: str, folder: str | None = None, top_k: int | None = None
    ) -> tuple[dict[str, float], list[tuple[str, float]]]:
        """Score once; return (all_scores_map, folder_filtered_top_k).

        The full ranking was previously returned as a sorted list purely so the
        caller could index into it as a map — it is now returned as a dict, which
        is what every caller actually wanted, and skips a full corpus sort.
        """
        pairs = self._scored(query)
        if pairs is None:
            return {}, []
        all_scores = dict(pairs)
        candidates = (
            [(cid, s) for cid, s in pairs if self._folders.get(cid) == folder]
            if folder else pairs
        )
        if top_k is not None:
            filtered = heapq.nlargest(top_k, candidates, key=lambda x: x[1])
        else:
            filtered = sorted(candidates, key=lambda x: -x[1])
        return all_scores, filtered

    def update_folders(self, updates: dict[str, str]) -> None:
        """Patch folder metadata for specific chunks without a re-fit."""
        self._folders.update(updates)

    @property
    def size(self) -> int:
        return len(self._pos)


# module-level singleton
_index = BM25Index()


def search(query: str, folder: str | None = None, top_k: int | None = None) -> list[tuple[str, float]]:
    return _index.search(query, folder=folder, top_k=top_k)


def search_with_folder(
    query: str, folder: str | None = None, top_k: int | None = None
) -> tuple[dict[str, float], list[tuple[str, float]]]:
    return _index.search_with_folder(query, folder=folder, top_k=top_k)


def update_folders(updates: dict[str, str]) -> None:
    _index.update_folders(updates)


def size() -> int:
    return _index.size


def add(ids: list[str], texts: list[str], folders: dict[str, str] | None = None) -> None:
    _index.add(ids, texts, folders)


def remove_doc(doc_id: str) -> int:
    return _index.remove_prefix(f"{doc_id}_")


def commit(persist: bool = True) -> None:
    """Re-fit and persist. Call once per ingest batch, not once per file."""
    _index.commit()
    if persist:
        _index.save(config.BM25_PATH)


def clear() -> None:
    _index.build([], [])
    _index.save(config.BM25_PATH)


async def rebuild_from_store() -> None:
    """Cold-start fallback: re-read every chunk from Qdrant and rebuild.

    Only used when the persisted index is missing, corrupt, or a stale format.
    This is the expensive path — it streams the entire corpus — so it must never
    be on the ingest hot path.
    """
    import store

    log.info("BM25 rebuild from store — this reads the whole corpus")
    hits = await store.get_by_filter(None)
    ids, texts, folders = [], [], {}
    for cid, (text, meta, _dist) in hits.items():
        ids.append(cid)
        # Prepend filename so doc-specific queries ("GCA-2024-0002 recommendation")
        # match every chunk from that file, not just chunks that repeat the title.
        texts.append(f"[{meta.get('filename', '')}] {text}")
        folders[cid] = meta.get("folder", "General")
    _index.build(ids, texts, folders)
    _index.save(config.BM25_PATH)
    log.info("BM25 rebuild done (%d chunks)", len(ids))


async def load_or_rebuild() -> None:
    """Load the persisted index; rebuild from the store if missing or corrupt."""
    if not _index.load(config.BM25_PATH):
        await rebuild_from_store()
