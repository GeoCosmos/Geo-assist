"""
In-memory BM25 index over all ingested chunks, persisted to disk between restarts.

BM25 (Best Match 25) ranks documents by keyword relevance using two improvements
over TF-IDF:
  - Term frequency saturation: a word appearing 100x is not 100x more relevant
    than one appearing once — relevance grows but plateaus (controlled by k1).
  - Document length normalisation: longer docs naturally have more occurrences;
    BM25 divides by a length factor so short, keyword-dense chunks score higher
    (controlled by b, default 0.75).

Used alongside semantic search because embeddings miss exact tokens (part numbers,
acronyms, version strings). RRF fuses both rankings without score normalisation.

Persistence: the built index is pickled to BM25_PATH after every rebuild so
server restarts load instantly instead of re-tokenising the full corpus.
BM25 has no incremental API, so a full rebuild is still required after every
ingest or delete — but that rebuild is now also saved.
"""
import logging
import os
import pickle
import re

from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi

import config

log = logging.getLogger(__name__)

_stemmer = SnowballStemmer("english")


def _tokenize(text: str) -> list[str]:
    return [_stemmer.stem(t) for t in re.findall(r"\b\w+\b", text.lower())]


class BM25Index:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._folders: dict[str, str] = {}  # chunk_id → folder name

    def build(self, ids: list[str], texts: list[str], folders: dict[str, str] | None = None) -> None:
        self._ids = list(ids)
        self._folders = folders or {}
        corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self._ids, "bm25": self._bm25, "folders": self._folders}, f)
        log.debug("BM25 index saved (%d chunks) → %s", len(self._ids), path)

    def load(self, path: str) -> bool:
        """Load from disk. Returns True on success, False if missing or corrupt."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._ids = data["ids"]
            self._bm25 = data["bm25"]
            self._folders = data.get("folders", {})
            log.debug("BM25 index loaded (%d chunks) ← %s", len(self._ids), path)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            log.warning("BM25 index load failed (%s); will rebuild from ChromaDB", exc)
            return False

    def search(self, query: str, folder: str | None = None) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) sorted best-first. Optionally filter by folder."""
        if not self._bm25 or not self._ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        pairs = zip(self._ids, scores)
        if folder:
            pairs = ((cid, s) for cid, s in pairs if self._folders.get(cid) == folder)
        return sorted(pairs, key=lambda x: -x[1])

    @property
    def size(self) -> int:
        return len(self._ids)


# module-level singleton
_index = BM25Index()


def search(query: str, folder: str | None = None) -> list[tuple[str, float]]:
    return _index.search(query, folder=folder)


def size() -> int:
    return _index.size


def rebuild(col) -> None:
    """Rebuild BM25 from ChromaDB and persist to disk."""
    result = col.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    texts = result.get("documents") or []
    metas = result.get("metadatas") or []
    log.info("BM25 rebuild: %d chunks", len(ids))
    # Prepend filename so doc-specific queries ("GCA-2024-0002 recommendation")
    # match every chunk from that file, not just chunks that repeat the title.
    augmented = [f"[{m.get('filename', '')}] {t}" for t, m in zip(texts, metas)]
    folders = {cid: m.get("folder", "General") for cid, m in zip(ids, metas)}
    _index.build(ids, augmented, folders)
    _index.save(config.BM25_PATH)
    log.info("BM25 rebuild done")


def load_or_rebuild(col) -> None:
    """Load persisted index from disk; rebuild from ChromaDB if missing or corrupt."""
    if not _index.load(config.BM25_PATH):
        log.info("No persisted BM25 index found — rebuilding from ChromaDB")
        rebuild(col)
