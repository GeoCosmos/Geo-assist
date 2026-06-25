import logging
import config

log = logging.getLogger(__name__)
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(config.RERANK_MODEL, max_length=512)
            log.info("reranker loaded: %s", config.RERANK_MODEL)
        except ImportError:
            raise RuntimeError(
                "Reranking requires sentence-transformers. "
                "Install with: pip install -r requirements-reranker.txt  "
                "Then pre-download the model (needs internet once): "
                "python3 -c \"from sentence_transformers import CrossEncoder; "
                "CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')\""
            )
    return _model


def rerank(query: str, candidates: list[tuple[str, str]]) -> list[int]:
    """Score (chunk_id, text) pairs with a cross-encoder.

    Returns indices into candidates sorted by relevance score descending.
    Falls back to original order on any error so retrieval is never blocked.
    """
    if not candidates:
        return []
    try:
        model = _get_model()
        pairs = [(query, text) for _, text in candidates]
        scores = model.predict(pairs)
        return sorted(range(len(scores)), key=lambda i: -scores[i])
    except Exception:
        log.warning("reranking failed, falling back to RRF order", exc_info=True)
        return list(range(len(candidates)))
