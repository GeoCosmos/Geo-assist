import logging
import config

log = logging.getLogger(__name__)
_model = None
_available: bool | None = None  # None = not yet checked; False = permanently unavailable


def _get_model():
    global _model, _available
    if _available is False:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANK_MODEL, max_length=512)
        _available = True
        log.info("reranker loaded: %s", config.RERANK_MODEL)
    except ImportError:
        _available = False
        log.warning(
            "sentence-transformers not installed — reranking disabled. "
            "Install: pip install -r requirements-reranker.txt  "
            "then pre-download (needs internet once): "
            "python3 -c \"from sentence_transformers import CrossEncoder; "
            "CrossEncoder('%s')\"",
            config.RERANK_MODEL,
        )
    except Exception as exc:
        _available = False
        log.warning("reranker unavailable (%s) — falling back to RRF order", exc)
    return _model


def rerank(query: str, candidates: list[tuple[str, str]]) -> list[int]:
    """Score (chunk_id, text) pairs with a cross-encoder.

    Returns indices into candidates sorted by relevance score descending.
    Falls back to original order if the model is unavailable or prediction fails.
    """
    if not candidates:
        return []
    model = _get_model()
    if model is None:
        return list(range(len(candidates)))
    try:
        pairs = [(query, text) for _, text in candidates]
        scores = model.predict(pairs)
        return sorted(range(len(scores)), key=lambda i: -scores[i])
    except Exception:
        log.warning("reranking failed, falling back to RRF order", exc_info=True)
        return list(range(len(candidates)))
