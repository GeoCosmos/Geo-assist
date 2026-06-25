import asyncio
import base64
import json
import logging
import httpx
import config

log = logging.getLogger(__name__)

_client    = httpx.AsyncClient(
    base_url=config.OLLAMA_BASE,
    timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
)
_embed_sem = asyncio.Semaphore(config.EMBED_CONCURRENCY)
_vision_sem = asyncio.Semaphore(config.VISION_CONCURRENCY)


def _build_messages(system: str, user: str, history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages

_VISION_PROMPT = (
    "You are analyzing an image extracted from a document. "
    "Describe all content precisely: diagrams, schematics, block diagrams, "
    "wiring diagrams, circuit diagrams, flow charts, plots, graphs, data tables, "
    "figures, measurements, labels, dimensions, part numbers, callouts, annotations, "
    "photos, illustrations, and any text visible in the image. "
    "If it is a chart or plot, describe the axes, units, and key data trends. "
    "If it is a table, transcribe the data row by row. "
    "Be complete and precise — this description will be used for document Q&A."
)


async def embed(texts: list[str], prefix: str = "search_document") -> list[list[float]]:
    # nomic-embed-text is trained with task prefixes; omitting them degrades accuracy
    prefixed = [f"{prefix}: {t}" for t in texts]
    batches  = [prefixed[i : i + config.EMBED_BATCH] for i in range(0, len(prefixed), config.EMBED_BATCH)]

    async def _embed_batch(batch: list[str]) -> list[list[float]]:
        async with _embed_sem:
            r = await _client.post("/api/embed", json={"model": config.EMBED_MODEL, "input": batch, "keep_alive": -1})
            r.raise_for_status()
            return r.json()["embeddings"]

    results = await asyncio.gather(*[_embed_batch(b) for b in batches])
    return [emb for batch_result in results for emb in batch_result]


async def chat(
    system: str,
    user: str,
    model: str | None = None,
    history: list[dict] | None = None,
) -> str:
    r = await _client.post(
        "/api/chat",
        json={
            "model": model or config.CHAT_MODEL,
            "stream": False,
            "keep_alive": -1,
            "messages": _build_messages(system, user, history),
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


async def chat_stream(
    system: str,
    user: str,
    model: str | None = None,
    history: list[dict] | None = None,
):
    """Async generator that yields tokens from a streaming Ollama response."""
    async with _client.stream(
        "POST",
        "/api/chat",
        json={
            "model": model or config.CHAT_MODEL,
            "stream": True,
            "keep_alive": -1,
            "messages": _build_messages(system, user, history),
        },
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token


async def analyze_image(image_bytes: bytes, filename: str, page: int) -> str | None:
    """Describe an engineering image using the local vision model.

    Returns None if VISION_MODEL is not configured or the call fails.
    """
    if not config.VISION_MODEL:
        return None
    b64 = base64.b64encode(image_bytes).decode()
    prompt = f"Document: '{filename}', page {page}.\n\n{_VISION_PROMPT}"
    async with _vision_sem:
        try:
            r = await _client.post(
                "/api/chat",
                json={
                    "model": config.VISION_MODEL,
                    "stream": False,
                    "keep_alive": -1,
                    "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                },
                timeout=httpx.Timeout(connect=10.0, read=config.VISION_TIMEOUT,
                                      write=10.0, pool=10.0),
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception:
            log.warning("image analysis failed for %s page %d", filename, page, exc_info=True)
            return None


async def reachable() -> bool:
    try:
        r = await _client.get("/api/tags", timeout=10.0)
        return r.is_success
    except Exception:
        return False
