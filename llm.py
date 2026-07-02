import asyncio
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


def _build_messages(system: str, user: str, history: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


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
            "think": False,
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
            "think": False,
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


async def reachable() -> bool:
    try:
        r = await _client.get("/api/tags", timeout=10.0)
        return r.is_success
    except Exception:
        return False
