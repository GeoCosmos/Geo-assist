import asyncio
import json
import logging

import httpx

import config

log = logging.getLogger(__name__)

# trust_env=False is a hard air-gap requirement, not a preference. httpx honours
# HTTP_PROXY / HTTPS_PROXY / ALL_PROXY from the environment by default, so on any
# machine with a corporate proxy configured, every prompt and document excerpt
# sent to "localhost" Ollama would instead be routed through that proxy — data
# leaving the machine. It also stops httpx reading ~/.netrc and SSL_CERT_FILE.
_client    = httpx.AsyncClient(
    base_url=config.OLLAMA_BASE,
    timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
    trust_env=False,
)
_embed_sem = asyncio.Semaphore(config.EMBED_CONCURRENCY)
# Ollama serialises generation on CPU-only hardware, so firing N concurrent chat
# requests does not make them finish faster — it just multiplies peak RAM and
# makes every one of them slower. Ingest fans out to _PREPARE_CONCURRENCY=16
# files, each of which may need an LLM summary; without this bound that is 16
# simultaneous generations on a 16 GB box.
_chat_sem  = asyncio.Semaphore(config.CHAT_CONCURRENCY)


def _log_stats(chunk: dict, model: str, thinking_chars: int = 0) -> None:
    """Log Ollama's own timing counters from a final ("done") response.

    These three numbers are what separate the causes of a slow answer, and every
    one of them used to be discarded — the done chunk carries empty content, so
    the `if token:` filter in chat_stream dropped it whole:

      load_duration      large  → the model was (re)loaded for this request
      prompt_eval_count  large  → the assembled context is oversized
      eval_count >> visible tokens → the model spent the time on hidden reasoning

    Without them a 6-minute answer is indistinguishable from a hung process.
    """
    ns = 1_000_000_000
    log.info(
        "ollama %s: load=%.1fs prompt=%d tok/%.1fs eval=%d tok/%.1fs thinking=%d chars",
        model,
        chunk.get("load_duration", 0) / ns,
        chunk.get("prompt_eval_count", 0),
        chunk.get("prompt_eval_duration", 0) / ns,
        chunk.get("eval_count", 0),
        chunk.get("eval_duration", 0) / ns,
        thinking_chars,
    )
    if thinking_chars:
        # Not cosmetic: reasoning is generated at full token cost and then thrown
        # away, so this is pure latency the user waits through staring at nothing.
        log.warning(
            "%s returned %d chars of reasoning that were never shown — "
            '"think": false is not taking effect on this server/model',
            model, thinking_chars,
        )


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
            r = await _client.post("/api/embed", json={"model": config.EMBED_MODEL, "input": batch, "keep_alive": config.KEEP_ALIVE})
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
    async with _chat_sem:
        r = await _client.post(
            "/api/chat",
            json={
                "model": model or config.CHAT_MODEL,
                "stream": False,
                "keep_alive": config.KEEP_ALIVE,
                "think": False,
                "messages": _build_messages(system, user, history),
            },
        )
        r.raise_for_status()
        body    = r.json()
        message = body.get("message", {})
        _log_stats(body, model or config.CHAT_MODEL, len(message.get("thinking") or ""))
        # A reasoning model can spend its whole budget on `thinking` and return no
        # `content` at all. Callers here are ingest summaries and doc-number
        # extraction, which handle an empty string — but not a KeyError.
        return message.get("content", "")


async def chat_stream(
    system: str,
    user: str,
    model: str | None = None,
    history: list[dict] | None = None,
):
    """Async generator that yields tokens from a streaming Ollama response.

    Deliberately NOT gated by _chat_sem: this is the interactive answer path, and
    queueing a user's question behind a bulk ingest's summary generations would
    make the UI appear frozen for minutes. Only batch/background callers (which
    go through chat()) are bounded.
    """
    async with _client.stream(
        "POST",
        "/api/chat",
        json={
            "model": model or config.CHAT_MODEL,
            "stream": True,
            "keep_alive": config.KEEP_ALIVE,
            "think": False,
            "messages": _build_messages(system, user, history),
        },
    ) as r:
        r.raise_for_status()
        thinking_chars = 0
        async for line in r.aiter_lines():
            if line:
                chunk = json.loads(line)
                message = chunk.get("message", {})
                # Reasoning models return chain-of-thought in `thinking`, not
                # `content`. It is deliberately not yielded — it is not an answer —
                # but it is measured, because it is generated at full token cost and
                # is the difference between "the model is working" and "we are hung".
                thinking_chars += len(message.get("thinking") or "")
                token = message.get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    _log_stats(chunk, model or config.CHAT_MODEL, thinking_chars)


async def reachable() -> bool:
    try:
        r = await _client.get("/api/tags", timeout=10.0)
        return r.is_success
    except Exception:
        return False


async def unload(model: str) -> bool:
    """Evict a model from Ollama's memory immediately.

    `keep_alive: 0` tells Ollama to drop the model as soon as the (empty) request
    completes. Used by the benchmark between models so a multi-model sweep does
    not accumulate every model it has touched in RAM.
    """
    try:
        r = await _client.post("/api/generate", json={"model": model, "keep_alive": 0})
        return r.is_success
    except Exception:
        log.warning("could not unload %s", model, exc_info=True)
        return False


async def aclose() -> None:
    """Close the shared HTTP client. Called from the FastAPI lifespan shutdown."""
    await _client.aclose()
