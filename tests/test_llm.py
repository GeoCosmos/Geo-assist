"""Tests for the Ollama wrappers in llm.py.

These drive the HTTP layer with a mock transport rather than the `mock_ollama`
fixture, which replaces `llm.chat` wholesale and so cannot exercise the streaming
response parser.
"""
import json
import logging

import httpx

import llm

_DONE_CHUNK = {
    "message": {"content": ""},
    "done": True,
    "load_duration":        3_000_000_000,   # 3.0s
    "prompt_eval_count":    1200,
    "prompt_eval_duration": 12_000_000_000,  # 12.0s
    "eval_count":           900,
    "eval_duration":        90_000_000_000,  # 90.0s
}


def _mock_client(chunks: list[dict]) -> httpx.AsyncClient:
    """An AsyncClient that replays `chunks` as an Ollama NDJSON stream."""
    body = "\n".join(json.dumps(c) for c in chunks).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             base_url="http://mock-ollama")


async def test_chat_stream_logs_ollama_timing_counters(monkeypatch, caplog):
    """The done-chunk counters must reach the logs.

    Without these, a slow answer is indistinguishable between a cold model load,
    an oversized prompt, and reasoning tokens — which is exactly the ambiguity
    that forced a live-VM investigation.
    """
    monkeypatch.setattr(llm, "_client", _mock_client([
        {"message": {"content": "Hello"}, "done": False},
        _DONE_CHUNK,
    ]))

    with caplog.at_level(logging.INFO, logger="llm"):
        tokens = [t async for t in llm.chat_stream("sys", "user")]

    assert tokens == ["Hello"]
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "load=3.0s" in logged
    assert "prompt=1200" in logged
    assert "eval=900" in logged


async def test_chat_stream_reports_hidden_reasoning_tokens(monkeypatch, caplog):
    """Regression: reasoning tokens were generated, discarded, and never reported.

    qwen3.6:27B ignored `"think": false` and spent minutes emitting `message.thinking`,
    which the stream parser dropped because it only read `message.content`. The user
    saw a frozen UI and the logs said nothing.
    """
    think_a, think_b = "First I should consider ", "every requirement in turn."
    monkeypatch.setattr(llm, "_client", _mock_client([
        {"message": {"thinking": think_a}, "done": False},
        {"message": {"thinking": think_b}, "done": False},
        {"message": {"content": "Answer."}, "done": False},
        _DONE_CHUNK,
    ]))

    with caplog.at_level(logging.INFO, logger="llm"):
        tokens = [t async for t in llm.chat_stream("sys", "user")]

    # Reasoning must not leak into the answer stream...
    assert tokens == ["Answer."]
    # ...but it must be visible to whoever is debugging the latency.
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert f"thinking={len(think_a) + len(think_b)}" in logged
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


async def test_chat_stream_stays_quiet_when_model_does_not_reason(monkeypatch, caplog):
    """A well-behaved model must not produce the reasoning warning."""
    monkeypatch.setattr(llm, "_client", _mock_client([
        {"message": {"content": "Answer."}, "done": False},
        _DONE_CHUNK,
    ]))

    with caplog.at_level(logging.INFO, logger="llm"):
        [t async for t in llm.chat_stream("sys", "user")]

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_chat_stream_logs_counters_when_done_chunk_omits_message(monkeypatch, caplog):
    """Ollama's real final chunk may carry no `message` key at all — the counters
    still have to be recorded."""
    bare_done = {k: v for k, v in _DONE_CHUNK.items() if k != "message"}
    monkeypatch.setattr(llm, "_client", _mock_client([
        {"message": {"content": "Answer."}, "done": False},
        bare_done,
    ]))

    with caplog.at_level(logging.INFO, logger="llm"):
        tokens = [t async for t in llm.chat_stream("sys", "user")]

    assert tokens == ["Answer."]
    assert "eval=900" in " ".join(r.getMessage() for r in caplog.records)


async def test_chat_survives_a_response_with_no_visible_content(monkeypatch):
    """A reasoning model can burn its whole budget on `thinking` and return no
    `content` key at all. Observed live: one run produced zero visible tokens.

    chat() feeds ingest summaries and doc-number extraction, so a KeyError here
    would surface as a failed ingest rather than an empty answer.
    """
    monkeypatch.setattr(llm, "_client", _mock_client([
        {**_DONE_CHUNK, "message": {"thinking": "reasoned, never answered"}},
    ]))

    assert await llm.chat("sys", "user") == ""


async def test_chat_logs_counters_on_the_non_streaming_path(monkeypatch, caplog):
    """chat() feeds ingest summaries and procedure synthesis — it needs the same
    visibility as the interactive path."""
    monkeypatch.setattr(llm, "_client", _mock_client([
        {**_DONE_CHUNK, "message": {"content": "Answer."}},
    ]))

    with caplog.at_level(logging.INFO, logger="llm"):
        reply = await llm.chat("sys", "user")

    assert reply == "Answer."
    assert "eval=900" in " ".join(r.getMessage() for r in caplog.records)
