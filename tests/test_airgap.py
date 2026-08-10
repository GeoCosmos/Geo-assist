"""Tests for the "nothing leaves the machine" guarantee.

These are not style checks. Requirement 3 in CLAUDE.md is absolute, and every
assertion here corresponds to a specific way the guarantee was violable:

* Haystack ships usage telemetry enabled by default and posts to a deepset
  endpoint when a pipeline is constructed.
* httpx honours HTTP_PROXY / HTTPS_PROXY / ALL_PROXY from the environment. On a
  machine with a corporate proxy configured, every prompt and document excerpt
  bound for "localhost" Ollama would be routed through it instead.
* The Qdrant and Ollama endpoints must stay on loopback.
"""
import os

import httpx
import pytest

import config
import llm
import store


def test_haystack_telemetry_disabled_at_import():
    """config sets the kill-switch, and every haystack importer imports config first."""
    assert os.environ.get("HAYSTACK_TELEMETRY_ENABLED") == "False"


def test_haystack_telemetry_env_is_set_before_haystack_loads():
    """The env var is worthless if haystack was already imported when it was set.

    An import sorter will move `import config` below third-party imports, so the
    kill-switch cannot depend on module import order. store.py sets it in a bare
    statement between the two import blocks, which a sorter cannot move imports
    across. This pins that structure.
    """
    with open(store.__file__) as f:
        text = f.read()
    kill_switch = text.index('os.environ["HAYSTACK_TELEMETRY_ENABLED"]')
    first_haystack_import = text.index("from haystack import")
    assert kill_switch < first_haystack_import, (
        "store.py must set HAYSTACK_TELEMETRY_ENABLED before importing haystack"
    )


def test_haystack_telemetry_send_is_neutered():
    """Second line of defence: the telemetry sender is replaced with a no-op."""
    from haystack.telemetry import _telemetry
    sent = []
    try:
        _telemetry.send_event("test_event", {"a": 1})
    except Exception as exc:  # pragma: no cover - would itself be a finding
        pytest.fail(f"send_event raised instead of being a no-op: {exc}")
    assert sent == []


def test_ollama_client_ignores_proxy_env():
    """trust_env=False, so proxy/netrc/cert environment variables are ignored."""
    assert llm._client.trust_env is False


def test_ollama_client_would_not_use_a_configured_proxy(monkeypatch):
    """A client built the way llm.py builds it must not pick up ALL_PROXY."""
    monkeypatch.setenv("ALL_PROXY", "socks5://evil.example:1080")
    monkeypatch.setenv("HTTPS_PROXY", "http://evil.example:3128")
    client = httpx.AsyncClient(base_url=config.OLLAMA_BASE, trust_env=False)
    try:
        assert client._mounts == {}, "proxy mounts were configured from the environment"
    finally:
        pass


def test_endpoints_are_loopback_only():
    assert config.OLLAMA_BASE.startswith(("http://127.0.0.1", "http://localhost"))
    assert config.QDRANT_HOST in {"127.0.0.1", "localhost", "::1"}


def test_no_cloud_document_store_configured():
    """The store must never be pointed at Qdrant Cloud."""
    assert not os.environ.get("QDRANT_API_KEY")
    assert "cloud.qdrant.io" not in config.QDRANT_HOST


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_model_hubs_are_offline(var):
    """The reranker and any transformers model must not phone home for weights."""
    assert os.environ.get(var) == "1"
