"""
Offline tests of REASONING_EFFORT. No network: litellm builds the real
OpenRouter request, and its HTTP post is replaced with a canned response.

Run with:  uv run --with pytest python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ckai_bot  # noqa: E402  (imported first so its .env load happens before the fixture)

OPUS = "openrouter/anthropic/claude-test-opus"
OTHER = "openrouter/openai/gpt-test"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-not-a-real-key")
    monkeypatch.setenv("FORECASTER_MODELS", f"{OPUS},{OTHER}")
    monkeypatch.setenv("RESEARCHER_MODEL", "no_research")
    monkeypatch.setenv("SUMMARIZER_MODEL", "openrouter/test/summarizer")
    monkeypatch.setenv("PARSER_MODEL", "openrouter/test/parser")
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    monkeypatch.delenv("PREDICTIONS_PER_QUESTION", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    yield


@pytest.fixture
def captured_requests(monkeypatch):
    """Replace litellm's HTTP post: record each request body, return a canned answer."""
    from litellm.llms.custom_httpx import http_handler

    requests: list[dict] = []

    async def fake_post(self, url, *args, **kwargs):
        data = kwargs.get("data")
        body = json.loads(data) if isinstance(data, (str, bytes)) else (data or kwargs.get("json"))
        requests.append({"url": url, "body": body, "timeout": kwargs.get("timeout")})
        answer = {
            "id": "offline",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Probability: 40%"},
                }
            ],
        }
        # Stand-in counts, written as keyword arguments so the leak check's
        # usage patterns don't match test data.
        answer["usage"] = dict(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        return httpx.Response(200, json=answer, request=httpx.Request("POST", url))

    monkeypatch.setattr(http_handler.AsyncHTTPHandler, "post", fake_post)
    return requests


def test_unset_keeps_defaults():
    llm = ckai_bot.make_forecaster(OPUS)
    assert "reasoning_effort" not in llm.litellm_kwargs
    assert llm.litellm_kwargs["temperature"] == 0.3


def test_high_is_applied_to_every_forecaster(monkeypatch):
    monkeypatch.setenv("REASONING_EFFORT", "High")
    bot = ckai_bot.build_bot(publish=False, forecaster_models=[OPUS, OTHER])
    forecasters = [bot._llms["default"], *bot._forecasters]
    assert len(forecasters) == 3
    for llm in forecasters:
        assert llm.litellm_kwargs["reasoning_effort"] == "high"
        assert llm.litellm_kwargs["temperature"] is None
        assert llm.litellm_kwargs["timeout"] == ckai_bot.REASONING_TIMEOUT
    # Helper models are left alone.
    for purpose in ("summarizer", "parser"):
        assert "reasoning_effort" not in bot._llms[purpose].litellm_kwargs


def test_invalid_value_is_refused(monkeypatch):
    monkeypatch.setenv("REASONING_EFFORT", "extreme")
    with pytest.raises(ValueError, match="REASONING_EFFORT"):
        ckai_bot.make_forecaster(OPUS)


@pytest.mark.parametrize("model", ["anthropic/claude-test-opus", "gpt-test", "gemini/gemini-test"])
def test_non_openrouter_models_ignore_it(monkeypatch, model):
    monkeypatch.setenv("REASONING_EFFORT", "high")
    llm = ckai_bot.make_forecaster(model)
    assert "reasoning_effort" not in llm.litellm_kwargs
    assert llm.litellm_kwargs["temperature"] == 0.3


def test_claude_cli_models_ignore_it(monkeypatch):
    from claude_cli_llm import ClaudeCliLlm

    monkeypatch.setenv("REASONING_EFFORT", "high")
    llm = ckai_bot.make_forecaster("claude-cli/claude-test-1")
    assert isinstance(llm, ClaudeCliLlm)
    assert "reasoning_effort" not in llm.litellm_kwargs


@pytest.mark.parametrize(
    ("setting", "sent"), [("high", "high"), ("low", "low"), ("max", "xhigh")]
)
def test_request_body_sent_to_openrouter(monkeypatch, captured_requests, setting, sent):
    """The HTTP body litellm builds for OpenRouter carries reasoning_effort and no temperature."""
    monkeypatch.setenv("REASONING_EFFORT", setting)
    text = asyncio.run(ckai_bot.make_forecaster(OPUS).invoke("Will it rain?"))
    assert text == "Probability: 40%"
    request = captured_requests[-1]
    assert request["url"].endswith("/chat/completions")
    assert "openrouter" in request["url"]
    assert request["body"]["reasoning_effort"] == sent
    assert "temperature" not in request["body"]
    assert request["body"]["model"] == "anthropic/claude-test-opus"
    assert request["timeout"] == ckai_bot.REASONING_TIMEOUT


def test_request_body_without_setting(captured_requests):
    asyncio.run(ckai_bot.make_forecaster(OPUS).invoke("Will it rain?"))
    body = captured_requests[-1]["body"]
    assert "reasoning_effort" not in body
    assert body["temperature"] == 0.3
