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
CANNED_FINISH_REASON = ["stop"]  # what the fake OpenRouter returns


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-not-a-real-key")
    monkeypatch.setenv("FORECASTER_MODELS", f"{OPUS},{OTHER}")
    monkeypatch.setenv("RESEARCHER_MODEL", "no_research")
    monkeypatch.setenv("SUMMARIZER_MODEL", "openrouter/test/summarizer")
    monkeypatch.setenv("PARSER_MODEL", "openrouter/test/parser")
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    monkeypatch.delenv("FORECASTER_MAX_TOKENS", raising=False)
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
                    "finish_reason": CANNED_FINISH_REASON[0],
                    "message": {"role": "assistant", "content": "Probability: 40%"},
                }
            ],
        }
        # Stand-in counts, written as keyword arguments so the leak check's
        # usage patterns don't match test data.
        answer["usage"] = dict(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        return httpx.Response(200, json=answer, request=httpx.Request("POST", url))

    monkeypatch.setattr(http_handler.AsyncHTTPHandler, "post", fake_post)
    CANNED_FINISH_REASON[0] = "stop"
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


# FORECASTER_MAX_TOKENS


@pytest.mark.parametrize("effort", ["low", "medium", "high", "High"])
def test_default_cap_at_low_medium_high(monkeypatch, captured_requests, effort):
    monkeypatch.setenv("REASONING_EFFORT", effort)
    asyncio.run(ckai_bot.make_forecaster(OPUS).invoke("Will it rain?"))
    body = captured_requests[-1]["body"]
    assert body["max_tokens"] == ckai_bot.DEFAULT_MAX_TOKENS == 32000
    assert body["reasoning_effort"] == effort.lower()


@pytest.mark.parametrize("effort", [None, "none", "minimal", "xhigh", "max"])
def test_no_default_cap_elsewhere(monkeypatch, captured_requests, effort):
    if effort:
        monkeypatch.setenv("REASONING_EFFORT", effort)
    asyncio.run(ckai_bot.make_forecaster(OPUS).invoke("Will it rain?"))
    assert "max_tokens" not in captured_requests[-1]["body"]


@pytest.mark.parametrize(("effort", "sent"), [("high", "high"), ("max", "xhigh")])
def test_setting_is_sent_with_effort(monkeypatch, captured_requests, effort, sent):
    monkeypatch.setenv("REASONING_EFFORT", effort)
    monkeypatch.setenv("FORECASTER_MAX_TOKENS", "100000")
    asyncio.run(ckai_bot.make_forecaster(OPUS).invoke("Will it rain?"))
    body = captured_requests[-1]["body"]
    assert body["max_tokens"] == 100000
    assert body["reasoning_effort"] == sent
    assert "temperature" not in body


def test_setting_without_effort_keeps_temperature(monkeypatch, captured_requests):
    monkeypatch.setenv("FORECASTER_MAX_TOKENS", "8000")
    asyncio.run(ckai_bot.make_forecaster(OPUS).invoke("Will it rain?"))
    body = captured_requests[-1]["body"]
    assert body["max_tokens"] == 8000
    assert body["temperature"] == 0.3
    assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("effort", "minimum"),
    [(None, 4000), ("none", 4000), ("minimal", 4445), ("low", 5000), ("medium", 8000),
     ("high", 20000), ("xhigh", 80000), ("max", 80000)],
)
def test_minimum_leaves_room_for_the_answer(monkeypatch, effort, minimum):
    if effort:
        monkeypatch.setenv("REASONING_EFFORT", effort)
    monkeypatch.setenv("FORECASTER_MAX_TOKENS", str(minimum))
    assert ckai_bot.make_forecaster(OPUS).litellm_kwargs["max_tokens"] == minimum
    monkeypatch.setenv("FORECASTER_MAX_TOKENS", str(minimum - 1))
    with pytest.raises(ValueError, match="FORECASTER_MAX_TOKENS"):
        ckai_bot.make_forecaster(OPUS)


@pytest.mark.parametrize("value", ["abc", "0", "-5000", "1.5", "32000.0", "128001"])
def test_invalid_cap_is_refused(monkeypatch, value):
    monkeypatch.setenv("FORECASTER_MAX_TOKENS", value)
    with pytest.raises(ValueError, match="FORECASTER_MAX_TOKENS"):
        ckai_bot.make_forecaster(OPUS)


def test_upper_limit_passes(monkeypatch):
    monkeypatch.setenv("FORECASTER_MAX_TOKENS", "128000")
    assert ckai_bot.make_forecaster(OPUS).litellm_kwargs["max_tokens"] == 128000


def test_cap_only_on_openrouter_forecasters(monkeypatch, caplog):
    from claude_cli_llm import ClaudeCliLlm

    monkeypatch.setenv("REASONING_EFFORT", "high")
    monkeypatch.setenv("FORECASTER_MAX_TOKENS", "40000")
    bot = ckai_bot.build_bot(publish=False, forecaster_models=[OPUS, OTHER])
    for llm in [bot._llms["default"], *bot._forecasters]:
        assert llm.litellm_kwargs["max_tokens"] == 40000
    for purpose in ("summarizer", "parser"):
        assert "max_tokens" not in bot._llms[purpose].litellm_kwargs
    cli = ckai_bot.make_forecaster("claude-cli/claude-test-1")
    assert isinstance(cli, ClaudeCliLlm)
    assert "max_tokens" not in cli.litellm_kwargs
    with caplog.at_level("WARNING"):
        direct = ckai_bot.make_forecaster("anthropic/claude-test-opus")
    assert "max_tokens" not in direct.litellm_kwargs
    assert "openrouter/ models only" in caplog.text


@pytest.mark.parametrize(("finish_reason", "warned"), [("length", True), ("stop", False)])
def test_truncated_answer_is_flagged(monkeypatch, captured_requests, caplog, finish_reason, warned):
    CANNED_FINISH_REASON[0] = finish_reason
    monkeypatch.setenv("REASONING_EFFORT", "high")
    with caplog.at_level("INFO", logger="ckai_bot"):
        asyncio.run(ckai_bot.make_forecaster(OPUS).invoke("Will it rain?"))
    assert "Forecaster call openrouter/anthropic/claude-test-opus" in caplog.text
    assert f"finish_reason {finish_reason}" in caplog.text
    assert ("answer was cut off" in caplog.text) is warned
