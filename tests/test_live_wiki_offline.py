"""
Offline tests of the live bot's research handling: AskNews errors no longer
drop the question, and WIKI_RESEARCH (off, fallback, append) adds Wikipedia
research. No network, no model calls: AskNews and Wikipedia are stand-ins.

Run with:  uv run python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ["openrouter/test/model-a", "openrouter/test/model-b", "openrouter/test/model-c"]
NEWS = "Here are the relevant news articles:\n\n**Rain expected**\nForecasters expect rain.\n\n"
NO_NEWS = "Here are the relevant news articles:\n\nNo articles were found.\n\n"  # the library's exact text
WIKI = "Wikipedia articles as they stood on 2026-12-01:\n\n### Rain\nRain is water."


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.setenv("FORECASTER_MODELS", ",".join(MODELS))
    monkeypatch.setenv("RESEARCHER_MODEL", "asknews/news-summaries")
    monkeypatch.setenv("SUMMARIZER_MODEL", "openrouter/test/summarizer")
    monkeypatch.setenv("PARSER_MODEL", "openrouter/test/parser")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-not-a-real-key")
    monkeypatch.setenv("ASKNEWS_API_KEY", "offline-test-not-a-real-key")
    for var in ("REASONING_EFFORT", "WIKI_RESEARCH", "WIKI_HELPER_MODEL", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(var, raising=False)
    yield


def _question(n: int = 1):
    from forecasting_tools import BinaryQuestion

    return BinaryQuestion(
        question_text="Will it rain?",
        resolution_criteria="Rain at the airport.",
        fine_print="",
        background_info="",
        page_url=f"https://www.metaculus.com/questions/{n}/",
        tournament_slugs=["bot-testing-area"],
    )


def _asknews(monkeypatch, result):
    """AskNews stand-in: returns `result`, or raises it if it's an exception."""
    import main

    calls = []

    async def fake(self, preset, prompt):
        calls.append(prompt)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(main.AskNewsSearcher, "call_preconfigured_version", fake)
    return calls


def _wallet_empty():
    from asknews_sdk.errors import APIError

    return APIError(None, "Wallet balance depleted", 402001)


def _bot(monkeypatch, mode: str | None = None, wiki_text: str = WIKI):
    """A bot with WIKI_RESEARCH=mode and a stand-in for Wikipedia's research."""
    import ckai_bot

    if mode:
        monkeypatch.setenv("WIKI_RESEARCH", mode)
    bot = ckai_bot.build_bot(publish=False, forecaster_models=MODELS)
    wiki_calls = []
    if bot._wiki is not None:
        async def fake_research(question):
            wiki_calls.append(question.page_url)
            return wiki_text

        monkeypatch.setattr(bot._wiki, "research", fake_research)
    return bot, wiki_calls


# ---- AskNews errors no longer drop the question ---------------------------------


def test_off_keeps_todays_research(monkeypatch):
    _asknews(monkeypatch, NEWS)
    bot, _ = _bot(monkeypatch)
    assert bot._wiki is None
    assert asyncio.run(bot.run_research(_question())) == NEWS


def test_asknews_error_gives_empty_research_and_a_warning(monkeypatch, caplog):
    _asknews(monkeypatch, _wallet_empty())
    bot, _ = _bot(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(bot.run_research(_question())) == ""
    assert "Research failed" in caplog.text and "Wallet balance depleted" in caplog.text


def test_question_is_forecast_when_asknews_fails(monkeypatch, tmp_path):
    """The whole question, not just research: before the fix it failed."""
    import test_ckai_bot_offline as fakes

    monkeypatch.chdir(tmp_path)
    fakes._install_fakes(monkeypatch)
    _asknews(monkeypatch, _wallet_empty())
    bot, _ = _bot(monkeypatch)
    bot.folder_to_save_reports_to = None
    reports = asyncio.run(bot.forecast_questions([_question()], return_exceptions=True))
    assert not isinstance(reports[0], BaseException), reports[0]
    assert reports[0].prediction == pytest.approx(0.4)


def test_claude_cli_refusals_still_stop_research(monkeypatch):
    import main

    import ckai_bot

    async def refuse(self, question):
        raise ckai_bot.ClaudeCliRefused("no")

    monkeypatch.setattr(main.FallTemplateBot2026, "run_research", refuse)
    bot, _ = _bot(monkeypatch)
    with pytest.raises(ckai_bot.ClaudeCliRefused):
        asyncio.run(bot.run_research(_question()))


# ---- WIKI_RESEARCH ------------------------------------------------------------------


@pytest.mark.parametrize("news", [_wallet_empty(), NO_NEWS, ""])
def test_fallback_uses_wikipedia_when_asknews_gives_nothing(monkeypatch, news):
    _asknews(monkeypatch, news)
    bot, wiki_calls = _bot(monkeypatch, "fallback")
    research = asyncio.run(bot.run_research(_question()))
    assert WIKI in research and len(wiki_calls) == 1


def test_fallback_with_no_research_setting(monkeypatch):
    monkeypatch.setenv("RESEARCHER_MODEL", "no_research")
    bot, wiki_calls = _bot(monkeypatch, "fallback")
    assert asyncio.run(bot.run_research(_question())) == WIKI
    assert wiki_calls


def test_fallback_skips_wikipedia_when_asknews_found_articles(monkeypatch):
    _asknews(monkeypatch, NEWS)
    bot, wiki_calls = _bot(monkeypatch, "fallback")
    assert asyncio.run(bot.run_research(_question())) == NEWS
    assert wiki_calls == []  # no Wikipedia fetch at all, so no portal fetch either


def test_append_joins_news_then_wikipedia(monkeypatch):
    import wiki_research

    _asknews(monkeypatch, NEWS)
    bot, _ = _bot(monkeypatch, "append")
    research = asyncio.run(bot.run_research(_question()))
    assert research == wiki_research.join_research(NEWS, WIKI)
    assert research.index("Rain expected") < research.index("Wikipedia articles")


def test_wikipedia_finding_nothing_leaves_research_unchanged(monkeypatch):
    _asknews(monkeypatch, NEWS)
    bot, _ = _bot(monkeypatch, "append", wiki_text="")
    assert asyncio.run(bot.run_research(_question())) == NEWS


def test_join_drops_empty_parts():
    from wiki_research import NO_ARTICLES, join_research

    assert join_research("", WIKI) == WIKI
    assert join_research(NEWS, NO_ARTICLES) == NEWS.strip()
    assert join_research(NEWS, WIKI) == NEWS.strip() + "\n\n" + WIKI


def test_backtest_combined_text_is_the_live_join():
    from backtest import run
    import wiki_research

    news = {"q": {"text": NEWS}}
    wiki_rows = {"q": {"text": WIKI}}
    assert run.combined_row("q", news, wiki_rows)["text"] == wiki_research.join_research(NEWS, WIKI)
    with pytest.raises(ValueError):
        run.combined_row("missing", news, wiki_rows)


def test_summary_leaves_out_wikipedia(monkeypatch):
    import main

    _asknews(monkeypatch, NEWS)
    bot, _ = _bot(monkeypatch, "append")
    seen = []

    async def fake_summary(self, question, research):
        seen.append(research)
        return "summary"

    monkeypatch.setattr(main.FallTemplateBot2026, "summarize_research", fake_summary)
    question = _question()
    research = asyncio.run(bot.run_research(question))
    assert asyncio.run(bot.summarize_research(question, research)) == "summary"
    assert seen == [NEWS.strip()]


def test_summary_skipped_when_research_is_only_wikipedia(monkeypatch):
    monkeypatch.setenv("RESEARCHER_MODEL", "no_research")
    bot, _ = _bot(monkeypatch, "fallback")
    question = _question()
    research = asyncio.run(bot.run_research(question))
    assert "not summarized" in asyncio.run(bot.summarize_research(question, research))


def test_unknown_mode_fails(monkeypatch):
    import ckai_bot

    monkeypatch.setenv("WIKI_RESEARCH", "sometimes")
    with pytest.raises(ValueError, match="WIKI_RESEARCH"):
        ckai_bot.build_bot(publish=False, forecaster_models=MODELS)


def test_helper_defaults_to_backtest_model(monkeypatch):
    import ckai_bot

    monkeypatch.setenv("WIKI_RESEARCH", "fallback")
    bot = ckai_bot.build_bot(publish=False, forecaster_models=MODELS)
    assert bot._wiki.helper.model == "openrouter/anthropic/claude-haiku-5.5"
    assert bot._wiki.freeze is None  # nothing fetched until a question needs it


# ---- the live Wikipedia researcher ---------------------------------------------------


def test_wikipedia_failure_trips_the_breaker(monkeypatch, caplog):
    import ckai_bot
    import wiki_research

    calls = []

    async def failing(q, llm, api, include_today=False):
        calls.append(q)
        raise wiki_research.WikiStopped("HTTP 403")

    monkeypatch.setattr(wiki_research, "research_for", failing)
    live = ckai_bot.LiveWiki(helper=None)
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(live.research(_question(1))) == ""
        assert asyncio.run(live.research(_question(2))) == ""
    assert len(calls) == 1 and live.off and live.api.closed
    assert caplog.text.count("Wikipedia research off") == 1


def test_slow_question_is_skipped_alone(monkeypatch):
    import ckai_bot
    import wiki_research

    async def hangs(q, llm, api, include_today=False):
        await asyncio.sleep(60)

    monkeypatch.setattr(wiki_research, "research_for", hangs)
    monkeypatch.setattr(ckai_bot, "WIKI_DEADLINE", 0.1)
    live = ckai_bot.LiveWiki(helper=None)
    assert asyncio.run(live.research(_question())) == ""
    assert not live.off


def test_many_questions_take_turns_without_tripping(monkeypatch):
    """20 questions at once: one at a time, each deadline counted from its turn."""
    import ckai_bot
    import wiki_research

    running, most = 0, 0

    async def fake(q, llm, api, include_today=False):
        nonlocal running, most
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.02)
        running -= 1
        return {"articles": [], "portal_items": 0, "text": WIKI}

    monkeypatch.setattr(wiki_research, "research_for", fake)
    monkeypatch.setattr(ckai_bot, "WIKI_DEADLINE", 0.1)  # shorter than the whole queue (0.4 s)
    live = ckai_bot.LiveWiki(helper=None)

    async def all_at_once():
        return await asyncio.gather(*(live.research(_question(n)) for n in range(20)))

    assert asyncio.run(all_at_once()) == [WIKI] * 20
    assert most == 1 and not live.off
    assert asyncio.run(live.research(_question(99))) == WIKI  # a second event loop, as in tournament mode


def test_blocked_request_trips_breaker_and_queue_fails_fast(monkeypatch):
    """A real Api whose requests time out: the first question trips the breaker,
    and the next fails at once without a request."""
    import ckai_bot
    import wiki_research

    def times_out(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(wiki_research.urllib.request, "urlopen", times_out)

    class Helper:
        async def invoke(self, prompt):
            return '{"titles": ["Rain"], "keywords": ["rain"]}'

    api = wiki_research.live_api()
    api.gap, api.backoff = 0, 0
    live = ckai_bot.LiveWiki(helper=Helper(), api=api)
    assert asyncio.run(live.research(_question(1))) == ""
    assert live.off and api.requests == api.tries
    assert asyncio.run(live.research(_question(2))) == ""
    assert api.requests == api.tries  # nothing more was sent


def test_failed_claude_cli_research_call_is_not_hidden(monkeypatch):
    import main

    import ckai_bot

    async def fails(self, question):
        raise ckai_bot.ClaudeCliCallFailed("claude-cli/claude-test call failed")

    monkeypatch.setattr(main.FallTemplateBot2026, "run_research", fails)
    bot, _ = _bot(monkeypatch)
    with pytest.raises(ckai_bot.ClaudeCliCallFailed):
        asyncio.run(bot.run_research(_question()))


def test_one_bad_helper_reply_skips_only_that_question(monkeypatch):
    import ckai_bot
    import wiki_research

    async def bad_json(q, llm, api, include_today=False):
        raise ValueError("no JSON in helper reply")

    monkeypatch.setattr(wiki_research, "research_for", bad_json)
    live = ckai_bot.LiveWiki(helper=None)
    assert asyncio.run(live.research(_question())) == ""
    assert not live.off


def test_live_research_uses_one_time_point_and_todays_portal(monkeypatch):
    import ckai_bot
    import wiki_research

    seen = []

    async def fake(q, llm, api, include_today=False):
        seen.append((q, include_today))
        return {"articles": [{"title": "Rain", "revid": 1}], "portal_items": 0, "text": WIKI}

    monkeypatch.setattr(wiki_research, "research_for", fake)
    live = ckai_bot.LiveWiki(helper=None)
    asyncio.run(live.research(_question(1)))
    asyncio.run(live.research(_question(2)))
    assert seen[0][0]["freeze_datetime"] == seen[1][0]["freeze_datetime"]
    assert seen[0][0]["question"] == "Will it rain?" and seen[0][0]["criteria"] == "Rain at the airport."
    assert all(include for _, include in seen)


def test_closed_api_fails_without_a_request():
    import wiki_research

    api = wiki_research.live_api()
    api.close()
    with pytest.raises(wiki_research.WikiStopped):
        api.get({"action": "query"})
    assert api.requests == 0


def test_live_api_is_quick_to_give_up():
    import wiki_research

    api = wiki_research.live_api()
    assert api.tries == 2 and api.timeout <= 15 and api.max_wait <= 5
    assert api.memory is not None  # no disk cache in CI
    assert "github.com/CGosK/CKAI-metac-bot-template" in api.user_agent


def test_portal_day_zero_only_live():
    import wiki_research

    assert 0 not in wiki_research.portal_days()
    assert 0 in wiki_research.portal_days(include_today=True)
    assert len(wiki_research.portal_days()) == wiki_research.PORTAL_DAYS


# ---- claude-cli safety checks cover the helper ----------------------------------------


def test_cli_helper_with_api_forecasters_is_refused(monkeypatch):
    import ckai_bot

    monkeypatch.setenv("WIKI_RESEARCH", "fallback")
    monkeypatch.setenv("WIKI_HELPER_MODEL", "claude-cli/claude-test")
    assert ckai_bot.uses_claude_cli()
    with pytest.raises(ckai_bot.ClaudeCliRefused):
        ckai_bot.check_claude_cli_config("tournament", publish=True)
    with pytest.raises(ckai_bot.ClaudeCliRefused):
        ckai_bot.build_bot(publish=False, forecaster_models=MODELS)


def test_cli_models_need_a_cli_helper(monkeypatch):
    import ckai_bot

    cli = "claude-cli/claude-test"
    for var in ("FORECASTER_MODELS", "SUMMARIZER_MODEL", "PARSER_MODEL"):
        monkeypatch.setenv(var, cli)
    monkeypatch.setenv("RESEARCHER_MODEL", "no_research")
    ckai_bot.check_claude_cli_config("test_questions", publish=False)
    monkeypatch.setenv("WIKI_RESEARCH", "fallback")
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="WIKI_HELPER_MODEL"):
        ckai_bot.check_claude_cli_config("test_questions", publish=False)
    monkeypatch.setenv("WIKI_HELPER_MODEL", "openrouter/anthropic/claude-haiku-5.5")
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="WIKI_HELPER_MODEL"):
        ckai_bot.check_claude_cli_config("test_questions", publish=False)
    monkeypatch.setenv("WIKI_HELPER_MODEL", cli)
    ckai_bot.check_claude_cli_config("test_questions", publish=False)
