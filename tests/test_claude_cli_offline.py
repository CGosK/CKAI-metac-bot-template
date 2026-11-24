"""
Offline tests of the claude-cli backend (claude_cli_llm.py) and its limits in
ckai_bot.py. A stub stands in for the `claude` binary, so nothing reaches
Claude, Metaculus or any API.

The limit tests run anywhere. The backend tests need rkit's runner, found
through RKIT_PATH (read from .env); where rkit isn't available, as in the
GitHub offline-tests workflow, they skip.

Run with:  uv run --with pytest python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import dotenv
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

_rkit = os.getenv("RKIT_PATH")
HAVE_RKIT = bool(_rkit) and Path(_rkit).expanduser().joinpath("rkit", "runner.py").is_file()
needs_rkit = pytest.mark.skipif(not HAVE_RKIT, reason="rkit not available (set RKIT_PATH)")

MODEL = "claude-test-1"
CLI = f"claude-cli/{MODEL}"

STUB = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys, time
    if "--version" in sys.argv:
        print("0.0.0 (stub)"); sys.exit(0)
    record = {record!r}
    n = len(os.listdir(record))
    with open(os.path.join(record, "call-%d.json" % n), "w") as fh:
        json.dump({{"argv": sys.argv[1:], "env": dict(os.environ)}}, fh)
    mode = {mode!r}
    argv = sys.argv[1:]
    model = argv[argv.index("--model") + 1]
    window = {{"five_hour": {{"utilization": 0.1, "resets" "At": time.time() + 3600, "status": "allowed"}}}}
    info = {{"status": "allowed", "isUsing" "Overage": False, "unified" "Windows": window}}
    print(json.dumps({{"type": "rate_limit_event", "rate_limit" "_info": info}}), flush=True)
    text = "Stub answer from %s." % model
    usage = {{model: {{"output" "Tokens": 25}}}}
    result = {{"type": "result", "subtype": "success", "is_error": False, "result": text}}
    if mode == "error":
        result.update(subtype="error_during_execution", is_error=True, result="API Error: overloaded")
    elif mode == "empty":
        result["result"] = "  "
    elif mode == "wrong_model":
        usage = {{"claude-other-9": {{"output" "Tokens": 25}}}}
    elif mode == "fallback":
        usage = {{model: {{"output" "Tokens": 3}}, "claude-older-4-8": {{"output" "Tokens": 400}}}}
    elif mode == "helper":
        usage["claude-haiku-test"] = {{"output" "Tokens": 5}}
    result["model" "Usage"] = usage
    print(json.dumps(result), flush=True)
    """
)


@pytest.fixture(autouse=True)
def local_env(monkeypatch, tmp_path):
    """A local, non-CI run whose `claude` can't run until a test installs a stub."""
    import claude_cli_llm

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CKAI_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RKIT_CLAUDE_BIN", str(tmp_path / "no-claude-here"))
    claude_cli_llm.reset_backend()
    yield
    claude_cli_llm.reset_backend()


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """Install a stub `claude` in the given mode; returns a reader for recorded calls."""
    import claude_cli_llm

    record = tmp_path / "calls"
    record.mkdir()
    identities = tmp_path / "identities.json"
    identities.write_text(json.dumps({"names": ["Test Operator"]}))
    monkeypatch.setenv("RKIT_IDENTITIES", str(identities))
    monkeypatch.setenv("RKIT_IDENTITIES_ONLY", "1")

    def install(mode: str = "ok"):
        path = tmp_path / f"claude-stub-{mode}"
        path.write_text(STUB.format(python=sys.executable, record=str(record), mode=mode))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("RKIT_CLAUDE_BIN", str(path))
        claude_cli_llm.reset_backend()

        def calls():
            return [
                json.loads(p.read_text())
                for p in sorted(record.iterdir(), key=lambda p: int(p.stem.split("-")[1]))
            ]

        return calls

    return install


def all_cli_env(monkeypatch, researcher: str = CLI):
    monkeypatch.setenv("FORECASTER_MODELS", f"{CLI},claude-cli/claude-test-2")
    monkeypatch.setenv("RESEARCHER_MODEL", researcher)
    monkeypatch.setenv("SUMMARIZER_MODEL", CLI)
    monkeypatch.setenv("PARSER_MODEL", CLI)
    monkeypatch.delenv("PREDICTIONS_PER_QUESTION", raising=False)


# ---- backend (needs rkit) ------------------------------------------------------


@needs_rkit
def test_text_comes_back(stub):
    from claude_cli_llm import ClaudeCliLlm

    calls = stub("ok")
    text = asyncio.run(ClaudeCliLlm(CLI).invoke("What is 2+2?"))
    assert text == f"Stub answer from {MODEL}."
    argv = calls()[0]["argv"]
    assert argv[argv.index("--model") + 1] == MODEL
    assert argv[argv.index("--tools") + 1] == ""
    assert "--system-prompt" in argv


@needs_rkit
def test_runner_environment_is_clean(stub, monkeypatch):
    from claude_cli_llm import ClaudeCliLlm

    monkeypatch.setenv("OPENROUTER_API_KEY", "should-not-reach-claude")
    monkeypatch.setenv("METACULUS_TOKEN", "should-not-reach-claude")
    calls = stub("ok")
    asyncio.run(ClaudeCliLlm(CLI).invoke("hi"))
    env = calls()[0]["env"]
    assert "OPENROUTER_API_KEY" not in env and "METACULUS_TOKEN" not in env
    assert env["CLAUDE_CONFIG_DIR"].endswith(".claude")


@needs_rkit
def test_research_mode_gets_only_web_tools(stub):
    from claude_cli_llm import RESEARCH, ClaudeCliLlm

    calls = stub("ok")
    asyncio.run(ClaudeCliLlm(CLI, mode=RESEARCH).invoke("Find news"))
    argv = calls()[0]["argv"]
    assert argv[argv.index("--tools") + 1] == "WebSearch,WebFetch"
    allowed = argv[argv.index("--allowedTools") + 1 : argv.index("--disallowedTools")]
    assert set(allowed) == {"WebSearch", "WebFetch"}
    denied = []
    for a in argv[argv.index("--disallowedTools") + 1 :]:
        if a.startswith("--"):
            break
        denied.append(a)
    assert "Bash" in denied and "Agent" in denied
    assert "WebSearch" not in denied and "WebFetch" not in denied


@needs_rkit
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("error", "call failed"),
        ("empty", "returned no text"),
        ("wrong_model", "wrote nothing"),
        ("fallback", "also had output from"),
        ("helper", "also had output from"),  # plain mode allows no helper output
    ],
)
def test_bad_results_raise(stub, mode, message):
    from claude_cli_llm import ClaudeCliCallFailed, ClaudeCliLlm

    stub(mode)
    with pytest.raises(ClaudeCliCallFailed, match=message):
        asyncio.run(ClaudeCliLlm(CLI, allowed_tries=1).invoke("hi"))


@needs_rkit
def test_error_messages_have_no_paths(stub):
    from claude_cli_llm import ClaudeCliCallFailed, ClaudeCliLlm

    stub("error")
    with pytest.raises(ClaudeCliCallFailed) as info:
        asyncio.run(ClaudeCliLlm(CLI, allowed_tries=1).invoke("hi"))
    assert "/" not in str(info.value).split("[log", 1)[1]


@needs_rkit
def test_haiku_helper_allowed_in_research(stub):
    from claude_cli_llm import RESEARCH, ClaudeCliLlm

    stub("helper")
    text = asyncio.run(ClaudeCliLlm(CLI, mode=RESEARCH).invoke("hi"))
    assert text.startswith("Stub answer")


@needs_rkit
def test_runners_share_one_guard(stub):
    import claude_cli_llm

    stub("ok")
    backend = claude_cli_llm.get_backend()
    a = backend.runner(MODEL, claude_cli_llm.PLAIN)
    b = backend.runner("claude-test-2", claude_cli_llm.RESEARCH)
    assert a.guard is b.guard is backend.guard


@needs_rkit
def test_no_new_calls_after_stop(stub):
    import claude_cli_llm

    calls = stub("ok")
    llm = claude_cli_llm.ClaudeCliLlm(CLI, allowed_tries=1)
    claude_cli_llm.get_backend()
    claude_cli_llm.kill_running_calls()
    with pytest.raises(claude_cli_llm.ClaudeCliRefused, match="Stopping"):
        asyncio.run(llm.invoke("hi"))
    with pytest.raises(claude_cli_llm.ClaudeCliRefused, match="Stopping"):
        claude_cli_llm.get_backend().call(MODEL, claude_cli_llm.PLAIN, "hi")
    assert calls() == []


@needs_rkit
def test_stop_during_guard_pause_starts_nothing(stub):
    import threading
    import time

    import claude_cli_llm

    calls = stub("ok")
    backend = claude_cli_llm.get_backend()
    with backend.guard.lock:
        backend.guard.paused_until = time.time() + 30
    outcome: dict = {}

    def worker():
        try:
            backend.call(MODEL, claude_cli_llm.PLAIN, "hi")
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=worker)
    started = time.time()
    thread.start()
    time.sleep(0.3)
    claude_cli_llm.kill_running_calls()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert time.time() - started < 5
    assert isinstance(outcome.get("error"), claude_cli_llm.ClaudeCliRefused)
    assert calls() == []


@needs_rkit
def test_parallel_must_be_positive(stub, monkeypatch):
    import claude_cli_llm

    stub("ok")
    monkeypatch.setenv("CLAUDE_CLI_MAX_PARALLEL", "0")
    with pytest.raises(claude_cli_llm.ClaudeCliRefused, match="at least 1"):
        claude_cli_llm.get_backend()


# ---- refusals (run anywhere) -------------------------------------------------------


def test_refuses_in_ci(monkeypatch):
    from claude_cli_llm import ClaudeCliLlm, ClaudeCliRefused

    llm = ClaudeCliLlm(CLI, allowed_tries=1)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(ClaudeCliRefused, match="GITHUB_ACTIONS"):
        ClaudeCliLlm(CLI)
    with pytest.raises(ClaudeCliRefused, match="GITHUB_ACTIONS"):
        asyncio.run(llm.invoke("hi"))


def test_refuses_without_rkit(monkeypatch, tmp_path):
    import claude_cli_llm

    monkeypatch.setenv("RKIT_PATH", str(tmp_path / "nowhere"))
    with pytest.raises(claude_cli_llm.ClaudeCliRefused, match="RKIT_PATH"):
        asyncio.run(claude_cli_llm.ClaudeCliLlm(CLI, allowed_tries=1).invoke("hi"))


def test_short_model_names_are_refused():
    from claude_cli_llm import ClaudeCliLlm

    with pytest.raises(ValueError, match="full Claude model ID"):
        ClaudeCliLlm("claude-cli/sonnet")


def test_config_refuses_publishing_and_live_modes(monkeypatch):
    import ckai_bot

    all_cli_env(monkeypatch)
    ckai_bot.check_claude_cli_config("test_questions", publish=False)
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="dry runs only"):
        ckai_bot.check_claude_cli_config("test_questions", publish=True)
    for mode in ("tournament", "metaculus_cup"):
        with pytest.raises(ckai_bot.ClaudeCliRefused, match="bot-testing area"):
            ckai_bot.check_claude_cli_config(mode, publish=False)


def test_config_refuses_mixed_or_missing_settings(monkeypatch):
    import ckai_bot

    all_cli_env(monkeypatch)
    monkeypatch.delenv("SUMMARIZER_MODEL")
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="SUMMARIZER_MODEL"):
        ckai_bot.check_claude_cli_config("test_questions", publish=False)
    all_cli_env(monkeypatch)
    monkeypatch.setenv("PARSER_MODEL", "openrouter/openai/gpt-test")
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="PARSER_MODEL"):
        ckai_bot.check_claude_cli_config("test_questions", publish=False)
    all_cli_env(monkeypatch, researcher="no_research")
    ckai_bot.check_claude_cli_config("test_questions", publish=False)


def test_bot_refuses_publishing(monkeypatch):
    import ckai_bot

    all_cli_env(monkeypatch)
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="publishing is on"):
        ckai_bot.build_bot(publish=True, forecaster_models=[CLI])


def test_bot_refuses_library_defaults(monkeypatch):
    import ckai_bot

    all_cli_env(monkeypatch)
    monkeypatch.delenv("SUMMARIZER_MODEL")
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="summarizer"):
        ckai_bot.build_bot(publish=False, forecaster_models=[CLI])


def test_bot_refuses_bare_model_strings(monkeypatch):
    import ckai_bot

    all_cli_env(monkeypatch)
    llms = ckai_bot.llms_from_env()
    llms["parser"] = CLI  # a string, not a ClaudeCliLlm
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="parser"):
        ckai_bot.CkaiBot(llms=llms, forecaster_models=[CLI], publish_reports_to_metaculus=False)


def test_bot_builds_with_all_cli_models(monkeypatch):
    import claude_cli_llm
    import ckai_bot

    all_cli_env(monkeypatch)
    bot = ckai_bot.build_bot(publish=False, forecaster_models=[CLI, "claude-cli/claude-test-2"])
    assert isinstance(bot._llms["researcher"], claude_cli_llm.ClaudeCliLlm)
    assert bot._llms["researcher"].cli_mode == claude_cli_llm.RESEARCH
    assert bot._llms["parser"].cli_mode == claude_cli_llm.PLAIN
    assert "cli_mode" not in str(bot.make_llm_dict())


def test_smart_searcher_on_claude_cli_is_refused(monkeypatch):
    import ckai_bot

    all_cli_env(monkeypatch, researcher=f"smart-searcher/{CLI}")
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="search strategy"):
        ckai_bot.llms_from_env()


@pytest.mark.parametrize(
    "slugs", [["fall-futureeval-2026"], ["bot-testing-area", "fall-futureeval-2026"], []]
)
def test_bot_refuses_questions_outside_testing_area(monkeypatch, slugs):
    import ckai_bot
    from forecasting_tools import BinaryQuestion

    all_cli_env(monkeypatch)
    bot = ckai_bot.build_bot(publish=False, forecaster_models=[CLI])
    question = BinaryQuestion(
        question_text="Will it rain?",
        page_url="https://www.metaculus.com/questions/1/",
        tournament_slugs=slugs,
    )
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="bot-testing area"):
        asyncio.run(bot.forecast_questions([question], return_exceptions=True))


def test_bot_rechecks_publishing_before_forecasting(monkeypatch):
    import ckai_bot
    from forecasting_tools import BinaryQuestion

    all_cli_env(monkeypatch)
    bot = ckai_bot.build_bot(publish=False, forecaster_models=[CLI])
    bot.publish_reports_to_metaculus = True
    question = BinaryQuestion(
        question_text="Will it rain?",
        page_url="https://www.metaculus.com/questions/1/",
        tournament_slugs=["bot-testing-area"],
    )
    with pytest.raises(ckai_bot.ClaudeCliRefused, match="publishing is on"):
        asyncio.run(bot.forecast_questions([question], return_exceptions=True))


def test_pick_questions_spreads_types_and_skips_conditionals():
    import ckai_bot
    from forecasting_tools import BinaryQuestion, ConditionalQuestion, NumericQuestion

    binaries = [
        BinaryQuestion(question_text=f"b{i}", page_url=f"https://x/b{i}") for i in range(3)
    ]
    numerics = [
        NumericQuestion(
            question_text=f"n{i}",
            page_url=f"https://x/n{i}",
            upper_bound=10,
            lower_bound=0,
            open_upper_bound=False,
            open_lower_bound=False,
        )
        for i in range(2)
    ]
    conditionals = [
        ConditionalQuestion.model_construct(question_text=f"c{i}", page_url=f"https://x/c{i}")
        for i in range(3)
    ]
    picked = ckai_bot.pick_questions(conditionals + binaries + numerics, 4)
    assert [q.question_text for q in picked] == ["b0", "n0", "b1", "n1"]
