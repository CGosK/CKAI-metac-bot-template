"""
Local development backend: run the bot's model calls through headless Claude
Code (`claude -p`) on your own Claude subscription, for dry runs on the
bot-testing area while you build. The live bot uses API keys (OpenRouter).

Model names look like `claude-cli/claude-sonnet-5-5`: the prefix, then a full
Claude model ID.

Every call goes through the Runner in rkit (the owner's private research
kit), found through RKIT_PATH. Without it this backend refuses to run; there is no
unguarded fallback. The Runner sets the Claude config folder, a clean
environment, a per-call temp folder, empty stdin and a neutral git identity,
and its usage guard stops calls before they would use extra usage. All
runners here share one guard, so a stop kills every running call.

Limits enforced in this file:
- refuses in CI (GITHUB_ACTIONS or CI set), at construction and on every call
- an error result, or an empty one, raises instead of becoming research or a
  forecast
- a call fails if the requested model wrote nothing, or if another model
  answered instead (a fallback). Only Haiku helper output in research calls,
  which Claude Code uses for WebFetch, is allowed and logged
- after Ctrl-C or SIGTERM, running calls are killed and queued ones refuse
  to start

Publishing and question-set limits are enforced in ckai_bot.py.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

logger = logging.getLogger(__name__)

PREFIX = "claude-cli/"
PLAIN = "plain"
RESEARCH = "research"

RESEARCH_TOOLS = ("WebSearch", "WebFetch")
# Everything else is denied explicitly. rkit's default deny list includes the
# web tools, which would silently block research.
RESEARCH_DENIED = (
    "Agent",
    "Bash",
    "Edit",
    "Write",
    "Read",
    "Glob",
    "Grep",
    "NotebookEdit",
    "TodoWrite",
)

PLAIN_SYSTEM_PROMPT = (
    "You are a careful assistant helping with forecasting. Answer the user's "
    "request directly in plain text, following any output format it asks for."
)
RESEARCH_SYSTEM_PROMPT = (
    "You are a research assistant to a forecaster. Use web search and web "
    "fetch to find current, relevant information for the request, then reply "
    "with a concise written rundown that cites sources by URL. Do not forecast."
)

DEFAULT_RUNS_DIR = "~/.cache/ckai-bot-runs"
PLAIN_TIMEOUT = 10 * 60
RESEARCH_TIMEOUT = 15 * 60


class ClaudeCliRefused(RuntimeError):
    """The subscription backend was asked to run somewhere it must not."""


class ClaudeCliCallFailed(RuntimeError):
    """A call finished but its result can't be used."""


def is_claude_cli(name: object) -> bool:
    if isinstance(name, ClaudeCliLlm):
        return True
    return isinstance(name, str) and name.startswith(PREFIX)


def refuse_in_ci() -> None:
    for var in ("GITHUB_ACTIONS", "CI"):
        if os.getenv(var):
            raise ClaudeCliRefused(
                f"claude-cli models run only on your own machine ({var} is set). "
                "Use an API key (OpenRouter) in CI."
            )


class _Backend:
    """Runners per (model, mode), one shared usage guard, a cap on parallel calls."""

    def __init__(self) -> None:
        raw = os.getenv("RKIT_PATH")
        if not raw:
            raise ClaudeCliRefused(
                "RKIT_PATH isn't set. claude-cli models only run through rkit's "
                "runner and usage guard; set RKIT_PATH in .env to the research-kit folder."
            )
        rkit_path = Path(raw).expanduser().resolve()
        if not (rkit_path / "rkit" / "runner.py").is_file():
            raise ClaudeCliRefused("RKIT_PATH has no rkit/runner.py")
        if str(rkit_path) not in sys.path:
            sys.path.append(str(rkit_path))
        try:
            from rkit import runner as rk_runner
            from rkit import usage_guard as rk_guard
        except ImportError as exc:
            raise ClaudeCliRefused(f"Can't import rkit from RKIT_PATH: {exc}") from exc
        for module in (rk_runner, rk_guard):
            if not Path(module.__file__).resolve().is_relative_to(rkit_path):
                raise ClaudeCliRefused(
                    "rkit was imported from somewhere other than RKIT_PATH; refusing to run."
                )
        self._rk_runner = rk_runner
        self._rk_guard = rk_guard

        base = Path(os.getenv("CKAI_RUNS_DIR", DEFAULT_RUNS_DIR)).expanduser()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = base / f"{stamp}-{uuid.uuid4().hex[:6]}"
        self.work_dir = self.run_dir / "work"
        self.calls_dir = self.run_dir / "calls"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        # Fixed across runs, so a restarted run still waits out a pause.
        self.state_path = str(base / "usage_state.json")
        self.guard = rk_guard.UsageGuard(self.state_path, log=logger.warning)

        self._runners: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        parallel = int(os.getenv("CLAUDE_CLI_MAX_PARALLEL", "3"))
        if parallel < 1:
            raise ClaudeCliRefused("CLAUDE_CLI_MAX_PARALLEL must be at least 1")
        self._slots = threading.BoundedSemaphore(parallel)
        self.stopping = threading.Event()
        self._counter = itertools.count(1)
        logger.info(f"claude-cli call logs: {self.calls_dir}")

    def runner(self, model: str, mode: str) -> Any:
        with self._lock:
            key = (model, mode)
            if key not in self._runners:
                Runner = self._rk_runner.Runner
                if mode == RESEARCH:
                    runner = Runner(
                        model,
                        self.state_path,
                        agentic=True,
                        tools=RESEARCH_TOOLS,
                        denied_tools=RESEARCH_DENIED,
                        extra_args=("--tools", ",".join(RESEARCH_TOOLS)),
                        system_prompt=RESEARCH_SYSTEM_PROMPT,
                        timeout=RESEARCH_TIMEOUT,
                        log=logger.warning,
                    )
                else:
                    runner = Runner(
                        model,
                        self.state_path,
                        agentic=False,
                        system_prompt=PLAIN_SYSTEM_PROMPT,
                        timeout=PLAIN_TIMEOUT,
                        log=logger.warning,
                    )
                runner.guard = self.guard
                self._runners[key] = runner
            return self._runners[key]

    def call(self, model: str, mode: str, prompt: str) -> str:
        refuse_in_ci()
        runner = self.runner(model, mode)
        n = next(self._counter)
        log_path = self.calls_dir / f"{n:04d}-{mode}-{model}.jsonl"
        session_id = str(uuid.uuid4())
        with self._slots:
            # Wait out any usage-guard pause here, where a stop can interrupt
            # it, rather than inside runner.call, where it can't.
            self._wait_for_guard_or_stop()
            try:
                res = runner.call(
                    prompt,
                    cwd=str(self.work_dir),
                    log_path=str(log_path),
                    session_id=session_id,
                )
            finally:
                runner.end_session(session_id)
        return check_result(model, mode, res.result, res.output_by_model, log_path.name)

    def _guard_clear_at(self) -> float:
        """When the shared guard will let a call start (its own rule in
        UsageGuard.wait_until_clear)."""
        now = time.time()
        with self.guard.lock:
            until = self.guard.paused_until
            for util, reset in self.guard.windows.values():
                if util >= self.guard.launch_at and reset > now:
                    until = max(until, reset + self.guard.margin)
        return until

    def _wait_for_guard_or_stop(self) -> None:
        announced = False
        while True:
            if self.stopping.is_set():
                raise ClaudeCliRefused("Stopping: not starting new claude calls.")
            wait = self._guard_clear_at() - time.time()
            if wait <= 0:
                return
            if not announced:
                logger.warning(f"Usage guard pause: waiting {wait / 60:.0f} min before the next call.")
                announced = True
            self.stopping.wait(min(wait, 1.0))

    def kill_running_calls(self) -> None:
        self.stopping.set()
        with self.guard.lock:
            procs = list(self.guard.procs)
        for proc in procs:
            self._rk_guard._kill(proc)


def _is_model(name: str, model: str) -> bool:
    return name == model or name.startswith(model + "[")


def check_result(
    model: str, mode: str, result: dict, output_by_model: dict, log_name: str
) -> str:
    """The call's text, or ClaudeCliCallFailed. Messages name models and the
    log file only, no paths or token counts, since they end up in reports."""
    if result.get("is_error") or result.get("subtype") != "success":
        detail = str(result.get("result") or "")[:300]
        raise ClaudeCliCallFailed(
            f"{model} call failed ({result.get('subtype')}): {detail} [log {log_name}]"
        )
    text = result.get("result")
    if not isinstance(text, str) or not text.strip():
        raise ClaudeCliCallFailed(f"{model} returned no text [log {log_name}]")
    written = sum(n for m, n in output_by_model.items() if _is_model(m, model))
    others = {m: n for m, n in output_by_model.items() if not _is_model(m, model) and n > 0}
    if written <= 0:
        raise ClaudeCliCallFailed(
            f"{model} wrote nothing; output came from {sorted(others)} [log {log_name}]"
        )
    # A fallback model answering in place of the requested one must not pass as
    # the requested model. Research calls may use Haiku for WebFetch.
    allowed_helpers = {m for m in others if mode == RESEARCH and m.startswith("claude-haiku")}
    unexpected = sorted(set(others) - allowed_helpers)
    if unexpected:
        raise ClaudeCliCallFailed(
            f"{model} call also had output from {unexpected} [log {log_name}]"
        )
    if allowed_helpers:
        logger.info(f"Helper model output in a {model} research call: {sorted(allowed_helpers)}")
    return text


_backend: _Backend | None = None
_backend_lock = threading.Lock()


def get_backend() -> _Backend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = _Backend()
        return _backend


def reset_backend() -> None:
    """For tests."""
    global _backend
    with _backend_lock:
        _backend = None


def kill_running_calls() -> None:
    """Stop every running claude process (they run in their own process groups)
    and refuse any call still queued."""
    with _backend_lock:
        backend = _backend
    if backend is not None:
        backend.kill_running_calls()


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ClaudeCliCallFailed("claude-cli models take text prompts only")
            parts.append(f"{message.get('role', 'user').upper()}:\n{message['content']}")
        return "\n\n".join(parts)
    raise ClaudeCliCallFailed(f"claude-cli models take text prompts only, not {type(prompt)}")


class ClaudeCliLlm(GeneralLlm):
    """A GeneralLlm whose calls run through `claude -p` on your subscription."""

    def __init__(self, model: str, mode: str = PLAIN, **kwargs: Any) -> None:
        if not isinstance(model, str) or not model.startswith(PREFIX):
            raise ValueError(f"ClaudeCliLlm needs a {PREFIX}<model id> name, not {model!r}")
        cli_model = model.removeprefix(PREFIX)
        if not cli_model.startswith("claude-"):
            raise ValueError(f"Use a full Claude model ID after {PREFIX}, not {cli_model!r}")
        if mode not in (PLAIN, RESEARCH):
            raise ValueError(f"Unknown mode {mode!r}")
        refuse_in_ci()
        super().__init__(model=model, **kwargs)
        # Plain attributes, not litellm kwargs, so they stay out of report metadata.
        self.cli_model = cli_model
        self.cli_mode = mode

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:
        # Refusals are checked here, before the library's retry wrapper, so they
        # fail at once instead of being retried.
        refuse_in_ci()
        backend = get_backend()
        if backend.stopping.is_set():
            raise ClaudeCliRefused("Stopping: not starting new claude calls.")
        return await super().invoke(prompt, system_prompt)

    async def _mockable_direct_call_to_model(self, prompt: Any) -> TextTokenCostResponse:
        refuse_in_ci()
        try:
            text = await asyncio.to_thread(
                get_backend().call, self.cli_model, self.cli_mode, _prompt_text(prompt)
            )
        except asyncio.CancelledError:
            # The thread can't be cancelled: stop its claude process and any
            # queued calls instead.
            kill_running_calls()
            raise
        return TextTokenCostResponse(
            data=text,
            prompt_tokens_used=0,
            completion_tokens_used=0,
            total_tokens_used=0,
            model=self.model,
            cost=0.0,
        )
