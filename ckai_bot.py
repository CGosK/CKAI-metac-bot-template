"""
CKAI forecasting bot.

A thin layer on top of Metaculus's template bot (main.py). main.py is left
untouched so each season's template update from Metaculus merges cleanly;
everything specific to this bot lives here.

What it adds to the template:
- Model ensemble: set FORECASTER_MODELS to a comma-separated list of models
  (for example one each from Anthropic, OpenAI and Google on OpenRouter).
  Each of the bot's predictions per question goes to the next model in the
  list, and the template's usual aggregation combines them. Each rationale in
  the posted comment is labelled with the model that wrote it.
- Model choice from environment variables, so models can change without a code
  edit: FORECASTER_MODELS, RESEARCHER_MODEL, SUMMARIZER_MODEL, PARSER_MODEL.
  REASONING_EFFORT (low, medium, high, ...) applies to OpenRouter forecasters;
  with it set, no custom temperature is sent and the timeout is longer.
  FORECASTER_MAX_TOKENS caps their output (thinking plus answer); it defaults
  to 32000 at low, medium and high effort.
  Anything left unset falls back to the template's defaults.
- Dry run: `--dry-run` (or DRY_RUN=true) forecasts without publishing anything
  to Metaculus and saves the reports to ./forecast_reports for review.
- `--max-questions N` (test_questions mode): forecast only N bot-testing-area
  questions, spread across question types, for a small first run.
- Local development on a Claude subscription: models named `claude-cli/<id>`
  run through headless Claude Code (see claude_cli_llm.py). They are refused
  in CI, when publishing, outside the bot-testing area, and unless every LLM
  setting is a claude-cli model (research may also be no_research or
  asknews/news-summaries).

Usage:
    uv run python ckai_bot.py --mode test_questions --dry-run
    uv run python ckai_bot.py --mode test_questions --dry-run --max-questions 3
    uv run python ckai_bot.py --mode test_questions
    uv run python ckai_bot.py --mode tournament

Tournament rule to keep in mind: you may not run the bot on open or upcoming
tournament questions, look at its output and then change the bot. Test on the
bot-testing-area (mode test_questions) or on closed questions instead.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import logging
import os
import signal
import sys
import time
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    ConditionalQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    PredictionTypes,
    ReasonedPrediction,
)

from claude_cli_llm import (  # noqa: E402
    PLAIN,
    RESEARCH,
    ClaudeCliLlm,
    ClaudeCliRefused,
    is_claude_cli,
    kill_running_calls,
    refuse_in_ci,
)
from main import FallTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

TRUE_VALUES = {"1", "true", "yes", "on"}

# The model the current prediction should use. Each prediction runs in its own
# asyncio task, so a context variable keeps concurrent predictions separate.
_current_forecaster: contextvars.ContextVar[GeneralLlm | None] = (
    contextvars.ContextVar("current_forecaster", default=None)
)


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip() or value.strip() == "REPLACE_ME":
        return None
    return value.strip()


def parse_model_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def make_llm(
    model: str, temperature: float = 0.3, purpose: str = "default"
) -> GeneralLlm:
    if is_claude_cli(model):
        return ClaudeCliLlm(
            model,
            mode=RESEARCH if purpose == "researcher" else PLAIN,
            temperature=temperature,
            allowed_tries=2,
        )
    return GeneralLlm(
        model=model,
        temperature=temperature,
        timeout=120,
        allowed_tries=2,
    )


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
REASONING_TIMEOUT = 600  # seconds; high reasoning can take minutes per forecast


def reasoning_effort() -> str | None:
    """REASONING_EFFORT for the forecasters, or None to use each model's default."""
    value = _env("REASONING_EFFORT")
    if value is None:
        return None
    value = value.lower()
    if value not in REASONING_EFFORTS:
        raise ValueError(
            f"REASONING_EFFORT={value!r}: use one of {', '.join(REASONING_EFFORTS)}"
        )
    return value


# OpenRouter sets an Anthropic model's thinking budget to this share of
# max_tokens (openrouter.ai/docs/use-cases/reasoning-tokens); the rest is left
# for the answer.
THINKING_SHARE = {
    "none": Fraction(0),
    "minimal": Fraction("0.1"),
    "low": Fraction("0.2"),
    "medium": Fraction("0.5"),
    "high": Fraction("0.8"),
    "xhigh": Fraction("0.95"),
    "max": Fraction("0.95"),
}
MIN_ANSWER_TOKENS = 4000  # room the template's rationale and final answer need
MAX_TOKENS_LIMIT = 128000
DEFAULT_MAX_TOKENS = 32000
DEFAULT_MAX_TOKENS_EFFORTS = ("low", "medium", "high")


def min_max_tokens(effort: str | None) -> int:
    """The smallest cap that leaves MIN_ANSWER_TOKENS after the thinking budget."""
    answer_share = 1 - THINKING_SHARE[effort or "none"]
    return -(-MIN_ANSWER_TOKENS // answer_share)  # ceiling division on a Fraction


def forecaster_max_tokens(effort: str | None) -> int | None:
    """FORECASTER_MAX_TOKENS for OpenRouter forecasters, the default for this
    effort if unset, or None to send no cap."""
    raw = _env("FORECASTER_MAX_TOKENS")
    if raw is None:
        return DEFAULT_MAX_TOKENS if effort in DEFAULT_MAX_TOKENS_EFFORTS else None
    minimum = int(min_max_tokens(effort))
    if not raw.isdigit() or not minimum <= int(raw) <= MAX_TOKENS_LIMIT:
        raise ValueError(
            f"FORECASTER_MAX_TOKENS={raw!r}: use a whole number from {minimum} to "
            f"{MAX_TOKENS_LIMIT} (at REASONING_EFFORT={effort or 'unset'}, smaller caps leave "
            f"under {MIN_ANSWER_TOKENS} tokens for the answer after thinking)"
        )
    return int(raw)


class CappedForecasterLlm(GeneralLlm):
    """An OpenRouter forecaster that logs each call's length and token use, and
    warns when the answer was cut off at max_tokens. forecasting-tools passes a
    truncated answer on as if it were complete."""

    async def _call_litellm(self, prompt):
        start = time.monotonic()
        response = await super()._call_litellm(prompt)
        seconds = time.monotonic() - start
        try:
            finish_reason = response.choices[0].finish_reason
        except (AttributeError, IndexError):
            finish_reason = None
        usage = getattr(response, "usage", None)
        completion = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None)
        cap = self.litellm_kwargs.get("max_tokens")
        logger.info(
            f"Forecaster call {self.model}: {seconds:.0f}s, {completion} completion tokens "
            f"({reasoning} reasoning), cap {cap}, finish_reason {finish_reason}"
        )
        if finish_reason == "length":
            logger.warning(
                f"Forecaster {self.model} hit max_tokens ({cap}) and its answer was cut off. "
                "Raise FORECASTER_MAX_TOKENS or lower REASONING_EFFORT."
            )
        return response


def make_forecaster(model: str) -> GeneralLlm:
    """A forecasting model, with REASONING_EFFORT and FORECASTER_MAX_TOKENS
    applied if set. Only OpenRouter models get them: litellm maps reasoning
    differently for providers called directly (a small Anthropic thinking
    budget, dropped or rejected values for OpenAI and Gemini). Reasoning models
    take no custom temperature, so none is sent with it. claude-cli models use
    Claude Code's own thinking settings."""
    effort = reasoning_effort()
    max_tokens = forecaster_max_tokens(effort)
    if is_claude_cli(model):
        return make_llm(model)
    if not model.startswith("openrouter/"):
        if effort is not None or _env("FORECASTER_MAX_TOKENS") is not None:
            logger.warning(
                f"REASONING_EFFORT and FORECASTER_MAX_TOKENS apply to openrouter/ models only; "
                f"{model} runs at its defaults."
            )
        return make_llm(model)
    kwargs: dict[str, object] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if effort is None:
        return CappedForecasterLlm(
            model=model, temperature=0.3, timeout=120, allowed_tries=2, **kwargs
        )
    return CappedForecasterLlm(
        model=model,
        temperature=None,
        timeout=REASONING_TIMEOUT,
        allowed_tries=2,
        reasoning_effort=effort,
        **kwargs,
    )


MODEL_SETTINGS = ("FORECASTER_MODELS", "RESEARCHER_MODEL", "SUMMARIZER_MODEL", "PARSER_MODEL")


# Research settings allowed alongside claude-cli models: none, or AskNews news
# search. Only news-summaries: AskNews's deep-research presets run a model on
# AskNews's servers (and the deeper ones ask for premium sources).
ASKNEWS_PRESETS = ("asknews/news-summaries",)


def is_allowed_research_strategy(value: object) -> bool:
    return value == "no_research" or value in ASKNEWS_PRESETS


def asknews_credentials_problem() -> str | None:
    """Why AskNews can't run with the current settings, or None if it can."""
    placeholders = [
        var
        for var in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET")
        if (os.getenv(var) or "").strip() and not _env(var)
    ]
    if placeholders:
        # The library reads the raw values and prefers this pair over an API key.
        return f"{' and '.join(placeholders)} still a placeholder; remove or fill in"
    if _env("ASKNEWS_API_KEY") or (_env("ASKNEWS_CLIENT_ID") and _env("ASKNEWS_SECRET")):
        return None
    return "ASKNEWS_API_KEY (or ASKNEWS_CLIENT_ID + ASKNEWS_SECRET) for AskNews research"


def uses_claude_cli() -> bool:
    """True if any model setting names a claude-cli model."""
    return any(
        "claude-cli/" in (_env(var) or "") for var in MODEL_SETTINGS
    )


def check_claude_cli_config(run_mode: str, publish: bool) -> None:
    """With claude-cli models, every setting must be one and the run must be a
    local dry run on the bot-testing area. Raises ClaudeCliRefused otherwise."""
    refuse_in_ci()
    if publish:
        raise ClaudeCliRefused(
            "claude-cli models are for dry runs only. Add --dry-run, or use API-key models to publish."
        )
    if run_mode != "test_questions":
        raise ClaudeCliRefused(
            "claude-cli models run only on the bot-testing area (--mode test_questions). "
            "Running on open tournament questions and then changing the bot breaks the tournament rules."
        )
    forecasters = parse_model_list(_env("FORECASTER_MODELS"))
    problems = []
    if not forecasters or not all(is_claude_cli(m) for m in forecasters):
        problems.append("FORECASTER_MODELS")
    researcher = _env("RESEARCHER_MODEL")
    if not (is_allowed_research_strategy(researcher) or is_claude_cli(researcher)):
        problems.append(
            "RESEARCHER_MODEL (claude-cli/..., no_research, or one of " + ", ".join(ASKNEWS_PRESETS) + ")"
        )
    elif researcher in ASKNEWS_PRESETS and asknews_credentials_problem():
        problems.append(asknews_credentials_problem())
    for var in ("SUMMARIZER_MODEL", "PARSER_MODEL"):
        if not is_claude_cli(_env(var)):
            problems.append(var)
    if problems:
        raise ClaudeCliRefused(
            "With claude-cli models, every LLM setting must be a claude-cli model (research may "
            "also be no_research or AskNews), so no library default is used. Set: "
            + ", ".join(problems)
        )


def llms_from_env() -> dict[str, str | GeneralLlm]:
    """LLM overrides from environment variables. Unset purposes keep the template defaults."""
    llms: dict[str, str | GeneralLlm] = {}
    forecasters = parse_model_list(_env("FORECASTER_MODELS"))
    if forecasters:
        llms["default"] = make_forecaster(forecasters[0])
    for purpose, var in (
        ("researcher", "RESEARCHER_MODEL"),
        ("summarizer", "SUMMARIZER_MODEL"),
        ("parser", "PARSER_MODEL"),
    ):
        value = _env(var)
        if not value:
            continue
        # asknews/... and smart-searcher/... are strategy names, not LLMs.
        if purpose == "researcher" and "claude-cli/" in value and not is_claude_cli(value):
            raise ClaudeCliRefused(
                f"RESEARCHER_MODEL={value}: a search strategy would call claude-cli outside "
                "the backend. Use claude-cli/<model id> directly."
            )
        if purpose == "researcher" and (
            value.startswith("asknews/")
            or value.startswith("smart-searcher/")
            or value == "no_research"
        ):
            llms[purpose] = value
        else:
            llms[purpose] = make_llm(value, temperature=0.1, purpose=purpose)
    return llms


class CkaiBot(FallTemplateBot2026):
    """The template bot with a round-robin model ensemble for forecasting."""

    def __init__(self, *args, forecaster_models: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._forecasters: list[GeneralLlm] = [
            make_forecaster(m) for m in (forecaster_models or [])
        ]
        self._next_forecaster = 0
        self._claude_cli = any(
            is_claude_cli(llm) for llm in [*self._llms.values(), *self._forecasters]
        )
        if self._claude_cli:
            self._check_claude_cli_bot()

    def _check_claude_cli_bot(self) -> None:
        """Checked after the library has filled in defaults, so none can slip in."""
        refuse_in_ci()
        if self.publish_reports_to_metaculus:
            raise ClaudeCliRefused("claude-cli models are for dry runs only; publishing is on.")
        # Instances only: a bare "claude-cli/..." string would become a plain
        # GeneralLlm in the library and go through litellm.
        not_cli = [
            purpose
            for purpose, llm in self._llms.items()
            if not (
                isinstance(llm, ClaudeCliLlm)
                # A research strategy counts only if set explicitly: the library
                # picks AskNews by default whenever its keys are present.
                or (
                    purpose == "researcher"
                    and is_allowed_research_strategy(llm)
                    and llm == _env("RESEARCHER_MODEL")
                )
            )
        ]
        not_cli += [llm.model for llm in self._forecasters if not isinstance(llm, ClaudeCliLlm)]
        if not_cli:
            raise ClaudeCliRefused(
                "With claude-cli models, every model must be one. Not claude-cli: "
                + ", ".join(not_cli)
            )

    async def forecast_questions(self, questions: Sequence[MetaculusQuestion], *args, **kwargs):  # type: ignore[override]
        if self._claude_cli:
            # Checked again here: publishing could have been switched on after the bot was built.
            self._check_claude_cli_bot()
            outside = [
                q.page_url for q in questions if set(q.tournament_slugs) != {"bot-testing-area"}
            ]
            if outside:
                raise ClaudeCliRefused(
                    "claude-cli models run only on questions in the bot-testing area and "
                    "nowhere else. Not allowed: " + ", ".join(str(u) for u in outside[:5])
                )
        return await super().forecast_questions(questions, *args, **kwargs)

    def _pick_forecaster(self) -> GeneralLlm | None:
        if not self._forecasters:
            return None
        llm = self._forecasters[self._next_forecaster % len(self._forecasters)]
        self._next_forecaster += 1
        return llm

    def get_llm(self, purpose="default", guarantee_type=None):  # type: ignore[override]
        if purpose == "default":
            override = _current_forecaster.get()
            if override is not None:
                if guarantee_type == "string_name":
                    return override.model
                return override
        return super().get_llm(purpose, guarantee_type)  # type: ignore[arg-type]

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[PredictionTypes]:
        forecaster = self._pick_forecaster()
        if forecaster is None:
            return await super()._make_prediction(question, research)
        token = _current_forecaster.set(forecaster)
        try:
            prediction = await super()._make_prediction(question, research)
        finally:
            _current_forecaster.reset(token)
        prediction.reasoning = f"(Model: {forecaster.model})\n\n{prediction.reasoning}"
        return prediction


TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/futureeval/",
    "metaculus_cup": "https://www.metaculus.com/tournaments/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def build_bot(publish: bool, forecaster_models: list[str]) -> CkaiBot:
    if _env("PREDICTIONS_PER_QUESTION"):
        predictions = int(_env("PREDICTIONS_PER_QUESTION"))  # type: ignore[arg-type]
    elif forecaster_models:
        # At least 5 predictions, spread evenly across the ensemble.
        n = len(forecaster_models)
        predictions = n * -(-5 // n)
    else:
        predictions = 5
    return CkaiBot(
        research_reports_per_question=1,
        predictions_per_research_report=predictions,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None if publish else "forecast_reports",
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=llms_from_env() or None,
        forecaster_models=forecaster_models,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the CKAI forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forecast without publishing anything to Metaculus",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="test_questions mode only: forecast this many bot-testing-area questions",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    dry_run = args.dry_run or (_env("DRY_RUN") or "").lower() in TRUE_VALUES
    publish = not dry_run
    if args.max_questions is not None and (run_mode != "test_questions" or args.max_questions < 1):
        parser.error("--max-questions needs --mode test_questions and a number of at least 1")
    try:
        effort = reasoning_effort()  # checked even if FORECASTER_MODELS is unset
        max_tokens = forecaster_max_tokens(effort)
    except ValueError as exc:
        sys.exit(str(exc))
    if effort:
        logger.info(f"Reasoning effort for OpenRouter forecasters: {effort}")
    if max_tokens:
        logger.info(f"Max tokens for OpenRouter forecasters: {max_tokens}")

    if uses_claude_cli():
        try:
            check_claude_cli_config(run_mode, publish)
        except ClaudeCliRefused as exc:
            sys.exit(f"Refused: {exc}")
        if not _env("METACULUS_TOKEN"):
            sys.exit("METACULUS_TOKEN is missing; it's needed to read the bot-testing-area questions.")
    else:
        check_environment(strict=True)
    print_startup_banner(run_mode, will_publish=publish)

    forecaster_models = parse_model_list(_env("FORECASTER_MODELS"))
    if forecaster_models:
        logger.info(f"Forecaster ensemble: {forecaster_models}")
    else:
        logger.info("FORECASTER_MODELS not set; using the template's default model")
    if publish and len(forecaster_models) == 0:
        logger.warning(
            "Publishing with the template's default forecaster. Set FORECASTER_MODELS "
            "to current models before the season starts."
        )

    bot = build_bot(publish, forecaster_models)
    client = MetaculusClient()

    # SIGTERM and SIGHUP (closing the terminal) take the same path as Ctrl-C,
    # so claude processes, which run in their own process groups, get stopped.
    if uses_claude_cli():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _interrupt)
    try:
        reports = _run(bot, client, run_mode, args.max_questions)
    except KeyboardInterrupt:
        kill_running_calls()
        raise

    bot.log_report_summary(reports)
    print_run_summary_banner(
        reports,
        will_publish=publish,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )


def _interrupt(signum, frame) -> None:  # noqa: ARG001
    raise KeyboardInterrupt


def pick_questions(questions: Sequence[MetaculusQuestion], n: int) -> list[MetaculusQuestion]:
    """Up to n questions, taking one of each question type in turn. Skips
    conditional questions, which cost several predictions each."""
    by_type: dict[str, list[MetaculusQuestion]] = {}
    for q in questions:
        if isinstance(q, ConditionalQuestion):
            continue
        by_type.setdefault(type(q).__name__, []).append(q)
    picked: list[MetaculusQuestion] = []
    queues = [list(v) for _, v in sorted(by_type.items())]
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    return picked


def _run(bot: CkaiBot, client: MetaculusClient, run_mode: str, max_questions: int | None) -> list:
    if run_mode == "tournament":
        reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        reports += asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif max_questions is not None:
        bot.skip_previously_forecasted_questions = False
        # Fetched outside asyncio.run: the client runs its own event loop.
        questions = client.get_all_open_questions_from_tournament("bot-testing-area")
        picked = pick_questions(questions, max_questions)
        logger.info(
            "Forecasting %d of %d bot-testing-area questions: %s",
            len(picked),
            len(questions),
            [(type(q).__name__, q.page_url) for q in picked],
        )
        reports = asyncio.run(bot.forecast_questions(picked, return_exceptions=True))
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )
    return reports


if __name__ == "__main__":
    main()
