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
  Anything left unset falls back to the template's defaults.
- Dry run: `--dry-run` (or DRY_RUN=true) forecasts without publishing anything
  to Metaculus and saves the reports to ./forecast_reports for review.
- `--max-questions N` (test_questions mode): forecast only N bot-testing-area
  questions, spread across question types, for a small first run.
- Local development on a Claude subscription: models named `claude-cli/<id>`
  run through headless Claude Code (see claude_cli_llm.py). They are refused
  in CI, when publishing, outside the bot-testing area, and unless every model
  setting is a claude-cli model.

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
from collections.abc import Sequence
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


MODEL_SETTINGS = ("FORECASTER_MODELS", "RESEARCHER_MODEL", "SUMMARIZER_MODEL", "PARSER_MODEL")


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
    if not (researcher == "no_research" or is_claude_cli(researcher)):
        problems.append("RESEARCHER_MODEL (claude-cli/... or no_research)")
    for var in ("SUMMARIZER_MODEL", "PARSER_MODEL"):
        if not is_claude_cli(_env(var)):
            problems.append(var)
    if problems:
        raise ClaudeCliRefused(
            "With claude-cli models, every model setting must be a claude-cli model, so no "
            "library default is used. Set: " + ", ".join(problems)
        )


def llms_from_env() -> dict[str, str | GeneralLlm]:
    """LLM overrides from environment variables. Unset purposes keep the template defaults."""
    llms: dict[str, str | GeneralLlm] = {}
    forecasters = parse_model_list(_env("FORECASTER_MODELS"))
    if forecasters:
        llms["default"] = make_llm(forecasters[0])
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
            make_llm(m) for m in (forecaster_models or [])
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
                or (purpose == "researcher" and llm == "no_research")
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
