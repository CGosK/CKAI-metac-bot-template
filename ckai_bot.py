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

Usage:
    uv run python ckai_bot.py --mode test_questions --dry-run
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
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    PredictionTypes,
    ReasonedPrediction,
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


def make_llm(model: str, temperature: float = 0.3) -> GeneralLlm:
    return GeneralLlm(
        model=model,
        temperature=temperature,
        timeout=120,
        allowed_tries=2,
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
        if purpose == "researcher" and (
            value.startswith("asknews/")
            or value.startswith("smart-searcher/")
            or value == "no_research"
        ):
            llms[purpose] = value
        else:
            llms[purpose] = make_llm(value, temperature=0.1)
    return llms


class CkaiBot(FallTemplateBot2026):
    """The template bot with a round-robin model ensemble for forecasting."""

    def __init__(self, *args, forecaster_models: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._forecasters: list[GeneralLlm] = [
            make_llm(m) for m in (forecaster_models or [])
        ]
        self._next_forecaster = 0

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
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    dry_run = args.dry_run or (_env("DRY_RUN") or "").lower() in TRUE_VALUES
    publish = not dry_run

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
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )

    bot.log_report_summary(reports)
    print_run_summary_banner(
        reports,
        will_publish=publish,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )


if __name__ == "__main__":
    main()
