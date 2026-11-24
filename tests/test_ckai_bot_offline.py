"""
Offline test of the CKAI bot: no Metaculus or LLM calls, nothing published.

The LLM and the output parser are replaced with stand-ins, and the bot runs on
locally built questions of each type (binary, multiple choice, numeric, date).
It checks that the model ensemble rotates, that every rationale is labelled
with its model, and that each question gets a valid aggregated forecast.

Run with:  uv run python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ["openrouter/test/model-a", "openrouter/test/model-b", "openrouter/test/model-c"]
BINARY_BY_MODEL = {"model-a": 0.2, "model-b": 0.4, "model-c": 0.6}


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.setenv("FORECASTER_MODELS", ",".join(MODELS))
    monkeypatch.setenv("RESEARCHER_MODEL", "no_research")
    monkeypatch.setenv("SUMMARIZER_MODEL", "openrouter/test/summarizer")
    monkeypatch.setenv("PARSER_MODEL", "openrouter/test/parser")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-not-a-real-key")
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    yield


def _install_fakes(monkeypatch):
    import main
    from forecasting_tools import (
        BinaryPrediction,
        DatePercentile,
        GeneralLlm,
        Percentile,
        PredictedOption,
        PredictedOptionList,
    )

    calls: list[str] = []

    async def fake_invoke(self, prompt, *args, **kwargs):
        calls.append(self.model)
        short = self.model.rsplit("/", 1)[-1]
        return f"Reasoning written by {short}. Status quo matters."

    async def fake_structure_output(text_to_structure=None, output_type=None, *args, **kwargs):
        if text_to_structure is None:
            text_to_structure = args[0]
        if output_type is None:
            output_type = args[1] if len(args) > 1 else kwargs["output_type"]
        model = re.search(r"model-[abc]", text_to_structure).group(0)
        if output_type is BinaryPrediction:
            return BinaryPrediction(prediction_in_decimal=BINARY_BY_MODEL[model])
        if output_type is PredictedOptionList:
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="Red", probability=0.5),
                    PredictedOption(option_name="Blue", probability=0.3),
                    PredictedOption(option_name="Green", probability=0.2),
                ]
            )
        if output_type == list[Percentile]:
            return [
                Percentile(percentile=p, value=v)
                for p, v in [(0.1, 20), (0.2, 30), (0.4, 45), (0.6, 55), (0.8, 70), (0.9, 80)]
            ]
        if output_type == list[DatePercentile]:
            base = datetime(2027, 3, 1, tzinfo=timezone.utc)
            return [
                DatePercentile(percentile=p, value=base + timedelta(days=d))
                for p, d in [(0.1, 5), (0.2, 15), (0.4, 30), (0.6, 45), (0.8, 60), (0.9, 75)]
            ]
        raise AssertionError(f"Unexpected output type {output_type}")

    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)
    monkeypatch.setattr(main, "structure_output", fake_structure_output)
    return calls


def _questions():
    from forecasting_tools import (
        BinaryQuestion,
        DateQuestion,
        MultipleChoiceQuestion,
        NumericQuestion,
    )

    common = dict(
        resolution_criteria="Offline test criteria.",
        fine_print="",
        background_info="Offline test background.",
    )
    return [
        BinaryQuestion(question_text="Will the offline test pass?", page_url="https://example.test/q/1", **common),
        MultipleChoiceQuestion(
            question_text="Which colour?", options=["Red", "Blue", "Green"],
            page_url="https://example.test/q/2", **common,
        ),
        NumericQuestion(
            question_text="How many?", upper_bound=100, lower_bound=0,
            open_upper_bound=True, open_lower_bound=False, unit_of_measure="units",
            page_url="https://example.test/q/3", **common,
        ),
        DateQuestion(
            question_text="When?",
            upper_bound=datetime(2027, 12, 31, tzinfo=timezone.utc),
            lower_bound=datetime(2027, 1, 1, tzinfo=timezone.utc),
            open_upper_bound=True, open_lower_bound=False,
            page_url="https://example.test/q/4", **common,
        ),
    ]


def test_model_settings_from_env():
    import ckai_bot

    assert ckai_bot.parse_model_list(" a, b ,,c ") == ["a", "b", "c"]
    llms = ckai_bot.llms_from_env()
    assert llms["default"].model == MODELS[0]
    assert llms["researcher"] == "no_research"
    assert llms["parser"].model == "openrouter/test/parser"


def test_predictions_spread_evenly():
    import ckai_bot

    bot = ckai_bot.build_bot(publish=False, forecaster_models=MODELS)
    assert bot.predictions_per_research_report == 6
    assert bot.publish_reports_to_metaculus is False


def test_ensemble_forecasts_every_question_type(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = _install_fakes(monkeypatch)
    import ckai_bot

    bot = ckai_bot.build_bot(publish=False, forecaster_models=MODELS)
    bot.folder_to_save_reports_to = None
    reports = asyncio.run(bot.forecast_questions(_questions(), return_exceptions=True))

    errors = [r for r in reports if isinstance(r, BaseException)]
    assert not errors, errors
    assert len(reports) == 4

    forecaster_calls = [c for c in calls if "model-" in c]
    for model in MODELS:
        assert forecaster_calls.count(model) == 8, forecaster_calls  # 2 per question x 4

    binary = reports[0]
    assert binary.prediction == pytest.approx(0.4)  # median of 0.2, 0.4, 0.6 (x2)
    for model in MODELS:
        assert f"(Model: {model})" in binary.explanation

    for report in reports:
        assert report.explanation.strip()
