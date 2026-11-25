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
    monkeypatch.delenv("WIKI_RESEARCH", raising=False)
    monkeypatch.delenv("WIKI_HELPER_MODEL", raising=False)
    monkeypatch.delenv("PROMPT_VERSION", raising=False)
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


def _captured_binary_prompt(monkeypatch, version, research="RESEARCH TEXT"):
    import ckai_bot
    from forecasting_tools import BinaryQuestion

    if version is None:
        monkeypatch.delenv("PROMPT_VERSION", raising=False)
    else:
        monkeypatch.setenv("PROMPT_VERSION", version)
    bot = ckai_bot.build_bot(publish=False, forecaster_models=MODELS)
    seen = []

    async def capture(self, question, prompt):
        seen.append(prompt)
        return None

    monkeypatch.setattr(ckai_bot.CkaiBot, "_binary_prompt_to_forecast", capture)
    q = BinaryQuestion(question_text="Will X happen?", background_info="BG", resolution_criteria="CRIT",
                       fine_print="FP", page_url="test:1")
    asyncio.run(bot._run_forecast_on_binary(q, research))
    return seen[0]


def test_template_prompt_is_the_default(monkeypatch):
    prompt = _captured_binary_prompt(monkeypatch, None)
    assert "good forecasters put extra weight on the status quo outcome" in prompt
    assert "reference class" not in prompt


def test_v2_prompt_has_the_three_changes(monkeypatch):
    prompt = _captured_binary_prompt(monkeypatch, "v2")
    for text in ("Will X happen?", "BG", "CRIT", "FP", "RESEARCH TEXT", "Today is ",
                 "readings of the resolution criteria", "reference class", "base rate",
                 "only when the research covers the last few days", '"Probability: ZZ%", 0-100'):
        assert text in prompt, text
    assert "good forecasters put extra weight on the status quo outcome" not in prompt


def test_unknown_prompt_version_fails(monkeypatch):
    import ckai_bot

    monkeypatch.setenv("PROMPT_VERSION", "v9")
    with pytest.raises(ValueError, match="PROMPT_VERSION"):
        ckai_bot.build_bot(publish=False, forecaster_models=MODELS)


def test_v2_prompt_differs_from_template_only_in_the_planned_lines(monkeypatch):
    from prompts_v2 import TEMPLATE_ONLY, V2_ONLY_PREFIXES

    research = "Line one\n    indented line\n        deeper line\n\nlast"
    template = _captured_binary_prompt(monkeypatch, None, research).splitlines()
    v2 = _captured_binary_prompt(monkeypatch, "v2", research).splitlines()
    assert [ln for ln in template if ln not in TEMPLATE_ONLY] == [
        ln for ln in v2 if not ln.startswith(V2_ONLY_PREFIXES)]
    assert all(ln in template for ln in TEMPLATE_ONLY)
    assert sum(ln.startswith(V2_ONLY_PREFIXES) for ln in v2) == len(V2_ONLY_PREFIXES)


def _parent_prompt(monkeypatch, env: dict, reverse=False):
    import ckai_bot
    import main
    from forecasting_tools import BinaryQuestion

    for k in ("PROMPT_VERSION", "PROMPT_EVIDENCE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    bot = ckai_bot.build_bot(publish=False, forecaster_models=MODELS)
    bot._reverse_binary = reverse
    seen = []

    async def capture(self, question, prompt):
        seen.append(prompt)
        return None

    monkeypatch.setattr(main.FallTemplateBot2026, "_binary_prompt_to_forecast", capture)
    q = BinaryQuestion(question_text="Will X happen?", background_info="BG", resolution_criteria="CRIT",
                       fine_print="FP", page_url="test:1")
    asyncio.run(bot._run_forecast_on_binary(q, "RESEARCH TEXT"))
    return seen[0]


def test_evidence_step_off_by_default_and_added_when_on(monkeypatch):
    from prompts_r4 import EVIDENCE_STEP

    plain = _parent_prompt(monkeypatch, {})
    assert "(i) The single fact" not in plain
    for version in ("template", "v2"):
        prompt = _parent_prompt(monkeypatch, {"PROMPT_EVIDENCE": "on", "PROMPT_VERSION": version})
        assert EVIDENCE_STEP.strip().split("\n")[1] in prompt
        assert prompt.index("(i) The single fact") < prompt.index("You write your rationale")
        base = _parent_prompt(monkeypatch, {"PROMPT_VERSION": version})
        added = [ln for ln in prompt.splitlines() if ln not in base.splitlines()]
        assert added == EVIDENCE_STEP.strip().split("\n")


def test_reverse_asks_for_no(monkeypatch):
    prompt = _parent_prompt(monkeypatch, {}, reverse=True)
    assert prompt.startswith("IMPORTANT: This task asks for the probability that the question below resolves NO")
    assert "These criteria have not yet been satisfied" not in prompt
    assert "Will X happen?" in prompt and "CRIT" in prompt and "RESEARCH TEXT" in prompt
    assert prompt.rstrip().endswith('written as "Probability: ZZ%", 0-100.')


def test_unknown_prompt_evidence_fails(monkeypatch):
    import ckai_bot

    monkeypatch.setenv("PROMPT_EVIDENCE", "maybe")
    with pytest.raises(ValueError, match="PROMPT_EVIDENCE"):
        ckai_bot.build_bot(publish=False, forecaster_models=MODELS)
