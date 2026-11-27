"""
Offline tests of the numeric and multiple-choice guards (numeric_guards.py and
the CkaiBot overrides) and regression tests for the library's numeric handling.
No model or network calls: the LLM and the parser are replaced with fakes.

Run with:  uv run --with pytest python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ckai_bot  # noqa: E402  (loads .env before the fixture clears settings)
import main  # noqa: E402
import numeric_guards as ng  # noqa: E402
from forecasting_tools import (  # noqa: E402
    DateQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.questions import DiscreteQuestion  # noqa: E402

LEVELS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
COMMON = dict(resolution_criteria="Offline criteria.", fine_print="", background_info="Offline background.")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-not-a-real-key")
    monkeypatch.setenv("FORECASTER_MODELS", "openrouter/test/model-a")
    monkeypatch.setenv("RESEARCHER_MODEL", "no_research")
    monkeypatch.setenv("SUMMARIZER_MODEL", "openrouter/test/summarizer")
    monkeypatch.setenv("PARSER_MODEL", "openrouter/test/parser")
    for var in ("REASONING_EFFORT", "PREDICTIONS_PER_QUESTION", "WIKI_RESEARCH", "WIKI_HELPER_MODEL",
                "FORECASTER_MAX_TOKENS", "CI", "GITHUB_ACTIONS", "PROMPT_VERSION"):
        monkeypatch.delenv(var, raising=False)
    yield


def pct(values, levels=LEVELS):
    return [Percentile(percentile=p, value=v) for p, v in zip(levels, values)]


def numeric_q(**kw):
    base = dict(question_text="How many people?", page_url="https://example.test/n", upper_bound=5_000_000,
                lower_bound=0, open_upper_bound=True, open_lower_bound=False, unit_of_measure="people",
                zero_point=None)
    base.update(kw)
    return NumericQuestion(**COMMON, **base)


def percentile_text(values, suffix=""):
    return "My reasoning.\n" + "\n".join(f"Percentile {int(p * 100)}: {v}{suffix}" for p, v in zip(LEVELS, values))


# ---- G3: the library's numeric handling (regression) -------------------------------


@pytest.mark.parametrize(
    ("label", "question", "values", "ok"),
    [
        ("linear sensible", numeric_q(), [1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6], True),
        ("linear, written in millions", numeric_q(), [1.0, 1.5, 2.0, 2.4, 3.0, 3.5], True),
        ("linear, 1000x too big", numeric_q(), [1e9, 1.5e9, 2e9, 2.4e9, 3e9, 3.5e9], False),
        ("log sensible", numeric_q(upper_bound=1e6, lower_bound=1, open_lower_bound=True, zero_point=0),
         [10, 50, 200, 1000, 5000, 20000], True),
        ("log near lower", numeric_q(upper_bound=1e6, lower_bound=1, open_lower_bound=True, zero_point=0),
         [1.2, 1.5, 2, 3, 5, 8], True),
        ("log below open lower", numeric_q(upper_bound=1e6, lower_bound=1, open_lower_bound=True, zero_point=0),
         [0.3, 0.6, 1.0, 1.5, 2, 3], True),
        ("log above open upper", numeric_q(upper_bound=1e6, lower_bound=1, open_lower_bound=True, zero_point=0),
         [2e5, 5e5, 9e5, 1.2e6, 1.5e6, 2e6], True),
        ("log far above open upper", numeric_q(upper_bound=1e6, lower_bound=1, open_lower_bound=True, zero_point=0),
         [3e6, 3.2e6, 3.4e6, 3.6e6, 3.8e6, 4e6], False),
    ],
)
def test_library_numeric_cases(label, question, values, ok):
    if ok:
        cdf = NumericDistribution.from_question(pct(values), question).get_cdf()
        assert len(cdf) == 201
    else:
        with pytest.raises(ValueError):
            NumericDistribution.from_question(pct(values), question).get_cdf()


def discrete_q():
    # Built as the API builds discrete questions: half-step bounds around nominal 0.1 to 0.8, 71 values.
    q = DiscreteQuestion(
        **COMMON, question_text="Which value?", page_url="https://example.test/d", upper_bound=0.805,
        lower_bound=0.095, open_upper_bound=False, open_lower_bound=False, unit_of_measure="rate",
        zero_point=None, cdf_size=72,
    )
    q.nominal_lower_bound, q.nominal_upper_bound = 0.1, 0.8
    return q


@pytest.mark.parametrize("values", [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.39, 0.395, 0.4, 0.4, 0.405, 0.41],
                                    [0.3, 0.3, 0.3, 0.3, 0.4, 0.4]])
def test_library_discrete_cases(values):
    cdf = NumericDistribution.from_question(pct(values), discrete_q()).get_cdf()
    assert len(cdf) == 72 and cdf[0].percentile == 0 and abs(cdf[-1].percentile - 1) < 1e-9


def test_library_date_case():
    from forecasting_tools import DatePercentile

    q = DateQuestion(**COMMON, question_text="When?", page_url="https://example.test/t",
                     upper_bound=datetime(2027, 12, 31, tzinfo=timezone.utc),
                     lower_bound=datetime(2027, 1, 1, tzinfo=timezone.utc),
                     open_upper_bound=True, open_lower_bound=False)
    days = [30, 60, 120, 180, 240, 300]
    dps = [DatePercentile(percentile=p, value=datetime(2027, 1, 1, tzinfo=timezone.utc).replace(day=1) +
                          __import__("datetime").timedelta(days=d)) for p, d in zip(LEVELS, days)]
    pcts = [Percentile(percentile=d.percentile, value=d.value.timestamp()) for d in dps]
    assert len(NumericDistribution.from_question(pcts, q).get_cdf()) == 201


@pytest.mark.parametrize(("total", "ok"), [(0.97, False), (0.985, False), (0.99, True), (1.005, True),
                                           (1.01, True), (1.02, False)])
def test_library_mc_sum_rule(total, ok):
    opts = [PredictedOption(option_name="A", probability=0.5), PredictedOption(option_name="B", probability=0.3),
            PredictedOption(option_name="C", probability=round(total - 0.8, 6))]
    if ok:
        PredictedOptionList(predicted_options=opts)
    else:
        with pytest.raises(ValueError):
            PredictedOptionList(predicted_options=opts)


# ---- G1: multiple choice -------------------------------------------------------------

OPTIONS = ["Red", "Blue – dark", "Green"]


def po(name, p):
    return PredictedOption(option_name=name, probability=p)


@pytest.mark.parametrize("total", [0.95, 0.985, 1.02, 1.08])
def test_g1_rescales_near_sums(total):
    parsed = [po("Red", 0.5), po("Blue – dark", 0.3), po("Green", round(total - 0.8, 6))]
    result, rescaled = ng.build_option_list(parsed, OPTIONS)
    assert rescaled == pytest.approx(total)
    assert sum(o.probability for o in result.predicted_options) == pytest.approx(1)


@pytest.mark.parametrize("total", [0.85, 1.15])
def test_g1_rejects_far_sums(total):
    parsed = [po("Red", 0.5), po("Blue – dark", 0.3), po("Green", round(total - 0.8, 6))]
    with pytest.raises(ValueError, match="too far from 1"):
        ng.build_option_list(parsed, OPTIONS)


def test_g1_rejects_zero_sum():
    with pytest.raises(ValueError):
        ng.build_option_list([po("Red", 0), po("Blue – dark", 0), po("Green", 0)], OPTIONS)


@pytest.mark.parametrize("total", [0.99, 1.0, 1.005, 1.01])
def test_g1_leaves_library_sums_alone(total):
    parsed = [po("Red", 0.5), po("Blue – dark", 0.3), po("Green", round(total - 0.8, 6))]
    ours, rescaled = ng.build_option_list(parsed, OPTIONS)
    library = PredictedOptionList(predicted_options=parsed)
    assert rescaled is None
    assert [o.probability for o in ours.predicted_options] == [o.probability for o in library.predicted_options]


def test_g1_matches_name_variants_and_reorders():
    parsed = [po("  green ", 0.2), po("Option A: Red", 0.5), po("blue - dark", 0.3)]
    result, _ = ng.build_option_list(parsed, OPTIONS)
    assert [o.option_name for o in result.predicted_options] == OPTIONS
    assert [o.probability for o in result.predicted_options] == pytest.approx([0.5, 0.3, 0.2])


@pytest.mark.parametrize(
    "parsed",
    [
        [po("Red", 0.5), po("Red", 0.3), po("Green", 0.2)],  # duplicate
        [po("Red", 0.7), po("Green", 0.3)],  # missing
        [po("Red", 0.5), po("Blue – dark", 0.2), po("Green", 0.2), po("Purple", 0.1)],  # extra
        [po("Option A", 0.5), po("Blue – dark", 0.3), po("Green", 0.2)],  # a bare letter label
    ],
)
def test_g1_rejects_bad_option_sets(parsed):
    with pytest.raises(ValueError):
        ng.build_option_list(parsed, OPTIONS)


def test_g1_option_word_not_stripped_when_options_use_it():
    with pytest.raises(ValueError):
        ng.build_option_list([po("option trading", 0.6), po("trading", 0.4)], ["Options trading", "trading"])


def test_g1_option_prefix_kept_when_some_option_starts_with_it():
    # "Option: Red" would match "Red" once stripped, but an option itself starts with "Option".
    with pytest.raises(ValueError):
        ng.build_option_list([po("Option: Red", 0.5), po("Option A", 0.5)], ["Option A", "Red"])


def test_reparse_instructions_state_nominal_bounds_in_plain_numbers():
    text = ng.reparse_instructions(discrete_q())
    assert "0.1 to 0.8" in text and "0.095" not in text
    big = ng.reparse_instructions(numeric_q(upper_bound=1e12, unit_of_measure=None))
    assert "1000000000000" in big and "e+" not in big and "(unit not stated)" in big


def test_g1_names_differing_only_by_case_need_exact_matches():
    options = ["Red", "red"]
    result, _ = ng.build_option_list([po("red", 0.4), po("Red", 0.6)], options)
    assert [o.probability for o in result.predicted_options] == pytest.approx([0.6, 0.4])
    with pytest.raises(ValueError):
        ng.build_option_list([po("RED", 0.4), po("Red", 0.6)], options)


def test_g1_unicode_forms_match():
    result, _ = ng.build_option_list([po("\uff32\uff45\uff44", 0.5), po("Blue \u2014 dark", 0.3), po("Green", 0.2)],
                                     OPTIONS)
    assert [o.option_name for o in result.predicted_options] == OPTIONS


def test_g1_keeps_option_prefix_when_part_of_the_name():
    options = ["Option 1: keep", "Option 2: drop"]
    result, _ = ng.build_option_list([po("option 1: keep", 0.6), po("Option 2: drop", 0.4)], options)
    assert [o.option_name for o in result.predicted_options] == options


# ---- G2: numeric units ---------------------------------------------------------------

MILLIONS = [1.0, 1.5, 2.0, 2.4, 3.0, 3.5]


def test_read_declared_percentiles():
    text = ("Draft: Percentile 10: 5\nFinal:\nPercentile 10: 1.0 million\nPercentile 20: $1,500,000\n"
            "**Percentile 40:** 2bn\nPercentile 60: 2.4M\nPercentile 80: 30%\nPercentile 90: -4.5 thousand\n"
            "Percentile 2: 1.2e6\nPercentile 3: 2.4Mt\nPercentile 4: 5 to 6 million\nPercentile 5: 1,000 - 2,000")
    got = ng.read_declared_percentiles(text)
    assert got[10] == ng.Reading(1.0, "million", False)
    assert got[20] == ng.Reading(1_500_000.0, None, False)
    assert got[40] == ng.Reading(2.0, "bn", False)
    assert got[60] == ng.Reading(2.4, "M", False)  # the letter kept as written
    assert got[80] == ng.Reading(30.0, None, True)
    assert got[90] == ng.Reading(-4.5, "thousand", False)
    assert got[2] == ng.Reading(1_200_000.0, None, False)  # scientific notation read whole
    assert got[3] == ng.Reading(2.4, None, False)  # "Mt" is a unit, not a magnitude
    assert 4 not in got and 5 not in got  # ranges are skipped


def test_read_declared_percentiles_line_ends():
    got = ng.read_declared_percentiles("Percentile 10: 5.\nPercentile 20: 80\nMillion-dollar losses\n"
                                       "Percentile 40: 90\n\n- 3 reasons follow")
    assert got[10] == ng.Reading(5.0, None, False)  # a full stop ending the sentence
    assert got[20] == ng.Reading(80.0, None, False)  # a word on the next line isn't a magnitude
    assert got[40] == ng.Reading(90.0, None, False)  # a list on the next line isn't a range
    units = ng.read_declared_percentiles("Percentile 10: 5 m\u00b2\nPercentile 20: 7 m/s\nPercentile 40: 9 km")
    assert units[10] == ng.Reading(5.0, None, False) and units[20] == ng.Reading(7.0, None, False)
    assert units[40] == ng.Reading(9.0, None, False)  # area, speed and km aren't magnitudes


@pytest.mark.parametrize(("unit", "magnitude"), [
    ("thousand people", 1e3), ("Millions of USD", 1e6), ("$M", 1e6), ("B $", 1e9), ("USD bn", 1e9),
    ("M USD", 1e6), ("people", 1), ("K", 1), ("m", 1), ("km", 1), ("MW", 1), (None, 1), ("trillions", 1e12),
])
def test_unit_magnitude(unit, magnitude):
    assert ng.unit_magnitude(unit) == magnitude


def test_g2_flags_millions_parsed_as_units():
    reason = ng.units_mismatch(pct(MILLIONS), percentile_text(MILLIONS, " million"), numeric_q())
    assert reason and "6 of 6" in reason


def test_g2_flags_percent_parsed_as_fraction_on_0_to_100():
    q = numeric_q(upper_bound=100, lower_bound=0, open_upper_bound=False, unit_of_measure="%")
    values = [5, 7, 9, 11, 13, 15]
    assert ng.units_mismatch(pct([v / 100 for v in values]), percentile_text(values, "%"), q)


def test_g2_does_not_flag_percent_on_0_to_1():
    q = numeric_q(upper_bound=1, lower_bound=0, open_upper_bound=False, unit_of_measure="share")
    values = [5, 7, 9, 11, 13, 15]
    assert ng.units_mismatch(pct([v / 100 for v in values]), percentile_text(values, "%"), q) is None


@pytest.mark.parametrize(
    ("question", "parsed", "text"),
    [
        (numeric_q(), [1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6], percentile_text(MILLIONS, " million")),
        (numeric_q(), [1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6],
         percentile_text([1_000_000, 1_500_000, 2_000_000, 2_400_000, 3_000_000, 3_500_000])),
        # unit already carries the magnitude: "2.4 billion" correctly parsed as 2.4
        (numeric_q(upper_bound=10, lower_bound=0, unit_of_measure="billion $"), [1, 1.5, 2, 2.4, 3, 3.5],
         percentile_text([1, 1.5, 2, 2.4, 3, 3.5], " billion")),
        # same unit, a small answer written in millions and correctly parsed in billions (at the range's edge)
        (numeric_q(upper_bound=10, lower_bound=0, unit_of_measure="billion $"),
         [0.01, 0.015, 0.02, 0.03, 0.04, 0.05], percentile_text([10, 15, 20, 30, 40, 50], " million")),
        # genuine near-zero and all-zero forecasts
        (numeric_q(upper_bound=5000), [0, 0.5, 1, 2, 2.5, 3], percentile_text([0, 0.5, 1, 2, 2.5, 3])),
        (numeric_q(upper_bound=10), [0, 0, 0, 0, 0, 0], percentile_text([0, 0, 0, 0, 0, 0])),
        # negative lower bound, log-scaled, unit not stated
        (numeric_q(upper_bound=50, lower_bound=-50, open_lower_bound=True), [-20, -10, -2, 3, 10, 20],
         percentile_text([-20, -10, -2, 3, 10, 20])),
        (numeric_q(upper_bound=1e6, lower_bound=1, open_lower_bound=True, zero_point=0),
         [10, 50, 200, 1000, 5000, 20000], percentile_text([10, 50, 200, 1000, 5000, 20000])),
        (numeric_q(unit_of_measure=None), [1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6],
         percentile_text([1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6])),
        # metres and kelvin abbreviations are not read as millions or thousands
        (numeric_q(upper_bound=100, lower_bound=0, unit_of_measure="m"), [5, 6, 7, 8, 9, 10],
         percentile_text([5, 6, 7, 8, 9, 10], " m")),
        (numeric_q(upper_bound=350, lower_bound=250, unit_of_measure="K"), [290, 295, 298, 300, 302, 305],
         percentile_text([290, 295, 298, 300, 302, 305], " K")),
        # no percentile lines at all
        (numeric_q(), MILLIONS, "I think about two million, give or take."),
    ],
)
def test_g2_does_not_flag(question, parsed, text):
    assert ng.units_mismatch(pct(parsed), text, question) is None


LOG = dict(upper_bound=1e7, lower_bound=1, open_lower_bound=True, zero_point=0)


@pytest.mark.parametrize(
    ("question", "parsed", "text"),
    [
        # the unit carries the magnitude and the forecaster restates it (linear and log ranges)
        (numeric_q(upper_bound=1e6, lower_bound=0, unit_of_measure="thousand people"),
         [300, 320, 340, 360, 380, 400], percentile_text([300, 320, 340, 360, 380, 400], " thousand")),
        (numeric_q(upper_bound=1e6, lower_bound=1, open_lower_bound=True, zero_point=0,
                   unit_of_measure="thousand people"),
         [300, 320, 340, 360, 380, 400], percentile_text([300, 320, 340, 360, 380, 400], " thousand")),
        # the unit is millions of dollars and the forecaster writes raw dollars
        (numeric_q(upper_bound=1e6, lower_bound=0.1, open_lower_bound=True, zero_point=0, unit_of_measure="million $"),
         [0.3, 0.32, 0.34, 0.36, 0.38, 0.4],
         percentile_text(["$300,000", "$320,000", "$340,000", "$360,000", "$380,000", "$400,000"])),
        # scientific notation, correctly parsed
        (numeric_q(upper_bound=1e9), [1.2e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6],
         percentile_text(["1.2e6", "1.5e6", "2e6", "2.4e6", "3e6", "3.5e6"])),
        # a 100x gap with the parsed value mid-range (not a squash)
        (numeric_q(upper_bound=1e6), [4e5, 4.2e5, 4.4e5, 4.6e5, 4.8e5, 5e5],
         percentile_text([4e3, 4.2e3, 4.4e3, 4.6e3, 4.8e3, 5e3])),
        # only 2 of 6 levels off
        (numeric_q(), [1.0, 1.5, 2e6, 2.4e6, 3e6, 3.5e6], percentile_text(MILLIONS, " million")),
        # opposite signs are never a units slip
        (numeric_q(upper_bound=50, lower_bound=-50, open_lower_bound=True), [-0.2, -0.1, -0.05, 0.03, 0.1, 0.2],
         percentile_text([20, 10, 5, 3, 10, 20])),
        # opposite signs at the edge of the range are never a units slip
        (numeric_q(upper_bound=1e6, lower_bound=-100, open_lower_bound=True),
         [-0.5, -0.4, -0.3, -0.2, -0.1, -0.05], percentile_text([5, 10, 20, 30, 40, 50])),
        # a 99x gap is below the ratio
        (numeric_q(), [v / 99 for v in [1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6]],
         percentile_text([1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6])),
    ],
)
def test_g2_more_cases_not_flagged(question, parsed, text):
    assert ng.units_mismatch(pct(parsed), text, question) is None


LOG_Q = dict(lower_bound=1, open_lower_bound=True, zero_point=0)


@pytest.mark.parametrize(
    ("question", "parsed", "written"),
    [
        # log-scaled questions where the forecaster uses an equivalent unit the reader doesn't know
        (numeric_q(upper_bound=1e6, unit_of_measure="MW", **LOG_Q), [12500, 13000, 14000, 15000, 16000, 17000],
         [f"{v} GW" for v in (12.5, 13, 14, 15, 16, 17)]),
        (numeric_q(upper_bound=1e4, unit_of_measure="USD", **LOG_Q), [0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
         [f"{v} cents" for v in (50, 55, 60, 65, 70, 75)]),
        (numeric_q(upper_bound=1e5, unit_of_measure="kilotonnes", **LOG_Q), [300, 320, 340, 360, 380, 400],
         [f"{v:,} tonnes" for v in (300000, 320000, 340000, 360000, 380000, 400000)]),
        (numeric_q(upper_bound=1e5, unit_of_measure="days", **LOG_Q), [730, 760, 790, 820, 850, 880],
         [f"{v} years" for v in (2, 2.08, 2.16, 2.25, 2.33, 2.41)]),
        (numeric_q(upper_bound=1e7, unit_of_measure="people", **LOG_Q), [150000, 160000, 170000, 180000, 190000, 200000],
         [f"{v} ('000s)" for v in (150, 160, 170, 180, 190, 200)]),
    ],
)
def test_g2_log_equivalent_units_not_flagged(question, parsed, written):
    text = "Reasoning.\n" + "\n".join(f"Percentile {int(p * 100)}: {w}" for p, w in zip(LEVELS, written))
    assert ng.units_mismatch(pct(parsed), text, question) is None


@pytest.mark.parametrize(
    ("question", "parsed", "written"),
    [
        (numeric_q(unit_of_measure="USD"), MILLIONS, [f"${v}M" for v in MILLIONS]),
        (numeric_q(upper_bound=5e9, unit_of_measure="USD"), MILLIONS, [f"{v}B" for v in MILLIONS]),
        (numeric_q(upper_bound=1e5), [24, 26, 28, 30, 32, 34], [f"{v}k" for v in (24, 26, 28, 30, 32, 34)]),
        (numeric_q(upper_bound=1e7, unit_of_measure="people", **LOG_Q), MILLIONS, [f"{v}M" for v in MILLIONS]),
    ],
)
def test_g2_single_letter_slips_flagged(question, parsed, written):
    text = "Reasoning.\n" + "\n".join(f"Percentile {int(p * 100)}: {w}" for p, w in zip(LEVELS, written))
    assert ng.units_mismatch(pct(parsed), text, question)


@pytest.mark.parametrize(
    ("question", "parsed", "text"),
    [
        # exactly 100x, squashed at the bottom of a linear range
        (numeric_q(upper_bound=1e8), [1e4, 1.5e4, 2e4, 2.4e4, 3e4, 3.5e4],
         percentile_text([1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6])),
        # log-scaled: "2.4 million" parsed as 2.4
        (numeric_q(**LOG), MILLIONS, percentile_text(MILLIONS, " million")),
        # every written value sits a little above the range (within the 25% slack)
        (numeric_q(upper_bound=1e6), [1.05, 1.08, 1.1, 1.13, 1.16, 1.2],
         percentile_text([1.05, 1.08, 1.1, 1.13, 1.16, 1.2], " million")),
        # a negative percentage range: "-5%" parsed as -0.05
        (numeric_q(upper_bound=0, lower_bound=-100, open_lower_bound=False, unit_of_measure="%"),
         [-0.15, -0.13, -0.11, -0.09, -0.07, -0.05], percentile_text([-15, -13, -11, -9, -7, -5], "%")),
        # discrete question with nominal bounds: counts written in thousands, parsed as units
        (None, [1, 1.2, 1.4, 1.6, 1.8, 2], percentile_text([1, 1.2, 1.4, 1.6, 1.8, 2], " thousand")),
    ],
)
def test_g2_more_cases_flagged(question, parsed, text):
    if question is None:
        question = DiscreteQuestion(
            **COMMON, question_text="How many?", page_url="https://example.test/d2", upper_bound=5000.5,
            lower_bound=-0.5, open_upper_bound=False, open_lower_bound=False, unit_of_measure="events",
            zero_point=None, cdf_size=5002,
        )
        question.nominal_lower_bound, question.nominal_upper_bound = 0, 5000
    assert ng.units_mismatch(pct(parsed), text, question)


# ---- CkaiBot overrides --------------------------------------------------------------


def make_bot(models=("openrouter/test/model-a",)):
    bot = ckai_bot.build_bot(publish=False, forecaster_models=list(models))
    bot.folder_to_save_reports_to = None
    return bot


class Capture:
    """A fake structure_output that records calls and answers from a queue."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def __call__(self, text_to_structure=None, output_type=None, *args, **kwargs):
        if text_to_structure is None:
            text_to_structure = args[0]
        if output_type is None:
            output_type = args[1] if len(args) > 1 else kwargs["output_type"]
        recorded = {k: (v.model if isinstance(v, GeneralLlm) else v) for k, v in kwargs.items()
                    if k not in ("text_to_structure", "output_type")}
        self.calls.append(dict(text=text_to_structure, output_type=output_type,
                               instructions=kwargs.get("additional_instructions"),
                               model=kwargs["model"].model, samples=kwargs.get("num_validation_samples"),
                               kwargs=recorded))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer(output_type) if callable(answer) else answer


def fake_reasoning(monkeypatch, text, seen=None):
    async def invoke(self, prompt, *args, **kwargs):
        if seen is not None:
            seen.append((self.model, prompt, args, kwargs))
        return text
    monkeypatch.setattr(GeneralLlm, "invoke", invoke)


def mc_question():
    return MultipleChoiceQuestion(**COMMON, question_text="Which colour?", options=OPTIONS,
                                  page_url="https://example.test/mc")


def mc_answer(output_type):
    opts = [po("Red", 0.5), po("Blue – dark", 0.3), po("Green", 0.2)]
    return PredictedOptionList(predicted_options=opts) if output_type is PredictedOptionList else opts


def test_pin_multiple_choice_matches_template(monkeypatch):
    """Everything but the parse type matches the template: the forecaster call, every parser
    keyword argument, and the result on a forecast that sums to 1."""
    bot, q = make_bot(), mc_question()
    template_seen, ours_seen = [], []
    fake_reasoning(monkeypatch, "Red: 50%, Blue – dark: 30%, Green: 20%", template_seen)
    template, ours = Capture([mc_answer]), Capture([mc_answer])
    monkeypatch.setattr(main, "structure_output", template)
    a = asyncio.run(main.FallTemplateBot2026._multiple_choice_prompt_to_forecast(bot, q, "PROMPT"))
    fake_reasoning(monkeypatch, "Red: 50%, Blue – dark: 30%, Green: 20%", ours_seen)
    monkeypatch.setattr(main, "structure_output", ours)
    b = asyncio.run(bot._multiple_choice_prompt_to_forecast(q, "PROMPT"))
    assert template_seen == ours_seen
    assert len(template.calls) == len(ours.calls) == 1
    assert template.calls[0]["text"] == ours.calls[0]["text"]
    assert template.calls[0]["kwargs"] == ours.calls[0]["kwargs"]
    assert template.calls[0]["output_type"] is PredictedOptionList
    assert ours.calls[0]["output_type"] == list[PredictedOption]
    assert a.reasoning == b.reasoning
    assert [(o.option_name, o.probability) for o in a.prediction_value.predicted_options] == \
        [(o.option_name, o.probability) for o in b.prediction_value.predicted_options]


def test_pin_numeric_matches_template(monkeypatch):
    bot, q = make_bot(), numeric_q()
    values = [1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6]
    fake_reasoning(monkeypatch, percentile_text(values))
    template, ours = Capture([pct(values)]), Capture([pct(values)])
    monkeypatch.setattr(main, "structure_output", template)
    template_seen, ours_seen = [], []
    fake_reasoning(monkeypatch, percentile_text(values), template_seen)
    a = asyncio.run(main.FallTemplateBot2026._numeric_prompt_to_forecast(bot, q, "PROMPT"))
    fake_reasoning(monkeypatch, percentile_text(values), ours_seen)
    monkeypatch.setattr(main, "structure_output", ours)
    b = asyncio.run(bot._numeric_prompt_to_forecast(q, "PROMPT"))
    assert template_seen == ours_seen
    assert len(template.calls) == len(ours.calls) == 1
    for key in ("text", "output_type", "kwargs"):
        assert template.calls[0][key] == ours.calls[0][key], key
    assert a.reasoning == b.reasoning
    assert a.prediction_value.get_cdf() == b.prediction_value.get_cdf()


def test_mc_override_rescales(monkeypatch):
    bot = make_bot()
    fake_reasoning(monkeypatch, "Red 50, Blue 30, Green 15")
    monkeypatch.setattr(main, "structure_output",
                        Capture([[po("Red", 0.5), po("blue - dark", 0.3), po("Green", 0.15)]]))
    result = asyncio.run(bot._multiple_choice_prompt_to_forecast(mc_question(), "PROMPT"))
    probs = [o.probability for o in result.prediction_value.predicted_options]
    assert [o.option_name for o in result.prediction_value.predicted_options] == OPTIONS
    assert sum(probs) == pytest.approx(1) and probs[0] == pytest.approx(0.5 / 0.95, abs=1e-3)


def test_mc_override_reparses_once_then_raises(monkeypatch):
    bot = make_bot()
    fake_reasoning(monkeypatch, "Red 70, Green 30")
    missing = [po("Red", 0.7), po("Green", 0.3)]
    capture = Capture([missing, [po("Red", 0.6), po("Blue – dark", 0.1), po("Green", 0.3)]])
    monkeypatch.setattr(main, "structure_output", capture)
    result = asyncio.run(bot._multiple_choice_prompt_to_forecast(mc_question(), "PROMPT"))
    assert len(capture.calls) == 2 and "exactly once" in capture.calls[1]["instructions"]
    assert [o.option_name for o in result.prediction_value.predicted_options] == OPTIONS
    monkeypatch.setattr(main, "structure_output", Capture([missing]))
    with pytest.raises(ValueError):
        asyncio.run(bot._multiple_choice_prompt_to_forecast(mc_question(), "PROMPT"))


def test_numeric_override_reparses_and_fixes(monkeypatch):
    bot = make_bot()
    fake_reasoning(monkeypatch, percentile_text(MILLIONS, " million"))
    capture = Capture([pct(MILLIONS), pct([v * 1e6 for v in MILLIONS])])
    monkeypatch.setattr(main, "structure_output", capture)
    result = asyncio.run(bot._numeric_prompt_to_forecast(numeric_q(), "PROMPT"))
    assert len(capture.calls) == 2 and "Units check" in capture.calls[1]["instructions"]
    assert result.prediction_value.declared_percentiles[0].value == pytest.approx(1e6)


def test_numeric_override_raises_when_reparse_still_wrong(monkeypatch):
    bot = make_bot()
    fake_reasoning(monkeypatch, percentile_text(MILLIONS, " million"))
    monkeypatch.setattr(main, "structure_output", Capture([pct(MILLIONS)]))
    with pytest.raises(ValueError, match="Guard G2"):
        asyncio.run(bot._numeric_prompt_to_forecast(numeric_q(), "PROMPT"))


# ---- end to end: how failed predictions decide a question -------------------------


@pytest.mark.parametrize(("failing", "published"), [(1, True), (3, False)])
def test_failed_predictions_rule(monkeypatch, failing, published):
    models = [f"openrouter/test/model-{c}" for c in "abcde"]
    bot = make_bot(models)
    assert bot.predictions_per_research_report == 5

    async def invoke(self, prompt, *args, **kwargs):
        return f"Written by {self.model}.\n" + percentile_text([1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6])
    monkeypatch.setattr(GeneralLlm, "invoke", invoke)
    bad = set(models[:failing])

    async def structure(text_to_structure=None, output_type=None, *args, **kwargs):
        text = text_to_structure if text_to_structure is not None else args[0]
        if re.search(r"Written by (\S+)\.", text).group(1) in bad:
            raise ValueError("parse failed")
        return pct([1e6, 1.5e6, 2e6, 2.4e6, 3e6, 3.5e6])
    monkeypatch.setattr(main, "structure_output", structure)
    result = asyncio.run(bot.forecast_questions([numeric_q()], return_exceptions=True))[0]
    assert isinstance(result, BaseException) != published


def test_backtest_forecast_one_propagates_a_guard_raise(monkeypatch):
    from backtest import run as bt

    bot = make_bot()
    fake_reasoning(monkeypatch, percentile_text(MILLIONS, " million"))
    monkeypatch.setattr(main, "structure_output", Capture([pct(MILLIONS)]))
    q = {"id": "x1", "question_type": "numeric", "freeze_datetime": "2026-10-01T00:00:00+00:00",
         "question": "How many people?", "background": "", "resolution_criteria": "", "unit": "people",
         "lower_bound": 0, "upper_bound": 5_000_000}
    monkeypatch.setattr(bt, "to_question", lambda _q: numeric_q())
    with pytest.raises(ValueError, match="Guard G2"):
        asyncio.run(bt.forecast_one(bot, q, 0))
