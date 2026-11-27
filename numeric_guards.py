"""
Guards for multiple-choice and numeric forecasts (no model calls here).

G1, multiple choice: match parsed option names to the question's options on a
normalised key, require each option once, and rescale probability sums between
0.90 and 1.10 (0.99 to 1.01 is left to the library) instead of letting the
library reject the whole prediction.

G2, numeric units: compare the parser's percentiles with the forecaster's own
"Percentile NN: value" lines (the "2.4 million parsed as 2.4" slip, or a
percentage parsed as a fraction). Each written value is turned into the
question's unit in every way it could be meant: magnitude words applied and
divided by a magnitude the unit carries ("300 thousand" on a "thousand people"
question is 300), a bare number read both as written and in the unit's
magnitude ("$300,000" on a "million $" question may be 0.3), and a single
letter also read as the unit itself when the question's unit names it ("5 m"
on a metres question). A level is flagged only if every reading differs from
the parser's by 100x or more and the reading sits inside the range (25% slack),
and then: on linear questions, the parser's value sits in the outer 1% of the
range; on log-scaled questions, the parser returned the written number with its
magnitude word dropped (or a written percentage divided by 100), the slip's own
signature, so an equivalent unit the reader doesn't know ("12.5 GW" on an MW
question) never flags. The forecast is flagged when at least 3 levels are found
in both and at least half of them are flagged. Round 7's 672 saved forecasts:
none flagged.

Used by CkaiBot (ckai_bot.py); see benchmarks/PLAN-numeric-guards.md (local).
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Sequence
from decimal import Decimal
from typing import NamedTuple

from forecasting_tools import Percentile, PredictedOption, PredictedOptionList

logger = logging.getLogger(__name__)

# ---- G1: multiple choice ---------------------------------------------------------

SUM_UNCHANGED = (0.99, 1.01)  # the library normalises these itself
SUM_RESCALED = (0.90, 1.10)

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}
_OPTION_PREFIX = re.compile(r"^option\s*(?:[a-z0-9]{1,2}\s*[:.)\-]|[:.)\-])?\s*", re.IGNORECASE)


def option_key(name: str) -> str:
    """Casefold, unify dashes and quotes, collapse spaces."""
    text = unicodedata.normalize("NFKC", str(name)).translate(_DASHES).translate(_QUOTES)
    return " ".join(text.split()).strip().casefold()


def _strip_option_prefix(name: str) -> str:
    return _OPTION_PREFIX.sub("", str(name).strip(), count=1)


def match_options(parsed: Sequence[PredictedOption], options: Sequence[str]) -> list[PredictedOption]:
    """Parsed options in the question's order with its canonical names. Raises
    ValueError if any option is missing, extra, duplicated or ambiguous."""
    options = [str(o) for o in options]
    keys = [option_key(o) for o in options]
    unique_keys = len(set(keys)) == len(keys)
    by_key = {k: o for k, o in zip(keys, options)} if unique_keys else {}
    # Only strip an "Option ..." prefix when no real option name starts with "option".
    may_strip = not any(k.startswith("option") for k in keys)
    found: dict[str, float] = {}
    for item in parsed:
        name = str(item.option_name)
        canonical = None
        if name in options:
            canonical = name
        elif by_key:
            canonical = by_key.get(option_key(name))
            if canonical is None and may_strip:
                stripped = _strip_option_prefix(name)
                if stripped != name:
                    canonical = by_key.get(option_key(stripped))
        if canonical is None:
            raise ValueError(f"Parsed option {name!r} matches none of {options}")
        if canonical in found:
            raise ValueError(f"Option {canonical!r} parsed more than once")
        found[canonical] = float(item.probability)
    missing = [o for o in options if o not in found]
    if missing:
        raise ValueError(f"Parsed options miss {missing}")
    return [PredictedOption(option_name=o, probability=found[o]) for o in options]


def build_option_list(
    parsed: Sequence[PredictedOption], options: Sequence[str]
) -> tuple[PredictedOptionList, float | None]:
    """A PredictedOptionList from parsed options. Returns (list, rescaled_from):
    rescaled_from is the original sum when it was rescaled, else None."""
    matched = match_options(parsed, options)
    probs = [p.probability for p in matched]
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs):
        raise ValueError(f"Option probabilities out of range: {probs}")
    total = sum(probs)
    rescaled_from = None
    if not SUM_UNCHANGED[0] <= total <= SUM_UNCHANGED[1]:
        if not SUM_RESCALED[0] <= total <= SUM_RESCALED[1]:
            raise ValueError(f"Option probabilities sum to {total:.3f}, too far from 1 to rescale")
        matched = [
            PredictedOption(option_name=p.option_name, probability=p.probability / total) for p in matched
        ]
        rescaled_from = total
    return PredictedOptionList(predicted_options=matched), rescaled_from


# ---- G2: numeric units -----------------------------------------------------------

RATIO = 100.0  # the two readings must differ by at least this factor
EDGE = 0.01  # "at the edge": within 1% of the range from a bound (log space on log-scaled questions)
SLACK = 0.25  # the forecaster's reading may sit up to 25% of the range outside it

_WORD_MAGNITUDE = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6, "mm": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
}
_LETTER_MAGNITUDE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_PERCENTILE_LINE = re.compile(
    r"percentile\s*(\d{1,2})\s*[*_]*\s*[:=]\s*[*_]*\s*[~≈]?\s*(?P<sign>[-−])?\s*[$€£]?\s*"
    r"(?P<num>(?>\d[\d,]*(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)(?!\d|\.\d)"
    r"(?:[ \t]*(?P<mag>thousand|million|billion|trillion|mn|mm|bn|tn|k|m|b|t)(?![a-z]))?"
    r"[ \t]*(?P<pct>%)?"
    r"(?P<range>[ \t]*(?:to|-|–|—)[ \t]*[$€£]?\d)?",
    re.IGNORECASE,
)
# Magnitudes carried by a question's unit: words ("thousand people", "Millions of USD", "bn") or a
# single letter next to a currency ("$M", "B $", "M USD"). A bare "m" or "K" (metres, kelvin) isn't one.
_UNIT_WORD = re.compile(r"\b(thousands?|millions?|billions?|trillions?|mn|mm|bn|tn)\b", re.IGNORECASE)
_UNIT_LETTER = re.compile(
    r"(?:[$€£]|\b(?:usd|eur|gbp)\b)\s*([kmbt])\b|\b([kmbt])\s*(?:[$€£]|\b(?:usd|eur|gbp)\b)",
    re.IGNORECASE,
)


class Reading(NamedTuple):
    """One "Percentile NN:" value as written: the signed number, its magnitude word as written (if any)
    and whether it carried %."""

    number: float
    word: str | None
    percent: bool


def read_declared_percentiles(text: str) -> dict[int, Reading]:
    """The last value written for each "Percentile NN:" level. Values written as a range
    ("1,000 to 2,000") are skipped. Empty if none are found."""
    found: dict[int, Reading] = {}
    for m in _PERCENTILE_LINE.finditer(text or ""):
        if m.group("range"):
            continue
        try:
            number = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        if m.group("sign"):
            number = -number
        word = m.group("mag") or None
        found[int(m.group(1))] = Reading(number, word, bool(m.group("pct")))
    return found


def unit_magnitude(unit: str | None) -> float:
    """The magnitude a question's unit carries (1e6 for "million $"), else 1."""
    if not unit:
        return 1.0
    m = _UNIT_WORD.search(unit)
    if m:
        word = m.group(1).lower()
        return _WORD_MAGNITUDE[word[:-1] if word.endswith("s") and word[:-1] in _WORD_MAGNITUDE else word]
    m = _UNIT_LETTER.search(unit)
    if m:
        return _LETTER_MAGNITUDE[(m.group(1) or m.group(2)).lower()]
    return 1.0


# A single letter after a number is read as the unit itself (not a magnitude) only when the question's
# unit names that unit: "5 m" on a metres question, "300 K" on kelvin, "5 t" on tonnes, "8 b" on bytes.
_LITERAL_LETTER_UNITS = {
    "m": re.compile(r"^\s*m\s*$|\bmet(?:re|er)s?\b", re.IGNORECASE),
    "k": re.compile(r"^\s*k\s*$|\bkelvin\b", re.IGNORECASE),
    "t": re.compile(r"^\s*t\s*$|\btonnes?\b|\btons?\b", re.IGNORECASE),
    "b": re.compile(r"^\s*b\s*$|\bbytes?\b", re.IGNORECASE),
}


def _candidates(reading: Reading, unit_mag: float, small_range: bool, unit: str | None) -> list[float]:
    """What the forecaster's written value could mean in the question's unit."""
    values = []
    if reading.word:
        word = reading.word.lower()
        values.append(reading.number * _WORD_MAGNITUDE[word] / unit_mag)
        if len(word) == 1 and unit and _LITERAL_LETTER_UNITS[word].search(unit):
            values.append(reading.number)
    else:
        values.append(reading.number)
        if unit_mag != 1:  # "$300,000" on a "million $" question
            values.append(reading.number / unit_mag)
    if reading.percent and small_range:  # a 0-to-1 range: "5%" means 0.05
        values += [v / 100 for v in values]
    return values


def _bounds(question) -> tuple[float, float]:
    lower = getattr(question, "nominal_lower_bound", None)
    upper = getattr(question, "nominal_upper_bound", None)
    lower = question.lower_bound if lower is None else lower
    upper = question.upper_bound if upper is None else upper
    return float(lower), float(upper)


def _position(x: float, lower: float, upper: float, zero_point: float | None) -> float:
    """Where x sits in the range: 0 at the lower bound, 1 at the upper (log space if log-scaled)."""
    if zero_point is not None and lower > zero_point and upper > lower:
        if x <= zero_point:
            return -math.inf
        span = math.log(upper - zero_point) - math.log(lower - zero_point)
        return (math.log(x - zero_point) - math.log(lower - zero_point)) / span
    return (x - lower) / (upper - lower)


def _level_mismatch(parsed: float, read: float, lower: float, upper: float, zero_point: float | None) -> bool:
    if parsed == 0 or read == 0 or (parsed > 0) != (read > 0):
        return False
    if abs(math.log10(abs(parsed)) - math.log10(abs(read))) < math.log10(RATIO) - 1e-9:
        return False
    if upper <= lower:
        return False
    read_position = _position(read, lower, upper, zero_point)
    parsed_position = _position(parsed, lower, upper, zero_point)
    read_plausible = -SLACK <= read_position <= 1 + SLACK
    if zero_point is not None:
        # Log-scaled: a 1000x slip still lands inside the range in log terms, so position can't tell
        # it apart. units_mismatch requires the slip's own signature there instead.
        return read_plausible
    parsed_at_edge = parsed_position < EDGE or parsed_position > 1 - EDGE
    return read_plausible and parsed_at_edge


def _slip_signature(parsed: float, reading: Reading) -> bool:
    """The parser returned the written number with its magnitude word dropped, or a written
    percentage divided by 100: the shape of a units slip. An unknown but equivalent unit
    ("12.5 GW" on an MW question) never has it."""
    def near(a: float, b: float) -> bool:
        return b != 0 and abs(a - b) <= 0.01 * abs(b)
    if reading.word and near(parsed, reading.number):
        return True
    return reading.percent and near(parsed, reading.number / 100)


def units_mismatch(parsed: Sequence[Percentile], text: str, question) -> str | None:
    """A reason string if the parser's percentiles look like a units slip against the
    forecaster's own lines, else None. Needs at least 3 levels found in both, and at
    least half of those flagged."""
    readings = read_declared_percentiles(text)
    if not readings:
        logger.debug("Guard G2: no 'Percentile NN:' lines found; units not checked.")
        return None
    lower, upper = _bounds(question)
    zero_point = getattr(question, "zero_point", None)
    unit = getattr(question, "unit_of_measure", None)
    unit_mag = unit_magnitude(unit)
    small_range = max(abs(lower), abs(upper)) <= 1.5
    checked = flagged = 0
    examples = []
    for p in parsed:
        level = int(round(p.percentile * 100))
        if level not in readings:
            continue
        reading = readings[level]
        checked += 1
        if zero_point is not None and not _slip_signature(p.value, reading):
            continue  # log-scaled: only the slip's own signature can flag a level
        if all(_level_mismatch(p.value, c, lower, upper, zero_point)
               for c in _candidates(reading, unit_mag, small_range, unit)):
            flagged += 1
            written = f"{reading.number:g}{' ' + reading.word if reading.word else ''}{'%' if reading.percent else ''}"
            examples.append(f"p{level}: parsed {p.value:g}, written {written}")
    if checked >= 3 and flagged * 2 >= checked:
        return f"{flagged} of {checked} percentiles off by 100x or more ({'; '.join(examples[:3])})"
    return None


def _plain(x: float) -> str:
    """A number without scientific notation."""
    text = format(Decimal(repr(float(x))), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def reparse_instructions(question) -> str:
    """Extra parser instructions for the one re-parse after a units flag."""
    lower, upper = _bounds(question)
    unit = question.unit_of_measure or "(unit not stated)"
    return (
        f"- Units check: the question's range is {_plain(lower)} to {_plain(upper)} in the unit: {unit}.\n"
        "- Write every value as a plain number in that unit. Write magnitude words out in full: "
        "'2.4 million' is 2400000 when the unit is people, but 2.4 when the unit is 'million people' "
        "or 'M'.\n"
        "- Percentages: when the range runs 0 to 100, '5%' is 5, not 0.05."
    )


def mc_reparse_instructions(options: Sequence[str]) -> str:
    """Extra parser instructions for the one re-parse after a G1 failure."""
    return (
        f"- List every option exactly once, using exactly these names: {list(options)}.\n"
        "- Give each probability as a decimal between 0 and 1; together they should sum to 1."
    )
