"""
Numeric and multiple-choice scoring for round 7 (benchmarks/PLAN-refine-round7-numeric.md), no model calls.

Baseline scores as Metaculus computes them (scores FAQ; github.com/Metaculus/metaculus scoring/score_math.py
lines 97-103 and utils/the_math/formulas.py unscaled_location_to_bucket_index, commit db277a4c), implemented
here for linear (zero_point None) questions:
- numeric: pmf = [cdf[0], diffs..., 1 - cdf[-1]]; 100 * ln(pmf[b] / base) / 2, base 0.05 for an out-of-bounds
  bucket, else (1 - 0.05 * open bounds) / inbound buckets;
- multiple choice: 100 * ln(p * N) / ln N.
Plus calibration from declared percentiles, PIT from the final CDF, the widening transform and the reference
forecasters, all through the same library CDF path the live bot submits.
"""

from __future__ import annotations

import math
from statistics import NormalDist

N_INBOUND = 200  # cdf_size 201
DECLARED = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)  # the template prompt's percentiles


def pmf(cdf: list[float]) -> list[float]:
    return [cdf[0], *(b - a for a, b in zip(cdf, cdf[1:])), 1 - cdf[-1]]


def bucket(value: float, lower: float, upper: float, inbound: int = N_INBOUND) -> int:
    """formulas.unscaled_location_to_bucket_index for a linear question."""
    loc = (value - lower) / (upper - lower)
    if loc < 0:
        return 0
    if loc > 1:
        return inbound + 1
    if loc == 1:
        return inbound
    return max(int(loc * inbound + 1 - 1e-10), 1)


def numeric_baseline(cdf: list[float], value: float, lower: float, upper: float, open_bounds: int = 2) -> float:
    p = pmf(cdf)
    b = bucket(value, lower, upper, len(p) - 2)
    base = 0.05 if b in (0, len(p) - 1) else (1 - 0.05 * open_bounds) / (len(p) - 2)
    return 100 * math.log(p[b] / base) / 2


def mc_baseline(probs: list[float], index: int) -> float:
    n = len(probs)
    return 100 * math.log(probs[index] * n) / math.log(n)


def declared_at(percentiles: list[list[float]], level: float) -> float:
    return min(percentiles, key=lambda pv: abs(pv[0] - level))[1]


def covered(percentiles: list[list[float]], value: float) -> int:
    """Outcome inside the declared [p10, p90]."""
    return int(declared_at(percentiles, 0.1) <= value <= declared_at(percentiles, 0.9))


def median(percentiles: list[list[float]]) -> float:
    """The declared p50, or the midpoint of p40 and p60 (the template asks for 10/20/40/60/80/90)."""
    if any(abs(p - 0.5) < 1e-9 for p, _ in percentiles):
        return declared_at(percentiles, 0.5)
    return (declared_at(percentiles, 0.4) + declared_at(percentiles, 0.6)) / 2


def below_median(percentiles: list[list[float]], value: float) -> int:
    return int(value < median(percentiles))


def pit(cdf: list[float], value: float, lower: float, upper: float) -> float:
    """The final CDF at the outcome (linear interpolation between the 201 points)."""
    loc = (value - lower) / (upper - lower)
    if loc <= 0:
        return cdf[0] if loc == 0 else cdf[0] / 2
    if loc >= 1:
        return cdf[-1] + (1 - cdf[-1]) / 2 if loc > 1 else cdf[-1]
    x = loc * (len(cdf) - 1)
    i = int(x)
    return cdf[i] + (cdf[min(i + 1, len(cdf) - 1)] - cdf[i]) * (x - i)


def library_cdf(percentiles: list[list[float]], lower: float, upper: float, strict: bool = False) -> list[float]:
    """One forecast's CDF from its declared percentiles (forecasting-tools, open bounds). strict=True applies the
    live parser's validation (from_question is strict) and raises where the live bot would reject it."""
    from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

    dist = NumericDistribution(
        declared_percentiles=[Percentile(percentile=p, value=v) for p, v in percentiles],
        open_upper_bound=True, open_lower_bound=True, upper_bound=upper, lower_bound=lower, zero_point=None,
        cdf_size=N_INBOUND + 1, strict_validation=strict)
    return [p.percentile for p in dist.get_cdf()]


def live_cdf(cdf: list[float], lower: float, upper: float) -> list[float]:
    """What the live bot submits for one forecast: NumericReport.aggregate_predictions takes the pointwise median
    of the forecasts' CDFs (here one) and passes it through NumericDistribution.from_question again, which
    standardizes a second time (ckai_bot runs 5 forecasts; the backtest runs one per arm)."""
    from forecasting_tools import NumericQuestion
    from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

    q = NumericQuestion(question_text="x", upper_bound=upper, lower_bound=lower, open_upper_bound=True,
                        open_lower_bound=True, zero_point=None, cdf_size=len(cdf))
    xs = [lower + (upper - lower) * (i / (len(cdf) - 1)) for i in range(len(cdf))]  # as the library computes it
    dist = NumericDistribution.from_question([Percentile(value=x, percentile=c) for x, c in zip(xs, cdf)], q)
    return [p.percentile for p in dist.get_cdf()]


def submitted(percentiles: list[list[float]], lower: float, upper: float) -> list[float]:
    """Declared percentiles to the live submission: library CDF, then the aggregation pass."""
    return live_cdf(library_cdf(percentiles, lower, upper), lower, upper)


def strict_ok(percentiles: list[list[float]], lower: float, upper: float) -> bool:
    try:
        library_cdf(percentiles, lower, upper, strict=True)
        return True
    except Exception:  # the live parser would reject this forecast
        return False


def widen(percentiles: list[list[float]], w: float) -> list[list[float]]:
    """Distances from the declared median scaled by w."""
    mid = median(percentiles)
    return [[p, mid + w * (v - mid)] for p, v in percentiles]


def normal_percentiles(mu: float, sd: float) -> list[list[float]]:
    return [[p, NormalDist(mu, sd).inv_cdf(p)] for p in DECLARED]
