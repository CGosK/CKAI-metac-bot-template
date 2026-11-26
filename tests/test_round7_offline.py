"""Round 7 (benchmarks/PLAN-refine-round7-numeric.md): offline checks of scoring, ranges, walk and run.py
question objects."""

import math


from backtest import round7 as r7
from backtest import run
from backtest import scoring_numeric as sn


def test_bucket_matches_metaculus_rule():
    assert sn.bucket(-1, 0, 10) == 0 and sn.bucket(11, 0, 10) == 201
    assert sn.bucket(10, 0, 10) == 200 and sn.bucket(0, 0, 10) == 1
    assert sn.bucket(0.05, 0, 10) == 1  # location x 200 = 1.0 exactly: int(1.0 + 1 - 1e-10) = 1
    assert sn.bucket(0.051, 0, 10) == 2
    assert sn.bucket(5, 0, 10) == 100


def test_baseline_scores_worked_cases():
    uniform = [0.0] + [i / 200 for i in range(1, 200)] + [1.0]  # all mass inbound, evenly
    assert abs(sn.numeric_baseline(uniform, 3, 0, 10) - 100 * math.log(1 / 0.9) / 2) < 1e-9
    oob = [0.05] + [0.05 + 0.9 * i / 200 for i in range(1, 200)] + [0.95]
    assert abs(sn.numeric_baseline(oob, 12, 0, 10)) < 1e-9  # 0.05 out of bounds scores 0
    assert abs(sn.mc_baseline([0.25] * 4, 2)) < 1e-12
    assert abs(sn.mc_baseline([0.7, 0.1, 0.1, 0.1], 0) - 100 * math.log(2.8) / math.log(4)) < 1e-9


def test_calibration_helpers_and_widen():
    p = [[0.1, -1.0], [0.2, -0.5], [0.4, -0.1], [0.6, 0.1], [0.8, 0.5], [0.9, 1.0]]
    assert sn.median(p) == 0.0 and sn.covered(p, 0.9) == 1 and sn.covered(p, 1.1) == 0
    assert sn.below_median(p, -0.01) == 1 and sn.below_median(p, 0.01) == 0
    w = sn.widen(p, 2.0)
    assert w[0][1] == -2.0 and w[-1][1] == 2.0 and sn.median(w) == 0.0
    cdf = sn.library_cdf(sn.normal_percentiles(0, 1), -5, 5)
    assert len(cdf) == 201 and cdf[0] > 0 and cdf[-1] < 1  # open bounds keep tail mass
    assert 0.45 < sn.pit(cdf, 0, -5, 5) < 0.55


def test_ranges_and_mc_options():
    st = {"anchor": 100.0, "mean_change": 1.0, "spread": 2.0, "places": 2}
    r = r7.numeric_range(st, 6, "salt")
    assert abs((r["upper_bound"] - r["lower_bound"]) - 24) < 0.02  # 2 x k x s
    assert 101 - 3 - 1e-9 <= r["midpoint"] <= 101 + 3 + 1e-9  # v in [-1.5, 1.5] spreads
    assert r["lower_shown"].count(".") == 1 and "e" not in r["upper_shown"]
    m = r7.mc_options(st, "salt")
    assert len(m["options"]) == 4 and m["cuts"] == sorted(m["cuts"])
    assert r7.mc_index(m["cuts"], m["cuts"][0] - 1) == 0 and r7.mc_index(m["cuts"], m["cuts"][2]) == 3


def test_walk_balances_k_and_caps_family():
    cands = [{"key": f"s{i % 50}", "family": f"f{i % 10}", "k": r7.KS[i % 4], "ok": i % 9 != 0} for i in range(300)]
    prim, back = r7.walk(cands)
    assert len(prim) == 12 and len(back) == 2
    assert all(sum(c["k"] == k for c in prim) == 3 for k in r7.KS)
    assert len({c["key"] for c in prim + back}) == 14


def test_scored_replaces_by_same_k():
    def q(i, role, rank, k, outcome=1.0):
        return {"id": str(i), "question_type": "numeric", "cluster": "2026-08-05", "horizon_days": 7,
                "role": role, "rank": rank, "k": k, "outcome": outcome}
    qs = [q(i, "primary", i, r7.KS[i % 4], None if i == 2 else 1.0) for i in range(12)]
    qs += [q(100, "backup", 0, 3), q(101, "backup", 1, r7.KS[2])]
    keep, notes = r7.scored(qs, {x["id"] for x in qs})
    assert len(keep) == 12 and "101" in {x["id"] for x in keep}  # same k (12) chosen over rank order
    assert notes["2026-08-05 h7"]["missing_primaries"] == 1
    short = [q(i, "primary", i, r7.KS[i % 4], None if i < 4 else 1.0) for i in range(12)]
    keep2, notes2 = r7.scored(short, {x["id"] for x in short})
    assert keep2 == [] and not notes2["2026-08-05 h7"]["scored"]  # 8 left, below MIN_SCORED_PER_CELL


def test_run_builds_numeric_and_mc_questions():
    from forecasting_tools import MultipleChoiceQuestion, NumericQuestion

    base = {"id": "x", "question": "What will S be on 2026-09-01?", "background": "", "criteria": "c"}
    nq = run.to_question({**base, "question_type": "numeric", "lower_bound": 1.0, "upper_bound": 9.0, "units": "%"})
    assert isinstance(nq, NumericQuestion) and nq.open_lower_bound and nq.cdf_size == 201
    mq = run.to_question({**base, "question_type": "multiple_choice", "options": ["a", "b"]})
    assert isinstance(mq, MultipleChoiceQuestion) and mq.options == ["a", "b"]
    from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

    dist = NumericDistribution(declared_percentiles=[Percentile(percentile=p, value=v) for p, v in
                                                     sn.normal_percentiles(5, 1)],
                               open_upper_bound=True, open_lower_bound=True, upper_bound=9.0, lower_bound=1.0,
                               zero_point=None, cdf_size=201)
    row = run.prediction_fields("numeric", dist)
    assert len(row["cdf"]) == 201 and len(row["percentiles"]) == 6 and abs(row["prediction"] - 5) < 1e-6


def test_submitted_cdf_and_strict_check():
    p = sn.normal_percentiles(0, 1)
    sub = sn.submitted(p, -5, 5)
    assert len(sub) == 201 and abs(sub[0] - 0.0508) < 5e-4  # the aggregation pass moves tail mass (review)
    assert sn.strict_ok(p, -5, 5)
    assert not sn.strict_ok(sn.normal_percentiles(0, 40), -5, 5)  # far beyond the bounds: the live parser rejects
