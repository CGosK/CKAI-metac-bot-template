"""
Score backtest batches (no model calls).

Reads backtest/data/questions.json and the predictions of one or more batches,
and reports, per question and overall:
- Brier and log scores for the bot (median of its predictions, as the live bot
  aggregates), the crowd at freeze, and an out-of-fold base rate;
- a calibration table;
- the pre-registered comparisons against the current bot (median of 5, no
  transform), each paired, with group-bootstrapped 95% intervals and a Holm
  correction: (a) extremizing fitted by group-wise cross-validation,
  (b) 1 and 3 predictions per question, (c) mean and trimmed mean.

A positive "gain" means the alternative scores better (lower Brier).

Usage:
    uv run python -m backtest.score pilot [main ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "data" / "questions.json"
RUNS = HERE / "runs"
OUT = HERE / "out"

CLIP = (0.01, 0.99)  # main.py clips each prediction to this range
FOLDS = 5
BOOTSTRAP = 2000
SEED = 20261124
ALPHAS = [x / 20 for x in range(10, 61)]  # extremizing strengths 0.5 to 3.0


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def log_score(p: float, y: int) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -math.log(p if y else 1 - p)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def extremize(p: float, alpha: float) -> float:
    return min(max(1 / (1 + math.exp(-alpha * logit(p))), CLIP[0]), CLIP[1])


def trimmed_mean(values: list[float]) -> float:
    v = sorted(values)
    if len(v) >= 5:
        v = v[1:-1]
    return statistics.fmean(v)


def load(batches: list[str]) -> tuple[dict, list[dict], dict[str, list[float]]]:
    meta = json.loads(QUESTIONS.read_text())
    by_id = {q["id"]: q for q in meta["questions"]}
    preds: dict[str, list[float]] = {}
    seen_in: dict[str, str] = {}
    for batch in batches:
        launch = (RUNS / batch / "launch.txt").read_text()
        if f"questions_sha256: {meta['sha256']}" not in launch:
            sys.exit(f"{batch} was run on a different question list")
        for line in (RUNS / batch / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("prediction") is None:
                continue
            if seen_in.setdefault(row["id"], batch) != batch:
                sys.exit(f"question {row['id']} appears in {seen_in[row['id']]} and {batch}")
            preds.setdefault(row["id"], []).append(float(row["prediction"]))
    return meta, [by_id[i] for i in preds], preds


def folds_by_group(questions: list[dict]) -> dict[str, int]:
    groups = sorted({q["group"] for q in questions})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    return {g: i % FOLDS for i, g in enumerate(groups)}


def fit_alpha(pairs: list[tuple[float, int]]) -> float:
    return min(ALPHAS, key=lambda a: sum(brier(extremize(p, a), y) for p, y in pairs))


def group_bootstrap(diffs: dict[str, float], groups: dict[str, str]) -> tuple[float, float, float, float]:
    """Mean paired difference, 95% interval and two-sided p, resampling groups."""
    by_group: dict[str, list[float]] = {}
    for qid, d in diffs.items():
        by_group.setdefault(groups[qid], []).append(d)
    keys = list(by_group)
    n = len(diffs)
    mean = sum(diffs.values()) / n
    rng = random.Random(SEED)
    stats = []
    for _ in range(BOOTSTRAP):
        sample = [d for g in rng.choices(keys, k=len(keys)) for d in by_group[g]]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    lo, hi = stats[int(0.025 * BOOTSTRAP)], stats[int(0.975 * BOOTSTRAP) - 1]
    below = sum(s <= 0 for s in stats) / BOOTSTRAP
    above = sum(s >= 0 for s in stats) / BOOTSTRAP
    return mean, lo, hi, min(1.0, 2 * min(below, above))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def subsample_brier(values: list[float], k: int, y: int) -> float:
    """Expected Brier of the median of k predictions drawn from the saved ones."""
    combos = list(itertools.combinations(values, k))
    return statistics.fmean(brier(statistics.median(c), y) for c in combos)


def analyse(questions: list[dict], preds: dict[str, list[float]], predictions: int) -> dict:
    full = [q for q in questions if len(preds[q["id"]]) >= predictions]
    groups = {q["id"]: q["group"] for q in full}
    y = {q["id"]: q["outcome"] for q in full}
    current = {q["id"]: statistics.median(preds[q["id"]][:predictions]) for q in full}
    base_brier = {i: brier(current[i], y[i]) for i in current}

    fold_of = folds_by_group(full)
    fold = {q["id"]: fold_of[q["group"]] for q in full}
    base_rate, extremized, alphas = {}, {}, []
    for f in range(FOLDS):
        train = [i for i in current if fold[i] != f]
        test = [i for i in current if fold[i] == f]
        if not train or not test:
            continue
        rate = sum(y[i] for i in train) / len(train)
        alpha = fit_alpha([(current[i], y[i]) for i in train])
        alphas.append(alpha)
        for i in test:
            base_rate[i] = rate
            extremized[i] = extremize(current[i], alpha)

    alternatives = {
        "a_extremize_cv": {i: brier(extremized[i], y[i]) for i in extremized},
        "b_1_prediction": {i: subsample_brier(preds[i][:predictions], 1, y[i]) for i in current},
        "b_3_predictions": {i: subsample_brier(preds[i][:predictions], 3, y[i]) for i in current},
        "c_mean": {i: brier(statistics.fmean(preds[i][:predictions]), y[i]) for i in current},
        "c_trimmed_mean": {i: brier(trimmed_mean(preds[i][:predictions]), y[i]) for i in current},
    }
    comparisons, pvalues = {}, {}
    for name, scores in alternatives.items():
        diffs = {i: base_brier[i] - scores[i] for i in scores}
        mean, lo, hi, p = group_bootstrap(diffs, groups)
        comparisons[name] = {"gain": mean, "ci95": [lo, hi], "p": p, "n": len(diffs)}
        pvalues[name] = p
    for name, p in holm(pvalues).items():
        comparisons[name]["p_holm"] = p
        comparisons[name]["passes"] = p < 0.05 and comparisons[name]["gain"] > 0

    crowd = {q["id"]: q["crowd"] for q in full}
    bot_vs_crowd = group_bootstrap(
        {i: brier(crowd[i], y[i]) - base_brier[i] for i in current}, groups
    )
    calibration = []
    for b in range(10):
        lo_b, hi_b = b / 10, (b + 1) / 10
        ids = [i for i in current if lo_b <= current[i] < hi_b or (b == 9 and current[i] == 1)]
        if ids:
            calibration.append({
                "bin": f"{lo_b:.1f}-{hi_b:.1f}",
                "n": len(ids),
                "mean_forecast": statistics.fmean(current[i] for i in ids),
                "yes_rate": statistics.fmean(y[i] for i in ids),
            })
    n = len(current)
    return {
        "questions_scored": n,
        "groups": len(set(groups.values())),
        "questions_with_fewer_predictions": len(questions) - n,
        "yes_rate": sum(y.values()) / n if n else None,
        "brier": {
            "bot_median": statistics.fmean(base_brier.values()) if n else None,
            "crowd": statistics.fmean(brier(crowd[i], y[i]) for i in current) if n else None,
            "base_rate_out_of_fold": statistics.fmean(brier(base_rate[i], y[i]) for i in base_rate)
            if base_rate else None,
        },
        "log_score": {
            "bot_median": statistics.fmean(log_score(current[i], y[i]) for i in current) if n else None,
            "crowd": statistics.fmean(log_score(crowd[i], y[i]) for i in current) if n else None,
        },
        "crowd_minus_bot_brier": {"mean": bot_vs_crowd[0], "ci95": [bot_vs_crowd[1], bot_vs_crowd[2]]},
        "extremize_alpha_per_fold": alphas,
        "comparisons": comparisons,
        "calibration": calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--predictions", type=int, default=5)
    args = parser.parse_args()
    meta, questions, preds = load(args.batches)
    result = analyse(questions, preds, args.predictions)
    result["batches"] = args.batches
    result["questions_sha256"] = meta["sha256"]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"score-{'+'.join(args.batches)}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("questions_scored", "groups", "yes_rate", "brier",
                                             "crowd_minus_bot_brier", "comparisons")}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
