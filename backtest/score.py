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
    uv run python -m backtest.score pilot main --compare research --predictions 3

With --compare, the one pre-registered comparison of PLAN-research-backtest.md:
the compared batch's median of its predictions against the baseline batches'
expected Brier of a median of the same number of their saved predictions,
paired by question.
"""

from __future__ import annotations

import argparse
import hashlib
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
        first = launch.split("\n\n")[0]
        if any(ln.startswith("reverse: ") and ln != "reverse: off" for ln in first.splitlines()):
            sys.exit(f"{batch} holds P(No) forecasts; combine it first (backtest.round4 combine)")
        for line in (RUNS / batch / "predictions.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut short by a hard kill
                continue
            if row.get("prediction") is None:
                continue
            if seen_in.setdefault(row["id"], batch) != batch:
                sys.exit(f"question {row['id']} appears in {seen_in[row['id']]} and {batch}")
            preds.setdefault(row["id"], []).append(float(row["prediction"]))
    return meta, [by_id[i] for i in preds], preds


def clusters(questions: list[dict], join_days: int = 0) -> dict[str, str]:
    """Each question's resampling cluster: questions that share a Polymarket
    event group or a resolution date are joined (elections, Fed meetings and
    month-end questions resolve together). Coarse on purpose, so intervals err
    wide rather than narrow. Items that carry their own `cluster` (round-5 dataset
    items: source, freeze date, resolution date) use it as given."""
    if questions and all(q.get("cluster") for q in questions):
        return {q["id"]: "c:" + q["cluster"] for q in questions}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for q in questions:
        parent[find("q:" + q["id"])] = find("g:" + q["group"])
        parent[find("g:" + q["group"])] = find("d:" + q.get("resolution_date", q["id"]))
    if join_days:  # resolution dates within join_days of each other are one cluster
        from datetime import date

        by_day: dict[str, str] = {}  # day -> one full resolution_date string on it
        for q in questions:
            if q.get("resolution_date"):
                day = q["resolution_date"][:10]
                if day in by_day:
                    parent[find("d:" + q["resolution_date"])] = find("d:" + by_day[day])
                else:
                    by_day[day] = q["resolution_date"]
        days = sorted(by_day)
        for a, b in zip(days, days[1:]):
            if (date.fromisoformat(b) - date.fromisoformat(a)).days <= join_days:
                parent[find("d:" + by_day[a])] = find("d:" + by_day[b])
    return {q["id"]: find("q:" + q["id"]) for q in questions}


def folds_by_group(questions: list[dict]) -> dict[str, int]:
    """Fold per cluster id; questions carry their cluster in "cluster"."""
    groups = sorted({q["cluster"] for q in questions})
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
    lo, hi = stats[int(0.025 * BOOTSTRAP) - 1], stats[int(0.975 * BOOTSTRAP) - 1]
    below = sum(s <= 0 for s in stats) / BOOTSTRAP
    above = sum(s >= 0 for s in stats) / BOOTSTRAP
    return mean, lo, hi, min(1.0, 2 * min(below, above))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by continued fraction (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    f = d
    for m in range(1, 300):
        for step in (0, 1):
            if step == 0:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + num / c if abs(c) > 1e-300 else 1e-300
            f *= c * d
        if abs(c * d - 1) < 1e-12:
            break
    return front * f / a


def t_two_sided_p(t: float, df: int) -> float:
    return _betainc(df / 2, 0.5, df / (df + t * t))


def t_quantile(q: float, df: int) -> float:
    """Upper quantile of Student's t by bisection (q in (0.5, 1))."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - t_two_sided_p(mid, df) / 2 < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cluster_t(diffs: dict[str, float], groups: dict[str, str]) -> tuple[float, float, float, float]:
    """Mean of cluster means, its 95% interval and two-sided p from a t-test on cluster means
    (df = clusters - 1): PLAN-refine-round5.md, where 22 clusters make a percentile bootstrap undercover."""
    by: dict[str, list[float]] = {}
    for qid, d in diffs.items():
        by.setdefault(groups[qid], []).append(d)
    means = [statistics.fmean(v) for v in by.values()]
    n = len(means)
    if n < 2:
        raise ValueError("cluster_t needs at least two clusters")
    mean, se = statistics.fmean(means), statistics.stdev(means) / math.sqrt(n)
    if se == 0:
        return mean, mean, mean, (1.0 if mean == 0 else 0.0)
    half = t_quantile(0.975, n - 1) * se
    return mean, mean - half, mean + half, t_two_sided_p(mean / se, n - 1)


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


def subsample_log(values: list[float], k: int, y: int) -> float:
    """Expected log score of the median of k predictions drawn from the saved ones."""
    combos = list(itertools.combinations(values, k))
    return statistics.fmean(log_score(statistics.median(c), y) for c in combos)


def ensemble(batch_preds: list[dict[str, list[float]]], k: int) -> dict[str, list[float]]:
    """Per question, each batch's median of its first k predictions; questions every batch has k for."""
    ids = set.intersection(*(set(i for i, v in p.items() if len(v) >= k) for p in batch_preds))
    return {i: [statistics.median(p[i][:k]) for p in batch_preds] for i in ids}


def analyse(questions: list[dict], preds: dict[str, list[float]], predictions: int) -> dict:
    full = [dict(q) for q in questions if len(preds[q["id"]]) >= predictions]
    short = sorted(q["id"] for q in questions if len(preds[q["id"]]) < predictions)
    cluster_of = clusters(full)
    for q in full:
        q["cluster"] = cluster_of[q["id"]]
    groups = {q["id"]: q["cluster"] for q in full}
    y = {q["id"]: q["outcome"] for q in full}
    current = {q["id"]: statistics.median(preds[q["id"]][:predictions]) for q in full}
    base_brier = {i: brier(current[i], y[i]) for i in current}

    fold_of = folds_by_group(full)
    fold = {q["id"]: fold_of[q["cluster"]] for q in full}
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
        # Adopted only if the Holm-adjusted p is under 0.05 and the interval excludes 0.
        comparisons[name]["passes"] = (
            p < 0.05 and comparisons[name]["gain"] > 0 and comparisons[name]["ci95"][0] > 0
        )

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
        "questions_with_fewer_predictions": short,
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


def compare(
    questions: list[dict], base: dict[str, list[float]], other: dict[str, list[float]], k: int,
    research: dict[str, dict] | None = None, base_k: int = 5, min_gain: float | None = None,
    join_days: int = 0, test: str = "bootstrap",
) -> dict:
    """Paired gain of `other` (median of k) over `base` (expected median of k of
    its first base_k saved predictions). Positive gain: `other` scores better.

    Pass rule: by default the research plans' (gain at least 0.01 and the 95%
    interval above 0). With min_gain, the live-Wikipedia plan's harm check:
    gain at least min_gain, no interval condition."""
    ids = [q["id"] for q in questions
           if len(other.get(q["id"], [])) >= k and len(base.get(q["id"], [])) >= base_k]
    full = [dict(q) for q in questions if q["id"] in ids]
    cluster_of = clusters(full, join_days)
    y = {q["id"]: q["outcome"] for q in full}
    crowd = {q["id"]: q["crowd"] for q in full}
    med = {i: statistics.median(other[i][:k]) for i in ids}
    base_b = {i: subsample_brier(base[i][:base_k], k, y[i]) for i in ids}
    other_b = {i: brier(med[i], y[i]) for i in ids}
    diffs = {i: base_b[i] - other_b[i] for i in ids}
    stat = cluster_t if test == "cluster-t" else group_bootstrap
    mean, lo, hi, p = stat(diffs, cluster_of)
    cluster_mean = None
    if test == "cluster-t":  # the gain stays the mean over questions; interval and p from cluster means
        cluster_mean, mean = mean, statistics.fmean(diffs.values())
    # The tournament's score is log-based: lower log loss is better, so gain = baseline - compared.
    log_diffs = {i: subsample_log(base[i][:base_k], k, y[i]) - log_score(med[i], y[i]) for i in ids}
    lmean, llo, lhi, lp = stat(log_diffs, cluster_of)
    sizes: dict[str, int] = {}
    for i in ids:
        sizes[cluster_of[i]] = sizes.get(cluster_of[i], 0) + 1
    biggest = max(sizes, key=sizes.get) if sizes else None
    result = {
        "questions": len(ids),
        "left_out_too_few_predictions": sorted(q["id"] for q in questions if q["id"] not in ids),
        "clusters": len(sizes),
        "largest_cluster": sizes.get(biggest),
        "brier": {
            "baseline_median_of_k": statistics.fmean(base_b.values()),
            "compared_median": statistics.fmean(other_b.values()),
            "crowd": (statistics.fmean(brier(crowd[i], y[i]) for i in ids)
                      if all(crowd[i] is not None for i in ids) else None),
        },
        "gain": {"mean": mean, "ci95": [lo, hi], "p": p},
        "test": ("t-test on cluster means (df = clusters - 1); gain is the mean over questions, interval and "
                 "p are for the mean of cluster means" if test == "cluster-t" else "cluster bootstrap"),
        "mean_of_cluster_means": cluster_mean,
        "log_gain": {"mean": lmean, "ci95": [llo, lhi], "p": lp},
        "rule": (f"gain >= {min_gain}" if min_gain is not None
                 else "gain >= 0.01 and 95% interval above 0"),
        "passes": (mean >= min_gain) if min_gain is not None else (mean >= 0.01 and lo > 0),
        "mean_forecast": statistics.fmean(med.values()),
        "yes_rate": statistics.fmean(y.values()),
        "mean_abs_from_crowd": None if any(crowd[i] is None for i in ids) else {
            f"baseline_median_of_{base_k}": statistics.fmean(
                abs(statistics.median(base[i][:base_k]) - crowd[i]) for i in ids),
            "compared": statistics.fmean(abs(med[i] - crowd[i]) for i in ids),
        },
    }
    result["by_outcome"] = {
        name: {"n": len(sel),
               "baseline_brier": statistics.fmean(base_b[i] for i in sel) if sel else None,
               "compared_brier": statistics.fmean(other_b[i] for i in sel) if sel else None}
        for name, sel in (("resolved_yes", [i for i in ids if y[i]]), ("resolved_no", [i for i in ids if not y[i]]))
    }
    calibration = []
    for b in range(10):
        lo_b, hi_b = b / 10, (b + 1) / 10
        sel = [i for i in ids if lo_b <= med[i] < hi_b or (b == 9 and med[i] == 1)]
        if sel:
            calibration.append({"bin": f"{lo_b:.1f}-{hi_b:.1f}", "n": len(sel),
                                "mean_forecast": statistics.fmean(med[i] for i in sel),
                                "yes_rate": statistics.fmean(y[i] for i in sel)})
    result["calibration"] = calibration
    # Exploratory: without the largest cluster, and split by research flags.
    if biggest is not None and len(sizes) > 1:
        rest = {i: d for i, d in diffs.items() if cluster_of[i] != biggest}
        m, l, h, pp = group_bootstrap(rest, cluster_of)
        result["exploratory_without_largest_cluster"] = {"n": len(rest), "mean": m, "ci95": [l, h]}
    if research:
        def split(flag) -> dict:
            out = {}
            for label, sel in (("flagged", [i for i in ids if flag(research.get(i, {}))]),
                               ("not_flagged", [i for i in ids if not flag(research.get(i, {}))])):
                out[label] = {"n": len(sel), "mean_gain": statistics.fmean(diffs[i] for i in sel) if sel else None}
            return out

        # AskNews rows carry leak.score and market_words; Wikipedia rows carry
        # leak.later_dates and leak.markets.
        result["exploratory_by_leak_flag"] = split(
            lambda r: r.get("leak", {}).get("score", 0) > 0 or bool(r.get("leak", {}).get("later_dates")))
        result["exploratory_by_market_words"] = split(
            lambda r: bool(r.get("market_words")) or bool(r.get("leak", {}).get("markets")))
        result["no_articles"] = sum("No articles were found." in research.get(i, {}).get("text", "") for i in ids)
        result["articles_dropped_after_freeze"] = sum(
            c["dropped_after_freeze"] for i in ids for c in research.get(i, {}).get("calls", [])
        )
    return result


CAL_ALPHAS = [round(0.2 + 0.05 * i, 2) for i in range(57)]  # 0.2 to 3.0
CAL_BETAS = [round(-1.0 + 0.05 * i, 2) for i in range(41)]  # -1.0 to 1.0
CAL_SEEDS = 20
CAL_PLAN = HERE.parent / "benchmarks" / "PLAN-refine-round2.md"


def _fit_calibration(lp, y, with_beta: bool) -> tuple[float, float]:
    """Grid fit of sigmoid(alpha * logit(p) + beta) by mean Brier (numpy arrays in)."""
    import numpy as np

    alphas = np.array(CAL_ALPHAS)[:, None, None]
    betas = (np.array(CAL_BETAS) if with_beta else np.zeros(1))[None, :, None]
    p = 1 / (1 + np.exp(-(alphas * lp[None, None, :] + betas)))
    p = np.clip(p, *CLIP)
    loss = ((p - y[None, None, :]) ** 2).mean(axis=2)
    a, b = np.unravel_index(np.argmin(loss), loss.shape)
    return float(CAL_ALPHAS[a]), (float(CAL_BETAS[b]) if with_beta else 0.0)


def _apply_calibration(p: float, alpha: float, beta: float) -> float:
    return min(max(1 / (1 + math.exp(-(alpha * logit(p) + beta))), CLIP[0]), CLIP[1])


def calibrate(questions: list[dict], preds: dict[str, list[float]], k: int) -> dict:
    """PLAN-refine-round2.md, section 2: variants A (alpha) and B (alpha, beta)
    against no adjustment, out of fold by cluster, repeated over CAL_SEEDS fold
    seeds; Holm over the two; then the adoption rule."""
    import numpy as np

    full = [dict(q) for q in questions if len(preds.get(q["id"], [])) >= k]
    cluster_of = clusters(full)
    ids = [q["id"] for q in full]
    y = {q["id"]: q["outcome"] for q in full}
    med = {i: min(max(statistics.median(preds[i][:k]), CLIP[0]), CLIP[1]) for i in ids}
    base = {i: brier(med[i], y[i]) for i in ids}
    groups = sorted(set(cluster_of.values()))
    variants = {"A_alpha": False, "B_alpha_beta": True}
    per_seed = {v: [] for v in variants}  # mean out-of-fold gain per seed
    first_seed_diffs = {}
    for seed in range(CAL_SEEDS):
        order = list(groups)
        random.Random(SEED + seed).shuffle(order)
        fold_of = {g: n % FOLDS for n, g in enumerate(order)}
        for name, with_beta in variants.items():
            diffs = {}
            for f in range(FOLDS):
                train = [i for i in ids if fold_of[cluster_of[i]] != f]
                test = [i for i in ids if fold_of[cluster_of[i]] == f]
                if not train or not test:
                    continue
                a, b = _fit_calibration(np.array([logit(med[i]) for i in train]),
                                        np.array([y[i] for i in train], dtype=float), with_beta)
                for i in test:
                    diffs[i] = base[i] - brier(_apply_calibration(med[i], a, b), y[i])
            per_seed[name].append(statistics.fmean(diffs.values()))
            if seed == 0:
                first_seed_diffs[name] = diffs
    result = {"plan": CAL_PLAN.name,
              "plan_sha256": hashlib.sha256(CAL_PLAN.read_bytes()).hexdigest() if CAL_PLAN.exists() else "missing",
              "questions": len(ids), "clusters": len(groups), "predictions": k,
              "brier_uncalibrated": statistics.fmean(base.values()),
              "mean_forecast": statistics.fmean(med.values()), "yes_rate": statistics.fmean(y.values()),
              "variants": {}}
    pvalues = {}
    for name in variants:
        m, lo, hi, p = group_bootstrap(first_seed_diffs[name], cluster_of)
        seeds = per_seed[name]
        result["variants"][name] = {
            "gain_mean_over_seeds": statistics.fmean(seeds),
            "gain_seed_spread": [min(seeds), max(seeds)],
            "gain_first_seed": m, "ci95_first_seed": [lo, hi], "p_first_seed": p,
        }
        pvalues[name] = p
    for name, p in holm(pvalues).items():
        result["variants"][name]["p_holm"] = p
    a, b = result["variants"]["A_alpha"], result["variants"]["B_alpha_beta"]
    a_pass = a["gain_mean_over_seeds"] >= 0.005 and a["p_holm"] < 0.05
    b_pass = b["gain_mean_over_seeds"] >= 0.005 and b["p_holm"] < 0.05
    b_better = b["gain_mean_over_seeds"] - a["gain_mean_over_seeds"] >= 0.003
    adopted = "B_alpha_beta" if (b_pass and b_better) else ("A_alpha" if a_pass else None)
    lp = np.array([logit(med[i]) for i in ids])
    yy = np.array([y[i] for i in ids], dtype=float)
    fits = {name: _fit_calibration(lp, yy, wb) for name, wb in variants.items()}
    result["rule"] = ("A if gain >= 0.005 and Holm p < 0.05; B instead if B passes the same and beats A by >= 0.003; "
                      "adoption waits for the held-out harm check")
    result["passes"] = {"A_alpha": a_pass, "B_alpha_beta": b_pass, "B_beats_A_by_0.003": b_better}
    result["adopted_pending_heldout"] = adopted
    result["fitted_on_all"] = {name: {"alpha": f[0], "beta": f[1]} for name, f in fits.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--predictions", type=int, default=5)
    parser.add_argument("--dataset", help="question list file (default backtest/data/questions.json)")
    parser.add_argument("--compare", help="batch to compare against the baseline batches")
    parser.add_argument("--ensemble", help="compare: comma-separated batches; the compared forecast is the "
                                           "median of their medians (each of its first --predictions)")
    parser.add_argument("--base-k", type=int, default=5, help="baseline predictions per question to use")
    parser.add_argument("--join-dates-within", type=int, default=0,
                        help="compare: resolution dates within this many days form one cluster")
    parser.add_argument("--calibrate", action="store_true",
                        help="the calibration test of PLAN-refine-round2.md on the given batch")
    parser.add_argument("--test", choices=["bootstrap", "cluster-t"], default="bootstrap",
                        help="compare: interval and p from the cluster bootstrap or a t-test on cluster means")
    parser.add_argument("--min-gain", type=float, default=None,
                        help="pass rule: gain at least this, no interval condition (default: the research rule)")
    args = parser.parse_args()
    if args.dataset:  # a held-out list instead of questions.json
        global QUESTIONS
        QUESTIONS = Path(args.dataset).resolve()
    meta, questions, preds = load(args.batches)
    if args.calibrate:
        result = calibrate(questions, preds, args.predictions)
        result["batches"] = args.batches
        OUT.mkdir(exist_ok=True)
        path = OUT / f"calibrate-{'+'.join(args.batches)}.json"
        path.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        print(f"wrote {path}")
        return
    if args.ensemble:
        if args.compare:
            parser.error("use --compare or --ensemble, not both")
        names = args.ensemble.split(",")
        if args.base_k < len(names):
            parser.error("--base-k must be at least the number of --ensemble batches")
        batch_preds = [load([b])[2] for b in names]
        other = ensemble(batch_preds, args.predictions)
        if not other:
            sys.exit(f"No question has {args.predictions} predictions in every --ensemble batch (check --predictions).")
        dropped = len(set().union(*batch_preds)) - len(other)
        result = compare(questions, preds, other, len(names), None, args.base_k, args.min_gain,
                         args.join_dates_within, args.test)
        if not result["questions"]:
            sys.exit("No question has enough baseline predictions (check --base-k).")
        result["join_dates_within"] = args.join_dates_within
        result["ensemble_dropped_too_few_predictions"] = dropped
        result["baseline"], result["compared"] = args.batches, f"median of {names}"
        OUT.mkdir(exist_ok=True)
        suffix = f"-join{args.join_dates_within}" if args.join_dates_within else ""
        path = OUT / f"compare-{'+'.join(args.batches)}-vs-ensemble-{'+'.join(names)}{suffix}.json"
        path.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        print(f"wrote {path}")
        return
    if args.compare:
        _, _, other = load([args.compare])
        rpath = RUNS / args.compare / "research.jsonl"
        research = {}
        if rpath.exists():
            for line in rpath.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # a line cut short by a hard kill
                    continue
                research[row["id"]] = row
        result = compare(questions, preds, other, args.predictions, research, args.base_k, args.min_gain,
                         args.join_dates_within, args.test)
        result["join_dates_within"] = args.join_dates_within
        result["baseline"], result["compared"] = args.batches, args.compare
        OUT.mkdir(exist_ok=True)
        suffix = (f"-join{args.join_dates_within}" if args.join_dates_within else "") + (
            "-clustert" if args.test == "cluster-t" else "")
        path = OUT / f"compare-{'+'.join(args.batches)}-vs-{args.compare}{suffix}.json"
        path.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        print(f"wrote {path}")
        return
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
