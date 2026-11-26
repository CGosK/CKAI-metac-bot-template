"""
Round 7: self-made numeric questions (benchmarks/PLAN-refine-round7-numeric.md).

    uv run python -m backtest.round7 build      # public data requests (cache mostly warm), no model calls
    uv run python -m backtest.round7 resolve    # on or after 2026-12-05
    uv run python -m backtest.round7 score --off r7-a0 --ours r7-a2 [--mc r7-mc]   # no model calls or requests

build: round 6's dates, horizons, series pool and the exclusions that don't depend on a threshold. Per
(T, horizon): 12 numeric questions (3 per width multiplier k) and 2 backups by a greedy hash walk (one series per
question, family cap 3), and one multiple-choice smoke-test question. Range: midpoint m = anchor + mean h-day change
+ v x s, v uniform on [-1.5, 1.5] by hash; bounds m +- k x s, both open, shown in fixed decimals. Writes
data/round7-numeric.json (never overwritten).

resolve: round6.observe for every item (same vintages, staleness rule and stored rows); writes
data/round7-outcomes.json (never overwritten).

score: model calibration from declared percentiles (A2 decides; A0 alongside), the library line from final
CDFs, k-balanced Metaculus baseline scores, the leave-one-T-out widening rule, reference forecasters, the MC smoke
test. Writes out/round7-score-<off>-vs-<ours>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import data_research as dr
from backtest import fetch, round5, round6, score
from backtest import scoring_numeric as sn

DATA = round5.DATA
ITEMS = DATA / "round7-numeric.json"
OUTCOMES = DATA / "round7-outcomes.json"
KS = (3, 6, 12, 24)
PER_K, NUMERIC_BACKUPS = 3, 2
W_GRID = [round(1.0 + 0.1 * i, 1) for i in range(31)]  # 1.0 .. 4.0 (v3.1: 3.0 capped a very narrow model)
MIN_SCORED_PER_CELL = 10  # v3.1: a cell is scored with at least 10 numeric questions (k-balanced means)


# ---------------------------------------------------------------- pure rules


def unit(text: str) -> float:
    """A uniform draw on [0, 1) from a hash."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:13], 16) / 16 ** 13


def stats(rows: list, h: int) -> dict | None:
    ch = round6.changes(rows, h)
    if len(ch) < 20:
        return None
    return {"anchor": rows[-1][1], "anchor_date": rows[-1][0].isoformat(), "mean_change": statistics.fmean(ch),
            "spread": statistics.pstdev(ch), "zero_share": sum(c == 0 for c in ch) / len(ch),
            "places": round6.places(rows)}


def exclusion(source: str, series_id: str, name: str, freq: str | None, st: dict | None, t_date: date) -> str | None:
    """Round 6's exclusions that don't depend on a threshold (plan, Items)."""
    if source == "dbnomics" and dr.dbnomics_id(series_id)[:2] not in round6.DBNOMICS_DATASETS:
        return "dbnomics dataset outside the rarely-revised daily set"
    if round6.POLICY_RATE.search(name):
        return "policy-administered rate"
    if freq is None:
        return "not daily or weekly"
    if st is None:
        return "under 20 past changes"
    if date.fromisoformat(st["anchor_date"]) < t_date - timedelta(days=round6.MAX_ANCHOR_AGE_DAYS):
        return "published with a lag over 21 days"
    if st["zero_share"] > round6.STICKY_ZERO_SHARE:
        return "sticky"
    if st["spread"] == 0:
        return "no spread"
    return None


def numeric_range(st: dict, k: int, salt: str) -> dict:
    v = -1.5 + 3 * unit("v|" + salt)
    p = st["places"]
    mid = st["anchor"] + st["mean_change"] + v * st["spread"]
    lower, upper = round(mid - k * st["spread"], p), round(mid + k * st["spread"], p)
    return {"v": v, "k": k, "midpoint": mid, "lower_bound": lower, "upper_bound": upper,
            "lower_shown": round6.show(lower, p), "upper_shown": round6.show(upper, p)}


def mc_options(st: dict, salt: str) -> dict:
    v = -1.5 + 3 * unit("mc|" + salt)
    p = st["places"]
    m = st["anchor"] + st["mean_change"] + v * st["spread"]
    cuts = [round(m + d * st["spread"], p) for d in (-1, 0, 1)]
    x, y, z = (round6.show(c, p) for c in cuts)
    return {"v": v, "cuts": cuts, "options": [f"below {x}", f"{x} to {y}", f"{y} to {z}", f"{z} or above"]}


def mc_index(cuts: list[float], value: float) -> int:
    """Bins: below x; x to y; y to z; z or above (a value on a cut goes to the upper bin)."""
    return sum(value >= c for c in cuts)


def walk(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Greedy over ranked candidates (key, family, k, ok): 12 primaries with 3 per k, then 2 backups; one series
    per question, family cap 3."""
    prim, back, used, fam, per_k = [], [], set(), Counter(), Counter()
    for c in candidates:
        if not c["ok"] or c["key"] in used or fam[c["family"]] >= round6.FAMILY_CAP:
            continue
        if len(prim) < PER_K * len(KS):
            if per_k[c["k"]] >= PER_K:
                continue
            prim.append(c)
            per_k[c["k"]] += 1
        elif len(back) < NUMERIC_BACKUPS:
            back.append(c)
        else:
            break
        used.add(c["key"])
        fam[c["family"]] += 1
    return prim, back


# ---------------------------------------------------------------- build


def build(http) -> dict:
    meta = round5.load_items(round5.DEFAULT_ITEMS)
    items = meta["questions"]
    keys = sorted({(q["source"], q["series_id"]) for q in items})
    backgrounds = round6.background_for(items)
    fetched, excluded, questions = {}, defaultdict(Counter), []

    def series(source, sid, t_date):
        k = (source, sid, t_date)
        if k not in fetched:
            freeze = datetime(t_date.year, t_date.month, t_date.day, tzinfo=timezone.utc)
            try:
                fetched[k] = dr.series_for(source, sid, freeze, http)
            except dr.DataTransient:
                raise
            except dr.DataUnavailable as exc:
                fetched[k] = exc
        return fetched[k]

    def candidate(source, sid, t_date, h, k):
        key = f"{source}:{sid}"
        if source == "dbnomics" and dr.dbnomics_id(sid)[:2] not in round6.DBNOMICS_DATASETS:
            excluded[key]["dbnomics dataset outside the rarely-revised daily set"] += 1
            return None
        s = series(source, sid, t_date)
        if isinstance(s, Exception):
            excluded[key][f"unavailable: {type(s).__name__}"] += 1
            return None
        st = stats(s.rows, h)
        freq = round6.frequency(s.rows)
        why = exclusion(source, sid, s.name, freq, st, t_date)
        if why:
            excluded[key][why] += 1
        units = s.units or ("degrees Celsius" if s.name.startswith("Degree Celsius") else "value")
        return {"key": key, "source": source, "series_id": sid, "name": s.name, "units": units,
                "family": round6.family(source, sid, s.name), "k": k, "ok": why is None, "freq": freq, "st": st}

    for t_date in round6.T_DATES:
        for h in round6.HORIZONS:
            cands = sorted(((s, sid, k) for s, sid in keys for k in KS),
                           key=lambda c: hashlib.sha256(f"round7:{t_date}{h}{c[0]}:{c[1]}{c[2]}".encode()).hexdigest())
            seen = []
            for source, sid, k in cands:
                c = candidate(source, sid, t_date, h, k)
                if c:
                    seen.append(c)
                prim, back = walk(seen)
                if len(prim) == PER_K * len(KS) and len(back) == NUMERIC_BACKUPS:
                    break
            prim, back = walk(seen)
            for role, chosen in (("primary", prim), ("backup", back)):
                for rank, c in enumerate(chosen):
                    questions.append(numeric_question(c, t_date, h, role, rank, backgrounds.get(c["key"], "")))
            if len(prim) != PER_K * len(KS) or len(back) != NUMERIC_BACKUPS:
                raise SystemExit(f"build check failed: T {t_date} h {h} has {len(prim)} numeric and {len(back)} "
                                 "backups; nothing written")
            used = {c["key"] for c in prim + back}
            mc_cands = sorted(keys, key=lambda c: hashlib.sha256(f"round7mc:{t_date}{h}{c[0]}:{c[1]}".encode()).hexdigest())
            mc = None
            for source, sid in mc_cands:
                if f"{source}:{sid}" in used:
                    continue
                c = candidate(source, sid, t_date, h, 0)
                if c and c["ok"]:
                    mc = mc_question(c, t_date, h, backgrounds.get(c["key"], ""))
                    break
            if mc is None:
                raise SystemExit(f"build check failed: T {t_date} h {h} has no MC question; nothing written")
            questions.append(mc)
            print(f"T {t_date} h {h}: {len(prim)} numeric, {len(back)} backups, MC 1 "
                  f"(requests {http.requests}, cache hits {http.cache_hits})", flush=True)
    return {"round": 7, "plan": "PLAN-refine-round7-numeric.md", "self_made": True,
            "dataset_commit": meta["dataset_commit"], "freeze_from": "2026-08-01",
            "source_items_sha256": meta["sha256"],
            "data_research_sha256": hashlib.sha256(Path(dr.__file__).read_bytes()).hexdigest(),
            "excluded": {k: dict(v) for k, v in sorted(excluded.items())},
            "sha256": fetch.list_hash(questions), "questions": questions}


def _common(c: dict, t_date: date, h: int, background: str, kind: str, role: str, rank: int, salt: str) -> dict:
    due = t_date + timedelta(days=h)
    st = c["st"]
    return {"id": hashlib.sha256(f"r7|{salt}".encode()).hexdigest()[:16], "question_type": kind,
            "source": c["source"], "series_id": c["series_id"], "group": c["key"], "cluster": t_date.isoformat(),
            "family": c["family"], "frequency": c["freq"], "role": role, "rank": rank, "horizon_days": h,
            "units": c["units"], "spread": st["spread"], "mean_change": st["mean_change"],
            "anchor_date": st["anchor_date"], "background": background,
            "freeze_datetime": f"{t_date.isoformat()}T00:00:00+00:00", "resolution_date": due.isoformat(),
            "freeze_value": st["anchor"], "freeze_value_explanation": f"The latest value of {c['name']}",
            "crowd": None, "outcome": None}


def criteria(c: dict, due: date) -> str:
    published = (f"as published {round6.VINTAGE_DAYS[c['freq']]} days after that date" if c["source"] == "fred"
                 else "as published when the question is resolved")
    return (f"Resolves to the value of {c['name']} for {due.isoformat()} (the latest observation dated on or before "
            f"{due.isoformat()}, {published}). Self-made test question.")


def numeric_question(c: dict, t_date: date, h: int, role: str, rank: int, background: str) -> dict:
    salt = f"{t_date}|{h}|{c['key']}|{c['k']}"
    q = _common(c, t_date, h, background, "numeric", role, rank, salt)
    r = numeric_range(c["st"], c["k"], salt)
    due = t_date + timedelta(days=h)
    units = f" ({c['units']})" if c["units"] else ""
    q.update(r, question=f"What will {c['name']}{units} be on {due.isoformat()}?", criteria=criteria(c, due))
    return q


def mc_question(c: dict, t_date: date, h: int, background: str) -> dict:
    salt = f"mc|{t_date}|{h}|{c['key']}"
    q = _common(c, t_date, h, background, "multiple_choice", "smoke", 0, salt)
    r = mc_options(c["st"], salt)
    due = t_date + timedelta(days=h)
    units = f" ({c['units']})" if c["units"] else ""
    q.update(r, question=f"Which range will {c['name']}{units} be in on {due.isoformat()}?",
             criteria=criteria(c, due).replace("Resolves to the value", "Resolves to the option holding the value"))
    return q


# ---------------------------------------------------------------- resolve


def resolve(http) -> dict:
    if date.today() < round6.RESOLVE_FROM:
        sys.exit(f"resolve runs on or after {round6.RESOLVE_FROM} (plan: the last weekly vintage exists then)")
    meta = json.loads(ITEMS.read_text())
    fresh, cache, out, transient = dr.PoliteHttp(), {}, [], []
    for q in meta["questions"]:
        try:
            r = round6.observe(q, http, fresh, cache)
        except dr.DataTransient as exc:
            transient.append(f"{q['id']} {dr.redact(str(exc))[:120]}")
            continue
        value = r.get("obs_value")
        if q["question_type"] == "multiple_choice" and value is not None:
            value = mc_index(q["cuts"], value)
        out.append({**q, "outcome": value, "resolution": r})
    if transient:
        sys.exit(f"{len(transient)} transient fetch errors; nothing written, rerun resolve. First: {transient[0]}")
    return {**{k: v for k, v in meta.items() if k != "questions"},
            "items_file_sha256": hashlib.sha256(ITEMS.read_bytes()).hexdigest(),
            "resolved_on": date.today().isoformat(), "questions": out}


# ---------------------------------------------------------------- score


def rows(batch: str) -> dict[str, dict]:
    out = {}
    for line in (score.RUNS / batch / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("prediction") is not None and r.get("k") == 0:
            out[r["id"]] = r
    return out


def scored(questions: list[dict], have: set[str]) -> tuple[list[dict], dict]:
    """Per (T, horizon): numeric primaries resolved and forecast in both arms; a missing one is replaced by a
    backup (scarcest k first); a cell left with fewer than MIN_SCORED_PER_CELL is not scored."""
    by = defaultdict(list)
    for q in questions:
        if q["question_type"] == "numeric":
            by[(q["cluster"], q["horizon_days"])].append(q)
    out, notes = [], {}
    for cell, qs in sorted(by.items()):
        ok = [q for q in qs if q["outcome"] is not None and q["id"] in have]
        prim = [q for q in sorted(qs, key=lambda q: q["rank"]) if q["role"] == "primary"]
        keep = [q for q in prim if q in ok]
        spare = [q for q in sorted(qs, key=lambda q: q["rank"]) if q["role"] == "backup" and q in ok]
        gone = [q for q in prim if q not in ok]
        # Fill the missing k values with the fewest same-k backups first, so a same-k backup isn't spent early.
        gone.sort(key=lambda g: (sum(b["k"] == g["k"] for b in spare), g["rank"]))
        for g in gone:
            same = [b for b in spare if b["k"] == g["k"]]
            pick = (same or spare)[:1]
            if pick:
                keep.append(pick[0])
                spare.remove(pick[0])
        notes[f"{cell[0]} h{cell[1]}"] = {"missing_primaries": len(gone), "scored_questions": len(keep),
                                          "k": dict(Counter(q["k"] for q in keep)),
                                          "scored": len(keep) >= MIN_SCORED_PER_CELL}
        if len(keep) >= MIN_SCORED_PER_CELL:
            out += keep
    return out, notes


def cluster_stat(values: dict[str, float], qs: list[dict], k_balanced: bool = False) -> dict:
    """Mean of cluster means (k-balanced: per cluster, the mean of the per-k means), 95% interval and p from the
    t-test on cluster means (score.cluster_t), and the median of cluster means."""
    by = defaultdict(lambda: defaultdict(list))
    for q in qs:
        if q["id"] in values:
            by[q["cluster"]][q["k"] if k_balanced else 0].append(values[q["id"]])
    means = {c: statistics.fmean(statistics.fmean(v) for v in ks.values()) for c, ks in by.items()}
    if len(means) < 2:
        return {"clusters": len(means), "note": "fewer than 2 clusters"}
    mean, lo, hi, p = score.cluster_t(means, {c: c for c in means})
    return {"clusters": len(means), "mean": mean, "ci95": [lo, hi], "p": p,
            "median_of_cluster_means": statistics.median(means.values())}


def calibration(qs: list[dict], preds: dict[str, dict]) -> dict:
    cov = {q["id"]: sn.covered(preds[q["id"]]["percentiles"], q["outcome"]) for q in qs}
    below = {q["id"]: sn.below_median(preds[q["id"]]["percentiles"], q["outcome"]) for q in qs}
    c, b = cluster_stat(cov, qs), cluster_stat(below, qs)
    # p against the target, not against 0
    c["p_vs_0_80"] = cluster_stat({i: v - 0.80 for i, v in cov.items()}, qs)["p"]
    b["p_vs_0_50"] = cluster_stat({i: v - 0.50 for i, v in below.items()}, qs)["p"]
    c.pop("p"), b.pop("p")
    flags = []
    if c["ci95"][1] < 0.80:
        flags.append("too narrow")
    if c["ci95"][0] > 0.80:
        flags.append("too wide")
    if not b["ci95"][0] <= 0.50 <= b["ci95"][1]:
        flags.append("off-centre")
    return {"coverage_p10_p90": c, "below_p50": b, "flags": flags or ["calibrated"]}


def score_of(q: dict, cdf: list[float]) -> float:
    """Baseline score of a submitted CDF (after the live aggregation pass)."""
    return sn.numeric_baseline(cdf, q["outcome"], q["lower_bound"], q["upper_bound"])


def widening(qs: list[dict], preds: dict[str, dict]) -> dict:
    """Widening applied to each forecast's declared percentiles before the library and the aggregation pass (as
    it would go into the live bot). w values the live parser would reject for any question (strict validation,
    which doesn't depend on outcomes) are left out of the grid."""
    allowed = [w for w in W_GRID if all(sn.strict_ok(sn.widen(preds[q["id"]]["percentiles"], w), q["lower_bound"],
                                                     q["upper_bound"]) for q in qs)]
    if 1.0 not in allowed:
        raise SystemExit("widening: w = 1.0 fails strict validation for some question; stored forecasts should not")
    grid = {q["id"]: {w: score_of(q, sn.submitted(sn.widen(preds[q["id"]]["percentiles"], w),
                                                  q["lower_bound"], q["upper_bound"])) for w in allowed}
            for q in qs}

    def best_w(sub):
        return max(allowed, key=lambda w: cluster_stat({q["id"]: grid[q["id"]][w] for q in sub}, sub, True)["mean"]
                   if len({q["cluster"] for q in sub}) > 1 else statistics.fmean(grid[q["id"]][w] for q in sub))

    folds, held_gain = {}, {}
    for t in sorted({q["cluster"] for q in qs}):
        train = [q for q in qs if q["cluster"] != t]
        test = [q for q in qs if q["cluster"] == t]
        w = best_w(train)
        per_k = defaultdict(list)
        for q in test:
            g = grid[q["id"]][w] - grid[q["id"]][1.0]
            held_gain[q["id"]] = g
            per_k[q["k"]].append(g)
        folds[t] = {"w": w, "gain": statistics.fmean(statistics.fmean(v) for v in per_k.values())}
    mean, lo, hi, p = score.cluster_t({t: f["gain"] for t, f in folds.items()}, {t: t for t in folds})
    median_q = statistics.median(held_gain.values())
    cov_w = min(allowed, key=lambda w: abs(statistics.fmean(
        sn.covered(sn.widen(preds[q["id"]]["percentiles"], w), q["outcome"]) for q in qs) - 0.80))
    adopt = mean > 0 and lo > 0 and median_q >= 0
    return {"folds": folds, "held_out_gain": {"mean": mean, "ci95": [lo, hi], "p": p},
            "median_per_question_gain": median_q, "w_all_dates": (w_all := best_w(qs)),
            "w_for_declared_coverage_0_80": cov_w, "grid_allowed": [allowed[0], allowed[-1]],
            "w_at_grid_top": w_all == allowed[-1],
            "per_k_w_exploratory": {k: best_w([q for q in qs if q["k"] == k]) for k in KS},
            "adopt_for_january": adopt,
            "rule": "held-out gain mean > 0, 95% lower bound > 0 (t over 12 folds), median per-question gain >= 0",
            "note": "folds share 11 of 12 dates: close to an in-sample test of one w"}


def score_batches(off: str, ours: str, mc: str | None) -> dict:
    meta = json.loads(OUTCOMES.read_text())
    a0, a2 = rows(off), rows(ours)
    qs, notes = scored(meta["questions"], set(a0) & set(a2))
    numeric = [q for q in meta["questions"] if q["question_type"] == "numeric"]
    fails = {b: failures(b, {q["id"] for q in numeric}) for b in (off, ours)}
    result = {"scored_cells": notes, "questions": len(qs),
              "primaries_replaced_or_lost": sum(n["missing_primaries"] for n in notes.values()),
              "unresolvable": sum(q["outcome"] is None for q in numeric),
              "parse_failures": fails,
              "parse_failure_stop_rule": any(f["share"] > 0.05 for f in fails.values())}
    for name, preds in (("A2", a2), ("A0", a0)):
        result[f"calibration_{name}"] = calibration(qs, preds)
        result[f"calibration_{name}_by_horizon"] = {h: calibration([q for q in qs if q["horizon_days"] == h], preds)
                                                    for h in round6.HORIZONS}
        result[f"library_{name}"] = {
            "final_cdf_coverage": cluster_stat({q["id"]: int(0.1 <= sn.pit(preds[q["id"]]["cdf"], q["outcome"],
                                                q["lower_bound"], q["upper_bound"]) <= 0.9) for q in qs}, qs),
            "pit_deciles": dict(sorted(Counter(min(int(10 * sn.pit(preds[q["id"]]["cdf"], q["outcome"], q["lower_bound"],
                                                                     q["upper_bound"])), 9) for q in qs).items()))}
        sc = {q["id"]: score_of(q, preds[q["id"]]["cdf_submitted"]) for q in qs}
        result[f"score_{name}"] = cluster_stat(sc, qs, True)
        result[f"score_{name}_by_horizon"] = {h: cluster_stat(sc, [q for q in qs if q["horizon_days"] == h], True)
                                              for h in round6.HORIZONS}
        result[f"score_{name}_by_family"] = {f: statistics.fmean(sc[q["id"]] for q in qs if q["family"] == f)
                                             for f in sorted({q["family"] for q in qs})}
        result[f"by_k_{name}"] = {k: {
            "coverage": statistics.fmean(sn.covered(preds[q["id"]]["percentiles"], q["outcome"]) for q in sub),
            "declared_width_in_spreads": statistics.median(
                (sn.declared_at(preds[q["id"]]["percentiles"], 0.9) - sn.declared_at(preds[q["id"]]["percentiles"], 0.1))
                / q["spread"] for q in sub),
            "score_mean": statistics.fmean(score_of(q, preds[q["id"]]["cdf_submitted"]) for q in sub)}
            for k in KS if (sub := [q for q in qs if q["k"] == k])}
    diff = {q["id"]: score_of(q, a2[q["id"]]["cdf_submitted"]) - score_of(q, a0[q["id"]]["cdf_submitted"]) for q in qs}
    result["A2_minus_A0_score"] = cluster_stat(diff, qs, True)
    result["A2_minus_A0_drop_one_T"] = {t: cluster_stat(diff, [q for q in qs if q["cluster"] != t], True)["mean"]
                                        for t in sorted({q["cluster"] for q in qs})}
    refs = {"R_mid": lambda q: sn.normal_percentiles((q["lower_bound"] + q["upper_bound"]) / 2,
                                                     (q["upper_bound"] - q["lower_bound"]) / 10),
            "R_data": lambda q: sn.normal_percentiles(q["freeze_value"] + q["mean_change"], q["spread"])}
    for name, f in refs.items():
        result[f"score_{name}"] = cluster_stat({q["id"]: score_of(q, sn.submitted(f(q), q["lower_bound"],
                                                                                   q["upper_bound"])) for q in qs}, qs, True)
    result["widening_A2"] = widening(qs, a2)
    result["by_family_A2"] = {f: {"n": len(sub), "coverage": statistics.fmean(
        sn.covered(a2[q["id"]]["percentiles"], q["outcome"]) for q in sub)}
        for f in sorted({q["family"] for q in qs}) if (sub := [q for q in qs if q["family"] == f])}
    if mc:
        m = rows(mc)
        smoke = [q for q in meta["questions"] if q["question_type"] == "multiple_choice"]
        parsed = [q for q in smoke if q["id"] in m and q["outcome"] is not None
                  and set(m[q["id"]]["options"]) == set(q["options"])]
        result["mc_smoke"] = {"questions": len(smoke), "parsed": sum(q["id"] in m for q in smoke),
                              "option_names_match": len(parsed),
                              "option_order_ok": sum(list(m[q["id"]]["options"]) == q["options"] for q in parsed),
                              "mean_baseline": statistics.fmean(mc_baseline_row(q, m[q["id"]]) for q in parsed)
                              if parsed else None}
    return result


def failures(batch: str, ids: set[str]) -> dict:
    """Questions whose last row in the batch is an error (after retries on resume)."""
    last = {}
    for line in (score.RUNS / batch / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"] in ids:
            last[r["id"]] = r.get("prediction") is not None
    failed = sum(not ok for ok in last.values())
    return {"forecast": len(last), "failed": failed, "share": failed / len(last) if last else 0.0}


def mc_baseline_row(q: dict, row: dict) -> float:
    probs = [row["options"].get(o, 0.0) for o in q["options"]]
    return sn.mc_baseline(probs, q["outcome"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["build", "resolve", "score"])
    parser.add_argument("--off")
    parser.add_argument("--ours")
    parser.add_argument("--mc")
    args = parser.parse_args()
    if args.step == "score":
        result = score_batches(args.off, args.ours, args.mc)
        round5.OUT.mkdir(exist_ok=True)
        path = round5.OUT / f"round7-score-{args.off}-vs-{args.ours}.json"
        path.write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps({k: result[k] for k in ("questions", "calibration_A2", "widening_A2")}, indent=1,
                         default=str)[:3000])
        print(f"wrote {path}")
        return
    import dotenv

    dotenv.load_dotenv()  # FRED_API_KEY for the fetchers; never printed
    http = dr.PoliteHttp(cache=dr.CACHE_DIR)
    path = ITEMS if args.step == "build" else OUTCOMES
    if path.exists():
        sys.exit(f"{path} exists; item lists are never overwritten.")
    try:
        result = build(http) if args.step == "build" else resolve(http)
    except dr.DataAuthError as exc:
        sys.exit(f"stopped: the data source refused the key or it is missing: {dr.redact(str(exc))}")
    path.write_text(json.dumps(result, indent=1, default=str))
    qs = result["questions"]
    print(f"wrote {len(qs)} questions to {path} (requests {http.requests}, cache hits {http.cache_hits})")
    print(json.dumps(Counter((q["question_type"], q["role"]) for q in qs).most_common()))


if __name__ == "__main__":
    main()
