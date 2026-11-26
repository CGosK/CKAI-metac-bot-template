"""
Round 6: self-made level questions (benchmarks/PLAN-refine-round6-levels.md).

    uv run python -m backtest.round6 build     # public data requests, no model calls
    uv run python -m backtest.round6 resolve   # on or after 2026-12-05, public data requests
    uv run python -m backtest.round6 score --off r6-a0 --ours r6-a2   # no model calls or requests

build: for each T (12 Wednesdays) and horizon (7, 30 days), walks the (series, multiple) candidates in
sha256("round6:" + T + horizon + series + multiple) order, fetching each series as published before T (the same
data_research.series_for call the A2 block makes), and accepts 20 primaries and 3 backups under the plan's
exclusion, family and balance rules. Writes data/round6-levels.json (never overwritten) and lists every excluded
series with its reason.

resolve: outcomes from the ALFRED vintage 7 (daily) or 14 (weekly) days after the due date for FRED, and from
values fetched at resolve time for DBnomics; fail closed when the observation used is more than 7 days before
the due date. Writes data/round6-outcomes.json (the build's questions with outcomes, never overwritten).

score: the harm-test decision and the claim bar, with drop-one-T and drop-one-family checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import data_research as dr
from backtest import fetch, round5, score

DATA = round5.DATA
LEVELS = DATA / "round6-levels.json"
OUTCOMES = DATA / "round6-outcomes.json"
T_DATES = [date(2026, 8, 5) + timedelta(weeks=i) for i in range(12)]  # Wednesdays, Aug 5 to Oct 21
HORIZONS = (7, 30)
MULTIPLES = (-1.0, -0.5, 0.5, 1.0)
PER_CLUSTER, BACKUPS, FAMILY_CAP, PER_MULTIPLE = 20, 3, 3, 5
HISTORY_DAYS = 2 * 365
YES_RATE_BOUNDS = (0.10, 0.90)
STICKY_ZERO_SHARE = 0.5
RESOLVE_FROM = date(2026, 12, 5)
VINTAGE_DAYS = {"daily": 7, "weekly": 14}
STALE_DAYS = 7
MAX_ANCHOR_AGE_DAYS = 21  # plan v3.1: a series whose last row before T is older can't resolve within the window
DBNOMICS_DATASETS = {("ECB", "EXR"), ("ECB", "YC"), ("meteofrance", "TEMPERATURE")}  # rarely revised, daily

# Policy-administered rates: set by, or averaging, a policy rate within the horizon (plan, Fixes item 4).
POLICY_RATE = re.compile(
    r"SOFR|Federal Funds (Effective|Target)|Interest Rate on Reserve|Reverse Repurchase Agreements Award Rate"
    r"|Euro Short-Term Rate|Deposit Facility Rate|Key interest rate|Discount Window|Bank Prime Loan"
    r"|Overnight Bank Funding|Overnight Index Average|(4-Week|3-Month) Treasury Bill|90-Day .*Commercial Paper"
    r"|at (1|3)-Month Constant Maturity", re.IGNORECASE)

# FRED families by name (plan, Fixes item 6); first match wins.
FRED_FAMILIES = [
    ("corporate-bonds", r"ICE BofA|Moody's"),
    ("h41-fed", r"Wednesday Level|Week Average|Liabilities and Capital|^Assets:|Reserve Balances"
                r"|Repurchase Agreements|Deposits with F\.R\."),  # before us-rates: some H.4.1 titles say Treasury
    ("us-rates", r"Treasury|Real Interest Rate|Expected Inflation|Breakeven|Forward Inflation|Term Premium"),
    ("h8-banks", r"All Commercial Banks|Bank Credit"),
    ("fx", r"Exchange Rate|Dollar Index"),
    ("claims", r"Claims"),
    ("mortgage", r"Mortgage"),
    ("volatility", r"Volatility Index|VIX"),
    ("conditions", r"Financial Conditions|Stress|Uncertainty|Weekly Economic Index"),
    ("equity-crypto", r"NASDAQ|Nikkei|Bitcoin"),
    ("energy", r"Crude|Natural Gas|Gas Price"),
    ("jobs", r"Job Postings"),
    ("money", r"^M2$|Money Market"),
]


# ---------------------------------------------------------------- rules (pure, tested offline)


def family(source: str, series_id: str, name: str) -> str:
    if source == "dbnomics":
        provider, dataset, _ = dr.dbnomics_id(series_id)
        return f"{provider}/{dataset}"
    for fam, pattern in FRED_FAMILIES:
        if re.search(pattern, name):
            return fam
    return "fred-other"


def frequency(rows: list[tuple[date, float]]) -> str | None:
    """daily (median gap up to 3 days) or weekly (5 to 9); anything else is out of scope."""
    recent = rows[-60:]
    if len(recent) < 10:
        return None
    gap = statistics.median((b - a).days for (a, _), (b, _) in zip(recent, recent[1:]))
    return "daily" if gap <= 3 else "weekly" if 5 <= gap <= 9 else None


def changes(rows: list[tuple[date, float]], h: int) -> list[float]:
    """Past h-day changes over the HISTORY_DAYS before the last row (all rows are before T)."""
    last = rows[-1][0]
    out = []
    for d, v in rows:
        if d < last - timedelta(days=HISTORY_DAYS) or d + timedelta(days=h) > last:
            continue
        later = round5.as_of(rows, d + timedelta(days=h))
        if later is not None:
            out.append(later - v)
    return out


def places(rows: list[tuple[date, float]]) -> int:
    """Decimals shown for a threshold: one more than the series' own precision, at most 6."""
    return min(6, max(round5.decimals(v) for _, v in rows[-20:]) + 1)


def threshold(rows: list[tuple[date, float]], h: int, m: float) -> dict | None:
    """Anchor (last row) + mean past h-day change + m × spread, rounded to places(); None when history is short."""
    ch = changes(rows, h)
    if len(ch) < 20:
        return None
    anchor = rows[-1][1]
    mean, spread = statistics.fmean(ch), statistics.pstdev(ch)
    p = places(rows)
    value = round(anchor + mean + m * spread, p)
    yes_rate = sum(c > value - anchor for c in ch) / len(ch)
    return {"anchor": anchor, "anchor_date": rows[-1][0].isoformat(), "mean_change": mean, "spread": spread,
            "threshold": value, "places": p, "yes_rate": yes_rate,
            "zero_share": sum(c == 0 for c in ch) / len(ch), "never_negative": min(v for _, v in rows) >= 0}


def exclusion(source: str, series_id: str, name: str, freq: str | None, t: dict | None,
              t_date: date | None = None) -> str | None:
    """Why a candidate is out, or None (plan, Fixes item 4; publication lag, v3.1)."""
    if source == "dbnomics" and dr.dbnomics_id(series_id)[:2] not in DBNOMICS_DATASETS:
        return "dbnomics dataset outside the rarely-revised daily set"
    if POLICY_RATE.search(name):
        return "policy-administered rate"
    if freq is None:
        return "not daily or weekly"
    if t is None:
        return "under 20 past changes"
    if t_date is not None and date.fromisoformat(t["anchor_date"]) < t_date - timedelta(days=MAX_ANCHOR_AGE_DAYS):
        return "published with a lag over 21 days"
    if t["zero_share"] > STICKY_ZERO_SHARE:
        return "sticky"
    if t["spread"] == 0:
        return "no spread"
    if t["threshold"] < 0 and t["never_negative"]:
        return "negative threshold on a non-negative series"
    if not YES_RATE_BOUNDS[0] <= t["yes_rate"] <= YES_RATE_BOUNDS[1]:
        return "historical Yes rate outside 10% to 90%"
    return None


def rank_key(t_date: date, h: int, key: str, m: float) -> str:
    return hashlib.sha256(f"round6:{t_date.isoformat()}{h}{key}{m:+.1f}".encode()).hexdigest()


def draw(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """The plan's greedy walk over candidates already in rank order, each with key, family, multiple and
    `ok` (passes the exclusions). 20 primaries (one per series, family cap 3, 5 per multiple), then 3 backups
    (series and family rules only)."""
    primaries, backups, used, fam = [], [], set(), Counter()
    per_m = Counter()
    for c in candidates:
        if not c["ok"] or c["key"] in used or fam[c["family"]] >= FAMILY_CAP:
            continue
        if len(primaries) < PER_CLUSTER:
            if per_m[c["multiple"]] >= PER_MULTIPLE:
                continue
            primaries.append(c)
            per_m[c["multiple"]] += 1
        elif len(backups) < BACKUPS:
            backups.append(c)
        else:
            break
        used.add(c["key"])
        fam[c["family"]] += 1
    return primaries, backups


def show(value: float, p: int) -> str:
    return f"{value:.{p}f}"  # fixed decimals, never scientific


# ---------------------------------------------------------------- build


def background_for(items: list[dict]) -> dict[str, str]:
    """Series key -> background text from a round-5 item of that series, its freeze value removed."""
    from backtest import run as bt

    out = {}
    for q in sorted(items, key=lambda q: q["id"]):
        k = f"{q['source']}:{q['series_id']}"
        if k not in out:
            out[k] = bt.scrub_value(q["background"], q["freeze_value"])[0]
    return out


def build(http) -> dict:
    meta = round5.load_items(round5.DEFAULT_ITEMS)
    items = meta["questions"]
    keys = sorted({(q["source"], q["series_id"]) for q in items})
    backgrounds = background_for(items)
    fetched: dict[tuple, object] = {}
    excluded: dict[str, Counter] = defaultdict(Counter)
    questions = []

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

    for t_date in T_DATES:
        for h in HORIZONS:
            cands = sorted(((s, sid, m) for s, sid in keys for m in MULTIPLES),
                           key=lambda c: rank_key(t_date, h, f"{c[0]}:{c[1]}", c[2]))
            walk = []
            for source, sid, m in cands:
                key = f"{source}:{sid}"
                if source == "dbnomics" and dr.dbnomics_id(sid)[:2] not in DBNOMICS_DATASETS:
                    excluded[key]["dbnomics dataset outside the rarely-revised daily set"] += 1
                    continue
                s = series(source, sid, t_date)
                if isinstance(s, Exception):
                    excluded[key][f"unavailable: {type(s).__name__}"] += 1
                    continue
                t = threshold(s.rows, h, m)
                why = exclusion(source, sid, s.name, frequency(s.rows), t, t_date)
                if why:
                    excluded[key][why] += 1
                walk.append({"key": key, "source": source, "series_id": sid, "name": s.name, "units": s.units,
                             "family": family(source, sid, s.name), "multiple": m, "ok": why is None,
                             "freq": frequency(s.rows), "t": t})
                primaries, backups = draw(walk)
                if len(primaries) == PER_CLUSTER and len(backups) == BACKUPS:
                    break
            primaries, backups = draw(walk)
            for role, chosen in (("primary", primaries), ("backup", backups)):
                for rank, c in enumerate(chosen):
                    questions.append(make_question(c, t_date, h, role, rank, backgrounds.get(c["key"], "")))
            print(f"T {t_date} h {h}: {len(primaries)} primaries, {len(backups)} backups "
                  f"(requests {http.requests}, cache hits {http.cache_hits})", flush=True)
    return {"round": 6, "plan": "PLAN-refine-round6-levels.md", "self_made": True,
            "dataset_commit": meta["dataset_commit"], "freeze_from": "2026-08-01",
            "source_items_sha256": meta["sha256"],
            "data_research_sha256": hashlib.sha256(Path(dr.__file__).read_bytes()).hexdigest(),
            "excluded_note": "per series: reasons and how many (T, horizon, multiple) candidates walked were "
                             "excluded for each; candidates after a cell filled were not walked",
            "excluded": {k: dict(v) for k, v in sorted(excluded.items())},
            "families": dict(sorted(Counter(q["family"] for q in questions).items())),
            "sha256": fetch.list_hash(questions), "questions": questions}


def make_question(c: dict, t_date: date, h: int, role: str, rank: int, background: str) -> dict:
    t = c["t"]
    due = t_date + timedelta(days=h)
    thr = show(t["threshold"], t["places"])
    units = f" ({c['units']})" if c["units"] else ""
    qid = hashlib.sha256(f"r6|{t_date}|{h}|{c['key']}|{c['multiple']}".encode()).hexdigest()[:16]
    return {
        "id": qid, "source": c["source"], "series_id": c["series_id"], "group": c["key"],
        "cluster": t_date.isoformat(), "family": c["family"], "frequency": c["freq"], "role": role, "rank": rank,
        "multiple": c["multiple"], "horizon_days": h, "threshold": t["threshold"], "threshold_shown": thr,
        "anchor_date": t["anchor_date"], "yes_rate_history": t["yes_rate"],
        "question": f"Will {c['name']} be above {thr}{units} on {due.isoformat()}?",
        "background": background,
        "criteria": (f"Resolves Yes if the value of {c['name']} for {due.isoformat()} (the latest observation dated "
                     f"on or before {due.isoformat()}, "
                     + (f"as published {VINTAGE_DAYS[c['freq']]} days after that date"
                        if c["source"] == "fred" else "as published when the question is resolved")
                     + f") is strictly above {thr}{units}. Self-made test question."),
        "freeze_datetime": f"{t_date.isoformat()}T00:00:00+00:00", "resolution_date": due.isoformat(),
        "freeze_value": t["anchor"], "freeze_value_explanation": f"The latest value of {c['name']}",
        "crowd": None, "outcome": None,
    }


# ---------------------------------------------------------------- resolve


def resolve_one(q: dict, http, fresh=None, dbn_cache: dict | None = None) -> dict:
    due = date.fromisoformat(q["resolution_date"])
    if q["source"] == "fred":
        vintage = due + timedelta(days=VINTAGE_DAYS[q["frequency"]])
        rows = dr.fred_vintage_rows(q["series_id"], vintage, http, start=due - timedelta(days=60))
        how = f"ALFRED vintage {vintage.isoformat()}"
    else:
        # Uncached, one series fetched once: cached DBnomics documents end at the date they were first fetched.
        dbn_cache = {} if dbn_cache is None else dbn_cache
        if q["series_id"] not in dbn_cache:
            dbn_cache[q["series_id"]] = dr.dbnomics_rows(dr.dbnomics_doc(*dr.dbnomics_id(q["series_id"]),
                                                                         fresh or dr.PoliteHttp()))
        rows = dbn_cache[q["series_id"]]
        how = f"DBnomics fetched {date.today().isoformat()}"
    used = [(d, v) for d, v in rows if d <= due]
    stored = [[d.isoformat(), v] for d, v in rows if due - timedelta(days=21) <= d <= due + timedelta(days=7)]
    if not used:
        return {"outcome": None, "why": "no observation on or before the due date", "how": how, "rows": stored}
    d, v = used[-1]
    if d < due - timedelta(days=STALE_DAYS):
        return {"outcome": None, "why": f"stale: last observation {d.isoformat()}", "how": how, "rows": stored}
    return {"outcome": int(v > q["threshold"]), "obs_date": d.isoformat(), "obs_value": v, "how": how,
            "rows": stored}


def resolve(http) -> dict:
    if date.today() < RESOLVE_FROM:
        sys.exit(f"resolve runs on or after {RESOLVE_FROM} (plan: the last weekly vintage exists then)")
    meta = json.loads(LEVELS.read_text())
    fresh, dbn_cache, out, transient = dr.PoliteHttp(), {}, [], []
    for q in meta["questions"]:
        try:
            r = resolve_one(q, http, fresh, dbn_cache)
        except dr.DataTransient as exc:  # retryable: nothing is written, so a rerun starts clean
            transient.append(f"{q['id']} {dr.redact(str(exc))[:120]}")
            continue
        out.append({**q, "outcome": r["outcome"], "resolution": r})
    if transient:
        sys.exit(f"{len(transient)} transient fetch errors; nothing written, rerun resolve. First: {transient[0]}")
    return {**{k: v for k, v in meta.items() if k != "questions"},
            "levels_file_sha256": hashlib.sha256(LEVELS.read_bytes()).hexdigest(),
            "resolved_on": date.today().isoformat(), "questions": out}


# ---------------------------------------------------------------- score


def scored(questions: list[dict]) -> tuple[list[dict], dict]:
    """Per (T, horizon): resolvable primaries, each unresolvable one replaced by backups in rank order; a
    (T, horizon) still short of 20 is not scored."""
    by = defaultdict(list)
    for q in questions:
        by[(q["cluster"], q["horizon_days"])].append(q)
    out, notes = [], {}
    for k, qs in sorted(by.items()):
        prim = [q for q in sorted(qs, key=lambda q: q["rank"]) if q["role"] == "primary"]
        back = [q for q in sorted(qs, key=lambda q: q["rank"]) if q["role"] == "backup" and q["outcome"] is not None]
        keep = [q for q in prim if q["outcome"] is not None]
        need = PER_CLUSTER - len(keep)
        keep += back[:need]
        notes[f"{k[0]} h{k[1]}"] = {"primaries": len(prim),
                                    "unresolvable_primaries": sum(q["outcome"] is None for q in prim),
                                    "backups_used": min(need, len(back)), "scored": len(keep) >= PER_CLUSTER}
        if len(keep) >= PER_CLUSTER:
            out += keep
    return out, notes


def summary(r: dict) -> dict:
    return {k: r[k] for k in ("questions", "clusters", "brier", "gain", "mean_over_questions", "log_gain")}


def rationale_share(ours: str, off: str, qs: list[dict]) -> dict:
    """Round-5 string rule (round5_d1_report.states) on the A2 rationales, A0 as control."""
    from backtest import round5_d1_report as rep

    text = {b: rep.reasoning(b) for b in (ours, off)}
    blocks = {}
    for line in (score.RUNS / ours / "research.jsonl").read_text().splitlines():
        row = json.loads(line)
        m = rep.LATEST.search(row.get("text", ""))
        if m:
            blocks[row["id"]] = (date.fromisoformat(m.group(2)), m.group(1))
    out = {b: Counter() for b in (ours, off)}
    for q in qs:
        if q["id"] not in blocks:
            continue
        for b in (ours, off):
            hit = rep.states(text[b].get(q["id"], ""), *blocks[q["id"]])
            out[b]["items"] += 1
            out[b].update(k for k, ok in hit.items() if ok)
    return {"A2": dict(out[ours]), "A0_control": dict(out[off]), "rule": "round5_d1_report.states"}


def score_batches(off: str, ours: str) -> dict:
    score.QUESTIONS = OUTCOMES
    _, _, base = score.load([off])
    _, _, other = score.load([ours])
    meta = json.loads(OUTCOMES.read_text())
    qs, notes = scored(meta["questions"])
    qs = [q for q in qs if q["id"] in base and q["id"] in other]

    def cmp(sub):
        if len({q["cluster"] for q in sub}) < 2:
            return {"questions": len(sub), "note": "fewer than 2 clusters: no t-test",
                    "gain": {"mean": None, "ci95": [None, None], "p": None}}
        return summary(score.compare(sub, base, other, 1, None, 1, test="cluster-t"))

    main = cmp(qs)
    lo = main["gain"]["ci95"][0]
    result = {"scored_clusters": notes, "all": main,
              "decision_harm_test": "add the block" if main["gain"]["mean"] > 0 and lo > -0.01 else "no block",
              "harm_rule": "mean of cluster means > 0 and 95% lower bound > -0.01 (t-test on cluster means)",
              "claim_detected_gain": main["gain"]["mean"] >= 0.01 and main["gain"]["p"] < 0.05
              and main["log_gain"]["mean"] >= 0,
              "drop_one_T": {t: cmp([q for q in qs if q["cluster"] != t])["gain"]["mean"]
                             for t in sorted({q["cluster"] for q in qs})},
              "drop_one_family": {f: cmp([q for q in qs if q["family"] != f])["gain"]["mean"]
                                  for f in sorted({q["family"] for q in qs})}}
    qs_weather = [q for q in qs if q["family"] == "meteofrance/TEMPERATURE"]
    result["weather_vs_not"] = {"weather": cmp(qs_weather),
                                "not_weather": cmp([q for q in qs if q not in qs_weather])}
    result["by_outcome"] = {"resolved_yes": cmp([q for q in qs if q["outcome"] == 1]),
                            "resolved_no": cmp([q for q in qs if q["outcome"] == 0])}
    result["weekly_7_day"] = cmp([q for q in qs if q["frequency"] == "weekly" and q["horizon_days"] == 7])
    result["rationale_states_last_row"] = rationale_share(ours, off, qs)
    result["fragile"] = any(g is not None and g < 0
                            for g in [*result["drop_one_T"].values(), *result["drop_one_family"].values()])
    for name, key in (("by_multiple", "multiple"), ("by_horizon", "horizon_days"), ("by_family", "family"),
                      ("by_source", "source")):
        result[name] = {str(v): cmp([q for q in qs if q[key] == v]) for v in sorted({q[key] for q in qs})}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["build", "resolve", "score"])
    parser.add_argument("--off")
    parser.add_argument("--ours")
    args = parser.parse_args()
    if args.step == "score":
        result = score_batches(args.off, args.ours)
        path = round5.write(f"6-score-{args.off}-vs-{args.ours}", result)
        a = result["all"]
        print(f"n={a['questions']} clusters {a['clusters']} gain {a['gain']['mean']:+.4f} "
              f"{[round(x, 4) for x in a['gain']['ci95']]} p {a['gain']['p']:.3f} log {a['log_gain']['mean']:+.4f}")
        print(f"decision: {result['decision_harm_test']}; detected gain: {result['claim_detected_gain']}; "
              f"fragile: {result['fragile']}")
        print(f"wrote {path}")
        return
    import dotenv

    dotenv.load_dotenv()  # FRED_API_KEY for the fetchers; never printed
    http = dr.PoliteHttp(cache=dr.CACHE_DIR)
    path = LEVELS if args.step == "build" else OUTCOMES
    if path.exists():
        sys.exit(f"{path} exists; item lists are never overwritten.")
    try:
        result = build(http) if args.step == "build" else resolve(http)
    except dr.DataAuthError as exc:
        sys.exit(f"stopped: the data source refused the key or it is missing: {dr.redact(str(exc))}")
    path.write_text(json.dumps(result, indent=1, default=str))
    qs = result["questions"]
    print(f"wrote {len(qs)} questions to {path} (requests {http.requests}, cache hits {http.cache_hits})")
    print(json.dumps(Counter((q["role"], q["horizon_days"]) for q in qs).most_common()))
    if args.step == "resolve":
        print(json.dumps(Counter(str(q["outcome"]) for q in qs)))


if __name__ == "__main__":
    main()
