"""
Round 5, Step 0 and builders (benchmarks/PLAN-refine-round5.md). No model calls.

    # coverage: one Meteo-France station series through DBnomics
    uv run python -m backtest.round5 coverage
    # F1: our last value and date before T vs ForecastBench's freeze value, per (series, freeze)
    uv run python -m backtest.round5 f1
    # F2: P(up) from past h-step changes, scored against earlier sets' base rates and 0.5 (context only)
    uv run python -m backtest.round5 f2
    # X3 (exploratory): self-made level questions from the same series; not run in Step 0
    uv run python -m backtest.round5 x3 --n 60

Reads backtest/data/datasets-<commit>.json (backtest.fetch --datasets). Public data requests go through
data_research.PoliteHttp (2 s apart, cached on disk); the FRED key is read from the environment at call time and
never printed (messages are redacted). Writes backtest/out/round5-<step>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import data_research as dr
from backtest import fetch, score

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "out"
DEFAULT_ITEMS = DATA / "datasets-86b92ba06ed3.json"
COVERAGE_SERIES = ("meteofrance", "TEMPERATURE", "celsius.07117.D")  # Ploumanac'h, daily
WEATHER_WINDOW_DAYS = 15
STAT_YEARS = 2  # F2: non-weather changes from the last two years before T
F1_PASS = 0.95


def load_items(path: Path) -> dict:
    return json.loads(path.read_text())


def _d(text: str) -> date:
    return date.fromisoformat(text[:10])


def decimals(x: float) -> int:
    text = repr(float(x))
    if "e" in text:
        return 6
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def same_value(a: float, b: float, places: int) -> bool:
    """Equal to the reference's stated precision (half a unit in its last place)."""
    return abs(a - b) <= 0.5 * 10 ** -places + 1e-9


def full_rows(item: dict, http) -> list[tuple[date, float]]:
    """Every row the source gives for the item, not cut at T: FRED as published the day before T (ALFRED),
    DBnomics as served today."""
    if item["source"] == "fred":
        return dr.fred_vintage_rows(item["series_id"], _d(item["freeze_datetime"]) - timedelta(days=1), http)
    return dr.dbnomics_rows(dr.dbnomics_doc(*dr.dbnomics_id(item["series_id"]), http))


# ---------------------------------------------------------------- coverage


def coverage(http) -> dict:
    t = datetime(2026, 8, 6)
    s = dr.dbnomics(*COVERAGE_SERIES, t, http)
    return {"series": "/".join(COVERAGE_SERIES), "name": s.name, "rows_before_cut": len(s.rows),
            "cutoff": s.cutoff.isoformat(), "last": [s.rows[-1][0].isoformat(), s.rows[-1][1]],
            "notes": s.notes, "ok": True}


# ---------------------------------------------------------------- F1


def f1_item(item: dict, http) -> dict:
    t = datetime.fromisoformat(item["freeze_datetime"])
    out = {"id": item["id"], "source": item["source"], "series_id": item["series_id"],
           "freeze": item["freeze_datetime"][:10], "fb_value": item["freeze_value"]}
    try:
        s = dr.series_for(item["source"], item["series_id"], t, http)
        rows = full_rows(item, http)
    except dr.DataAuthError:
        raise
    except dr.DataUnavailable as exc:
        return {**out, "status": "unavailable", "error": dr.redact(str(exc))[:300]}
    places = decimals(item["freeze_value"])
    last_d, last_v = s.rows[-1]
    matches = [d for d, v in rows if same_value(v, item["freeze_value"], places)]
    before_t = [d for d in matches if d < t.date()]
    # ForecastBench gives no date for its value: take the latest run of consecutive rows (before T) holding it,
    # and date it by the run's first row, so a repeated value never dates ForecastBench later than it could be.
    fb_date, ambiguous = None, False
    if before_t:
        dates = [d for d, _ in rows if d < t.date()]
        i = dates.index(max(before_t))
        while i > 0 and dates[i - 1] in set(before_t):
            i -= 1
        fb_date = dates[i]
        ambiguous = any(d < fb_date for d in before_t)  # the value also appears earlier, apart from this run
    out.update(our_date=last_d.isoformat(), our_value=last_v, cutoff=s.cutoff.isoformat(), places=places,
               agree=same_value(last_v, item["freeze_value"], places),
               fb_value_date=fb_date.isoformat() if fb_date else None, fb_value_date_ambiguous=ambiguous,
               fb_value_only_after_t=bool(matches) and not before_t,
               fb_value_in_our_rows=any(same_value(v, item["freeze_value"], places) for _, v in s.rows))
    if fb_date is not None:
        out["our_minus_fb_days"] = (last_d - fb_date).days
        out["later_than_fb"] = last_d > fb_date
    if out["agree"]:
        out["status"] = "agree"  # our last row holds ForecastBench's value, so it is not later than theirs
    elif fb_date is None:
        out["status"] = "undetermined"  # value not found before T: the leak check can't run
    else:
        out["status"] = "later" if out["later_than_fb"] else "disagree"
    return out


def f1_summary(rows: list[dict]) -> dict:
    def part(sel: list[dict]) -> dict:
        ok = [r for r in sel if r["status"] != "unavailable"]
        return {
            "items": len(sel), "fetched": len(ok),
            "unavailable": len(sel) - len(ok),
            "agree": sum(r["agree"] for r in ok),
            "agree_share_of_fetched": round(sum(r["agree"] for r in ok) / len(ok), 4) if ok else None,
            "later_than_fb": sum(bool(r.get("later_than_fb")) for r in ok),
            "undetermined": sum(r["status"] == "undetermined" for r in ok),
            "fb_date_ambiguous": sum(bool(r.get("fb_value_date_ambiguous")) for r in ok),
            "fb_value_only_after_t": sum(r["fb_value_only_after_t"] for r in ok),
            "fb_value_not_found": sum(r["fb_value_date"] is None and not r["fb_value_only_after_t"] for r in ok),
            "disagree_but_fb_value_in_our_rows": sum((not r["agree"]) and r["fb_value_in_our_rows"] for r in ok),
            "our_minus_fb_days": dict(sorted(_count(r.get("our_minus_fb_days") for r in ok if not r["agree"]).items(),
                                             key=lambda kv: (kv[0] is None, kv[0] or 0))),
        }

    by_source = {src: part([r for r in rows if r["source"] == src]) for src in sorted({r["source"] for r in rows})}

    def family(r: dict) -> str:  # DBnomics provider/dataset; "later" is judged per family against release timing
        return r["source"] if r["source"] != "dbnomics" else "dbnomics:" + "/".join(r["series_id"].split("_", 2)[:2])

    later_by_family = _count(family(r) for r in rows if r.get("later_than_fb"))
    whole = part(rows)
    share = whole["agree_share_of_fetched"]
    return {"all": whole, "by_source": by_source, "later_by_family": later_by_family,
            "pass_rule": f">= {F1_PASS:.0%} of items agree to the stated precision and none later than ForecastBench's",
            "passes": bool(share is not None and share >= F1_PASS and whole["later_than_fb"] == 0
                           and whole["unavailable"] == 0)}


def _count(values) -> dict:
    out: dict = defaultdict(int)
    for v in values:
        out[v] += 1
    return dict(out)


# ---------------------------------------------------------------- F2


def as_of(rows: list[tuple[date, float]], day: date) -> float | None:
    """The last value dated on or before `day`."""
    dates = [d for d, _ in rows]
    i = bisect_right(dates, day)
    return rows[i - 1][1] if i else None


def p_up(rows: list[tuple[date, float]], due: date, horizon: int, weather: bool) -> float | None:
    """Share of past h-day changes that were positive, from rows (all before T). Weather: start days within
    WEATHER_WINDOW_DAYS of the due date's calendar day in earlier years; otherwise the last STAT_YEARS years."""
    if not rows:
        return None
    last = rows[-1][0]
    starts = []
    if weather:
        for years in range(1, 31):
            try:
                centre = due.replace(year=due.year - years)
            except ValueError:  # Feb 29
                centre = due.replace(year=due.year - years, day=28)
            starts += [centre + timedelta(days=k) for k in range(-WEATHER_WINDOW_DAYS, WEATHER_WINDOW_DAYS + 1)]
    else:
        first = last - timedelta(days=365 * STAT_YEARS)
        starts = [d for d, _ in rows if first <= d and d + timedelta(days=horizon) <= last]
    ups = n = 0
    for d in starts:
        if d + timedelta(days=horizon) > last or d < rows[0][0]:
            continue
        a, b = as_of(rows, d), as_of(rows, d + timedelta(days=horizon))
        if a is None or b is None:
            continue
        n += 1
        ups += b > a
    return ups / n if n >= 10 else None


def base_rates(commit: str, before: str) -> dict:
    """Yes rates of fred and dbnomics items in sets frozen before `before`, by (source, horizon days) and source."""
    listing = fetch._get_json(f"https://api.github.com/repos/{fetch.REPO}/contents/datasets/resolution_sets?ref={commit}")
    dates = sorted(i["name"].split("_")[0] for i in listing if i["name"].endswith("_resolution_set.json"))
    raw = f"https://raw.githubusercontent.com/{fetch.REPO}/{commit}/datasets"
    sets = []
    for d in dates:
        if d < "2026-01-01" or d >= "2026-08-01":
            continue
        try:
            q = fetch._get_json(f"{raw}/question_sets/{d}-llm.json")
        except Exception:
            continue
        sets.append((d, q, fetch._get_json(f"{raw}/resolution_sets/{d}_resolution_set.json")))
    early = [s for s in sets if s[1]["questions"] and s[1]["questions"][0]["freeze_datetime"][:10] < before]
    items, _ = fetch.select_datasets(early, "2000-01-01")
    items = [i for i in items if i["freeze_datetime"][:10] < before]
    by_h, by_src = defaultdict(list), defaultdict(list)
    for i in items:
        by_h[f"{i['source']}:{i['horizon_days']}"].append(i["outcome"])
        by_src[i["source"]].append(i["outcome"])
    return {"sets": [s[0] for s in early], "items": len(items),
            "by_horizon": {k: [statistics.fmean(v), len(v)] for k, v in sorted(by_h.items())},
            "by_source": {k: [statistics.fmean(v), len(v)] for k, v in sorted(by_src.items())}}


def f2(items: list[dict], http, rates: dict) -> dict:
    rows_out = []
    for item in items:
        t = datetime.fromisoformat(item["freeze_datetime"])
        try:
            s = dr.series_for(item["source"], item["series_id"], t, http)
        except dr.DataUnavailable:
            continue
        weather = item["series_id"].startswith("meteofrance_")
        p = p_up(s.rows, _d(item["forecast_due_date"]), item["horizon_days"], weather)
        br = rates["by_horizon"].get(f"{item['source']}:{item['horizon_days']}",
                                     rates["by_source"].get(item["source"], [0.5]))[0]
        rows_out.append({"id": item["id"], "source": item["source"], "cluster": item["cluster"],
                         "outcome": item["outcome"], "p_stat": p, "p_base": br})
    scored = [r for r in rows_out if r["p_stat"] is not None]

    def brier(key, sel):
        return statistics.fmean((min(max(r[key], 0.01), 0.99) - r["outcome"]) ** 2 for r in sel) if sel else None

    def half(sel):
        return statistics.fmean((0.5 - r["outcome"]) ** 2 for r in sel) if sel else None

    out = {"items": len(items), "fetched": len(rows_out), "with_p_stat": len(scored),
           "brier": {"statistical": brier("p_stat", scored), "earlier_base_rate": brier("p_base", scored),
                     "half": half(scored)},
           "by_source": {}}
    for src in sorted({r["source"] for r in scored}):
        sel = [r for r in scored if r["source"] == src]
        out["by_source"][src] = {"n": len(sel), "statistical": brier("p_stat", sel),
                                 "earlier_base_rate": brier("p_base", sel), "half": half(sel),
                                 "yes_rate": statistics.fmean(r["outcome"] for r in sel)}
    diffs = {r["id"]: (min(max(r["p_base"], .01), .99) - r["outcome"]) ** 2
             - (min(max(r["p_stat"], .01), .99) - r["outcome"]) ** 2 for r in scored}
    groups = {r["id"]: r["cluster"] for r in scored}
    if len(set(groups.values())) > 1:
        m, lo, hi, p = score.cluster_t(diffs, groups)
        out["statistical_vs_base_rate"] = {"gain": m, "ci95": [lo, hi], "p": p, "clusters": len(set(groups.values())),
                                           "test": "t-test on cluster means, df = clusters - 1"}
    out["rows"] = rows_out
    return out


# ---------------------------------------------------------------- X3 builder (exploratory, no model calls)

X3_MULTIPLES = (-1, 0, 1)


def x3_question(item: dict, rows_before_t: list[tuple[date, float]], multiple: int, name: str) -> dict | None:
    """A level question set at T: will the series be above (value at T + multiple x the sd of past h-day
    changes) on the item's resolution date? Outcome filled later from fetched values (x3_resolve)."""
    h = item["horizon_days"]
    last = rows_before_t[-1][0]
    changes = []
    for d, v in rows_before_t:
        if d < last - timedelta(days=365 * STAT_YEARS) or d + timedelta(days=h) > last:
            continue
        later = as_of(rows_before_t, d + timedelta(days=h))
        if later is not None:
            changes.append(later - v)
    if len(changes) < 10:
        return None
    spread = statistics.pstdev(changes)
    value = rows_before_t[-1][1]
    threshold = round(value + multiple * spread, 6)
    due = item["resolution_date"]
    qid = hashlib.sha256(f"x3|{item['id']}|{multiple}".encode()).hexdigest()[:16]
    return {"id": qid, "source": item["source"], "series_id": item["series_id"], "group": item["group"],
            "cluster": item["cluster"], "self_made": True, "multiple": multiple, "threshold": threshold,
            "question": f"Will {name} be above {threshold:g} on {due}?",
            "background": item["background"],
            "criteria": (f"Resolves Yes if the value of {name} for {due} (the latest observation dated on or "
                         f"before {due}) is strictly above {threshold:g}. Self-made test question."),
            "freeze_datetime": item["freeze_datetime"], "resolution_date": due, "crowd": None, "outcome": None}


def x3_resolve(q: dict, rows_now: list[tuple[date, float]]) -> int | None:
    v = as_of(rows_now, _d(q["resolution_date"]))
    return None if v is None else int(v > q["threshold"])


def x3_pick(items: list[dict], n: int) -> list[tuple[dict, int]]:
    per = n // len(X3_MULTIPLES)
    out = []
    for i, m in enumerate(X3_MULTIPLES):
        ranked = sorted(items, key=lambda it: hashlib.sha256(f"x3pick|{m}|{it['id']}".encode()).hexdigest())
        out += [(it, m) for it in ranked[:per]]
    return out


def x3(items: list[dict], http, n: int) -> dict:
    questions = []
    for item, m in x3_pick(items, n):
        t = datetime.fromisoformat(item["freeze_datetime"])
        s = dr.series_for(item["source"], item["series_id"], t, http)
        q = x3_question(item, s.rows, m, s.name)
        if q is None:
            continue
        now = (dr.fred_vintage_rows(item["series_id"], None, http) if item["source"] == "fred"
               else dr.dbnomics_rows(dr.dbnomics_doc(*dr.dbnomics_id(item["series_id"]), http)))
        q["outcome"] = x3_resolve(q, now)
        if q["outcome"] is not None:
            questions.append(q)
    return {"self_made": True, "multiples": list(X3_MULTIPLES), "sha256": fetch.list_hash(questions),
            "questions": questions}


# ---------------------------------------------------------------- main


def write(name: str, result: dict) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"round5-{name}.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["coverage", "f1", "f2", "x3"])
    parser.add_argument("--items", default=str(DEFAULT_ITEMS))
    parser.add_argument("--n", type=int, default=60, help="x3: number of self-made questions")
    args = parser.parse_args()
    import dotenv

    dotenv.load_dotenv()  # FRED_API_KEY for the fetchers; never printed
    http = dr.PoliteHttp(cache=dr.CACHE_DIR)
    meta = load_items(Path(args.items))
    items = meta["questions"]
    started = datetime.now().isoformat(timespec="seconds")
    try:
        if args.step == "coverage":
            result = coverage(http)
        elif args.step == "f1":
            rows = []
            for i, item in enumerate(items, 1):
                rows.append(f1_item(item, http))
                if i % 50 == 0:
                    print(f"f1 {i}/{len(items)} (requests {http.requests}, cache hits {http.cache_hits})", flush=True)
            result = {"summary": f1_summary(rows), "rows": rows}
        elif args.step == "f2":
            rates = base_rates(meta["dataset_commit"], "2026-08-01")
            result = {"base_rates": rates, **f2(items, http, rates)}
        else:
            path = DATA / f"x3-levels-{Path(args.items).stem}.json"
            if path.exists():
                sys.exit(f"{path} exists; item lists are never overwritten.")
            result = x3(items, http, args.n)
            path.write_text(json.dumps(result, indent=1, default=str))
            print(f"wrote {len(result['questions'])} self-made questions to {path}")
    except dr.DataAuthError as exc:
        sys.exit(f"stopped: the data source refused the key or it is missing: {dr.redact(str(exc))}")
    result = {"step": args.step, "items_file": Path(args.items).name, "items_sha256": meta["sha256"],
              "data_research_sha256": hashlib.sha256(Path(dr.__file__).read_bytes()).hexdigest(),
              "started": started, "requests": http.requests, "cache_hits": http.cache_hits, **result}
    path = write(args.step, result)
    summary = result.get("summary") or {k: v for k, v in result.items() if k not in ("rows", "questions")}
    print(json.dumps(summary, indent=1, default=str)[:3000])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
