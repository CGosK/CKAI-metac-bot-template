"""
Build the backtest's question list from ForecastBench (no model calls).

Downloads ForecastBench's question and resolution sets at one pinned commit,
keeps binary market questions (Manifold, Metaculus, Polymarket) frozen on or
after the cutoff that resolved to 0 or 1, and writes them to
backtest/data/questions.json with the dataset commit and a SHA-256 of the list.

ForecastBench datasets: https://github.com/forecastingresearch/forecastbench-datasets
(CC-BY-SA-4.0). The data is downloaded at run time and never committed.

Usage:
    uv run python -m backtest.fetch [--commit SHA] [--freeze-from 2026-08-01]
    uv run python -m backtest.fetch --exclude-list backtest/data/questions.json   # held-out list

With --exclude-list, the output is a held-out list (PLAN-refine-round2.md):
questions not in the given list and sharing no event with it (same group, or a
resolution date within a day of one of its questions), written to
backtest/data/heldout-<commit>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = "forecastingresearch/forecastbench-datasets"
MARKETS = ("manifold", "metaculus", "polymarket")
DEFAULT_FREEZE_FROM = "2026-08-01"  # Opus 5.5's stated training cutoff is June 2026, plus a buffer
DATA_DIR = Path(__file__).resolve().parent / "data"
TEMPLATE_CRITERIA = "Resolves to the outcome of the question found at"
MISSING = ("", "N/A", "n/a", "None")
URL = re.compile(r"https?://\S+")


def _get_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "backtest-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _present(text: object) -> bool:
    return isinstance(text, str) and text.strip() not in MISSING


def criteria_source(question: dict) -> str | None:
    """The field holding the market's real resolution rules. ForecastBench's
    `resolution_criteria` is only a template line pointing at the market page."""
    for field in ("market_info_resolution_criteria", "background"):
        text = question.get(field)
        if _present(text) and TEMPLATE_CRITERIA not in text:
            return field
    return None


def criteria_for(question: dict) -> str | None:
    field = criteria_source(question)
    return URL.sub("[link removed]", question[field].strip()) if field else None


def group_for(question: dict) -> str:
    """Questions that belong to one event share a group, for resampling.
    Polymarket multi-outcome markets share their background text."""
    background = question.get("background") or ""
    if question["source"] == "polymarket" and _present(background):
        digest = hashlib.sha256(background.strip()[:120].encode()).hexdigest()[:12]
        return f"polymarket:{digest}"
    return f"{question['source']}:{question['id']}"


def select(
    sets: list[tuple[str, dict, dict]], freeze_from: str
) -> tuple[list[dict], dict[str, int]]:
    """Join, filter and deduplicate. `sets` is (date, question set, resolution set)
    in date order. Returns the questions and counts of what was dropped."""
    chosen: dict[str, dict] = {}
    dropped = {"not_market": 0, "frozen_too_early": 0, "unresolved": 0, "not_0_or_1": 0,
               "no_criteria": 0, "repeat": 0, "no_resolution": 0}
    for set_date, question_set, resolution_set in sets:
        # Combination questions have a list of ids; only single market questions are used.
        resolutions = {
            (r["source"], r["id"]): r
            for r in resolution_set["resolutions"]
            if isinstance(r["id"], str) and r["source"] in MARKETS
        }
        for q in question_set["questions"]:
            if q["source"] not in MARKETS or not isinstance(q["id"], str):
                dropped["not_market"] += 1
                continue
            if q["freeze_datetime"][:10] < freeze_from:
                dropped["frozen_too_early"] += 1
                continue
            if q["id"] in chosen:
                dropped["repeat"] += 1  # keep the earliest freeze
                continue
            r = resolutions.get((q["source"], q["id"]))
            if r is None:
                dropped["no_resolution"] += 1
                continue
            # Unresolved markets carry the current price in resolved_to.
            if r.get("resolved") is not True:
                dropped["unresolved"] += 1
                continue
            if r["resolved_to"] not in (0, 1, 0.0, 1.0):
                dropped["not_0_or_1"] += 1
                continue
            field = criteria_source(q)
            criteria = criteria_for(q)
            if criteria is None:
                dropped["no_criteria"] += 1
                continue
            background = q.get("background") or ""
            chosen[q["id"]] = {
                "id": q["id"],
                "source": q["source"],
                "set": set_date,
                "group": group_for(q),
                "question": q["question"].strip(),
                "background": URL.sub("[link removed]", background.strip())
                if _present(background) and field != "background"
                else "",
                "criteria": criteria,
                "freeze_datetime": q["freeze_datetime"],
                "crowd": float(q["freeze_datetime_value"]),
                "resolution_date": r["resolution_date"],
                "outcome": int(r["resolved_to"]),
            }
    questions = sorted(chosen.values(), key=lambda x: hashlib.sha256(x["id"].encode()).hexdigest())
    return questions, dropped


def exclude_overlap(questions: list[dict], existing: list[dict], days: int = 1) -> tuple[list[dict], dict[str, int]]:
    """Drop questions in `existing` by id, by group, or by a resolution date
    within `days` of one of its questions, so no event is in both lists."""
    from datetime import date, timedelta

    ids = {q["id"] for q in existing}
    groups = {q["group"] for q in existing}
    near: set[str] = set()
    for q in existing:
        d = date.fromisoformat(q["resolution_date"][:10])
        near.update((d + timedelta(days=k)).isoformat() for k in range(-days, days + 1))
    kept, dropped = [], {"same_id": 0, "same_group": 0, "resolution_date_within_%d_day" % days: 0}
    for q in questions:
        if q["id"] in ids:
            dropped["same_id"] += 1
        elif q["group"] in groups:
            dropped["same_group"] += 1
        elif q["resolution_date"][:10] in near:
            dropped["resolution_date_within_%d_day" % days] += 1
        else:
            kept.append(q)
    return kept, dropped


def list_hash(questions: list[dict]) -> str:
    return hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", help="dataset commit SHA (default: current main)")
    parser.add_argument("--freeze-from", default=DEFAULT_FREEZE_FROM)
    parser.add_argument("--exclude-list", help="build a held-out list: leave out this list's questions and events")
    args = parser.parse_args()

    commit = args.commit or _get_json(f"https://api.github.com/repos/{REPO}/commits/main")["sha"]
    listing = _get_json(
        f"https://api.github.com/repos/{REPO}/contents/datasets/resolution_sets?ref={commit}"
    )
    dates = sorted(
        item["name"].split("_")[0]
        for item in listing  # type: ignore[union-attr]
        if item["name"].endswith("_resolution_set.json")
    )
    raw = f"https://raw.githubusercontent.com/{REPO}/{commit}/datasets"
    sets = []
    for date in dates:
        try:
            q = _get_json(f"{raw}/question_sets/{date}-llm.json")
        except Exception as exc:  # a resolution set without an LLM question set
            print(f"skip {date}: {exc}", file=sys.stderr)
            continue
        r = _get_json(f"{raw}/resolution_sets/{date}_resolution_set.json")
        sets.append((date, q, r))

    questions, dropped = select(sets, args.freeze_from)
    excluded_from = None
    if args.exclude_list:
        existing = json.loads(Path(args.exclude_list).read_text())
        questions, overlap = exclude_overlap(questions, existing["questions"])
        dropped = {**dropped, **{f"overlap_{k}": v for k, v in overlap.items()}}
        excluded_from = {"file": Path(args.exclude_list).name, "sha256": existing["sha256"],
                         "dataset_commit": existing["dataset_commit"]}
    DATA_DIR.mkdir(exist_ok=True)
    out = {
        "dataset": f"https://github.com/{REPO}",
        "dataset_commit": commit,
        "licence": "CC-BY-SA-4.0",
        "freeze_from": args.freeze_from,
        "sets": [d for d, _, _ in sets],
        "dropped": dropped,
        "sha256": list_hash(questions),
        **({"excluded_from": excluded_from} if excluded_from else {}),
        "questions": questions,
    }
    path = DATA_DIR / (f"heldout-{commit[:12]}.json" if excluded_from else "questions.json")
    if excluded_from and path.exists():
        sys.exit(f"{path} exists; held-out lists are never overwritten.")
    path.write_text(json.dumps(out, indent=1))
    by_source: dict[str, int] = {}
    for q in questions:
        by_source[q["source"]] = by_source.get(q["source"], 0) + 1
    groups = len({q["group"] for q in questions})
    by_freeze: dict[str, int] = {}
    for q in questions:
        by_freeze[q["freeze_datetime"][:10]] = by_freeze.get(q["freeze_datetime"][:10], 0) + 1
    print(f"{len(questions)} questions in {groups} groups {by_source}; dropped {dropped}")
    print(f"by freeze date: {dict(sorted(by_freeze.items()))}")
    print(f"dataset commit {commit}; list sha256 {out['sha256']}; wrote {path}")


if __name__ == "__main__":
    main()
