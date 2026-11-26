"""
Round-5 D1 breakdowns (benchmarks/PLAN-refine-round5.md, "Also reported"), no model calls.

    uv run python -m backtest.round5_d1_report

Splits the A0 (r5-d1-off) vs A2 (r5-d1-ours) comparison by source, by horizon and by outcome with score.compare
(cluster-t, the same code as the primary test), and counts the A2 rationales that state the block's last value
and its date, read from the block text the forecaster saw (runs/r5-d1-ours/research.jsonl). The same check runs on
the A0 rationales as a control: A0 never saw the block. All splits are exploratory. Writes
backtest/out/round5-d1-report.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from backtest import round5, score

OFF, OURS = "r5-d1-off", "r5-d1-ours"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def date_forms(d: date) -> list[str]:
    m = MONTHS[d.month - 1]
    forms = [d.isoformat(), f"{m} {d.day}", f"{m[:3]} {d.day}", f"{d.day} {m}", f"{d.day} {m[:3]}",
             f"{d.month}/{d.day}/{d.year}"]
    return forms + ([f"Sept {d.day}", f"{d.day} Sept"] if d.month == 9 else [])


def value_forms(shown: str) -> list[str]:
    """The value as the block shows it, with a trailing zero added, or rounded (not truncated) to fewer places.
    Rounded forms keep two significant digits, so '4' for 3.65 doesn't count, and integers below 100 don't
    count ('18' for 18.04 matches ranges and percentages). Rounding is decimal half-up."""
    shown = shown.replace(",", "")
    places = len(shown.split(".")[1]) if "." in shown else 0
    forms = {shown, shown + ("0" if places else ".0")}
    for n in range(places):
        r = str(Decimal(shown).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))
        if len(r.lstrip("-0.").replace(".", "")) >= 2 and (n > 0 or abs(Decimal(shown)) >= 100):
            forms.add(r)
    return sorted(forms)


def states(text: str, d: date, shown: str) -> dict:
    flat = text.replace(",", "")
    has_date = any(re.search(rf"(?<!\d){re.escape(f)}(?!\d)", text, re.IGNORECASE) for f in date_forms(d))
    has_value = any(re.search(rf"(?<![\d.]){re.escape(f)}(?!\d)", flat) for f in value_forms(shown))
    return {"date": has_date, "value": has_value, "both": has_date and has_value}


LATEST = re.compile(r"latest (\S+) .*?on (\d{4}-\d{2}-\d{2}) \(")


def last_rows() -> dict[str, tuple[date, str]]:
    """Each A2 item's last (date, value as shown) from the block text the forecaster saw (research.jsonl)."""
    out = {}
    for line in (score.RUNS / OURS / "research.jsonl").read_text().splitlines():
        row = json.loads(line)
        m = LATEST.search(row.get("text", ""))
        if m:
            out[row["id"]] = (date.fromisoformat(m.group(2)), m.group(1))
    return out


def reasoning(batch: str) -> dict[str, str]:
    out = {}
    for line in (score.RUNS / batch / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("prediction") is not None:
            out[row["id"]] = row.get("reasoning") or ""
    return out


def summary(r: dict) -> dict:
    return {"questions": r["questions"], "clusters": r["clusters"], "gain": r["gain"],
            "mean_over_questions": r.get("mean_over_questions"), "log_gain": r["log_gain"],
            "brier": r["brier"]}


def main() -> None:
    score.QUESTIONS = round5.DEFAULT_ITEMS  # as score.py --dataset; load checks each launch's hash of this file
    _, questions, base = score.load([OFF])
    _, _, other = score.load([OURS])
    questions = [q for q in questions if q["id"] in other]

    def cmp(qs):
        return summary(score.compare(qs, base, other, 1, None, 1, test="cluster-t")) if qs else None

    result = {"note": "exploratory splits; primary result is compare-r5-d1-off-vs-r5-d1-ours-clustert.json",
              "all": cmp(questions), "by_source": {}, "by_horizon": {}, "by_outcome": {}}
    for s in sorted({q["source"] for q in questions}):
        result["by_source"][s] = cmp([q for q in questions if q["source"] == s])
    for h in sorted({q["horizon_days"] for q in questions}):
        result["by_horizon"][str(h)] = cmp([q for q in questions if q["horizon_days"] == h])
    for name, y in (("resolved_yes", 1), ("resolved_no", 0)):
        result["by_outcome"][name] = cmp([q for q in questions if q["outcome"] == y])

    shown = last_rows()
    text = {OURS: reasoning(OURS), OFF: reasoning(OFF)}
    counts = {OURS: Counter(), OFF: Counter()}
    no_block = 0
    for q in questions:
        if q["id"] not in shown:
            no_block += 1
            continue
        d, v = shown[q["id"]]
        for batch in (OURS, OFF):
            hit = states(text[batch].get(q["id"], ""), d, v)
            counts[batch]["items"] += 1
            counts[batch].update(k for k, ok in hit.items() if ok)
    result["rationale_states_last_row"] = {
        "rule": "approximate string rule. Date: ISO, 'Month D', 'Mon D', 'D Month', 'D Mon', M/D/YYYY or 'Sept D', with no digit "
                "next to it. Value: as the block shows it, with a trailing zero, or rounded half-up to fewer "
                "places keeping two significant digits (no integers below 100); commas ignored. A forecast number equal to the last value also "
                "counts.",
        "items_without_block_text": no_block,
        "A2_ours": {k: counts[OURS][k] for k in ("items", "date", "value", "both")},
        "A0_off_control": {k: counts[OFF][k] for k in ("items", "date", "value", "both")},
    }
    path = round5.write("d1-report", result)
    a = result["all"]
    print(f"all: n={a['questions']} clusters {a['clusters']} gain {a['gain']['mean']:+.4f} p {a['gain']['p']:.3f}")
    for part in ("by_source", "by_horizon", "by_outcome"):
        for k, r in result[part].items():
            print(f"{part} {k}: n={r['questions']} cl={r['clusters']} gain {r['gain']['mean']:+.4f} "
                  f"{[round(x, 4) for x in r['gain']['ci95']]} q-mean {r['mean_over_questions']:+.4f} "
                  f"log {r['log_gain']['mean']:+.4f}")
    print(json.dumps(result["rationale_states_last_row"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
