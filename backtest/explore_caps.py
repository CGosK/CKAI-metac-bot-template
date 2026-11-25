"""
Exploratory, no model calls: limits on the Wikipedia arm's saved forecasts (median of 3).

1. Cap extremes: clip to [c, 1 - c] for a few fixed c (not tuned; past bot tournaments found capping helped).
2. Keep current-value questions in [0.15, 0.85]: questions whose title matches VALUE_RULE (price, rate, count or
   ranking thresholds). The rule and the matched ids are written out with a sha256 before any model run uses
   them (round-3 plan, E2).

Paired gains against the uncapped arm with the scorer's cluster bootstrap.

    uv run python -m backtest.explore_caps
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics

from backtest import score

CAPS = (0.01, 0.02, 0.03, 0.05)
VALUE_RULE = re.compile(
    r"\b(price|close|closing|above|below|exceed|reach|hit|at least|more than|fewer than|less than|yield|rate|"
    r"index|cases|count|ranking|rank|top \d+|#1|number one|\$\d)",
    re.I,
)


def main() -> None:
    _, questions, preds = score.load(["wiki"])
    full = [q for q in questions if len(preds[q["id"]]) >= 3]
    groups = score.clusters(full)
    y = {q["id"]: int(q["outcome"]) for q in full}
    bot = {q["id"]: statistics.median(preds[q["id"]][:3]) for q in full}
    base = {i: score.brier(bot[i], y[i]) for i in bot}

    def gain(adjusted: dict[str, float], ids: list[str]) -> dict:
        diffs = {i: base[i] - score.brier(adjusted[i], y[i]) for i in ids}
        mean, lo, hi, p = score.group_bootstrap(diffs, {i: groups[i] for i in ids})
        return {"n": len(ids), "gain": mean, "ci95": [lo, hi], "p": p,
                "changed": sum(adjusted[i] != bot[i] for i in ids)}

    out: dict = {"n": len(full), "caps": {}}
    for c in CAPS:
        out["caps"][str(c)] = gain({i: min(max(p, c), 1 - c) for i, p in bot.items()}, list(bot))
    value_ids = sorted(q["id"] for q in full if VALUE_RULE.search(q["question"]))
    out["value_rule"] = VALUE_RULE.pattern
    out["value_ids_sha256"] = hashlib.sha256("\n".join(value_ids).encode()).hexdigest()
    out["value_ids"] = value_ids
    out["value_band_0.15_0.85"] = gain({i: min(max(bot[i], 0.15), 0.85) for i in value_ids}, value_ids)
    text = json.dumps(out, indent=1)
    (score.OUT / "explore-caps.json").write_text(text + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "value_ids"}, indent=1))


if __name__ == "__main__":
    main()
