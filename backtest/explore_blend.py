"""
Exploratory, no model calls: blend the Wikipedia arm's forecast with the market price at the freeze.

p_blend = w * bot + (1 - w) * crowd, for a few fixed weights. The bot is the median of the wiki arm's
first 3 predictions. Paired gains are against the crowd alone and against the bot alone, with the
scorer's cluster bootstrap. The weights were not tuned; 0.5 is the one stated in advance (Halawi et al.
2024 averaged with the crowd), the others are shown for shape only.

    uv run python -m backtest.explore_blend
"""

from __future__ import annotations

import json
import statistics

from backtest import score

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    _, questions, preds = score.load(["wiki"])
    full = [q for q in questions if len(preds[q["id"]]) >= 3]
    groups = score.clusters(full)
    y = {q["id"]: int(q["outcome"]) for q in full}
    bot = {q["id"]: statistics.median(preds[q["id"]][:3]) for q in full}
    crowd = {q["id"]: float(q["crowd"]) for q in full}
    out = {"n": len(full), "clusters": len(set(groups.values())), "weights": {}}
    for w in WEIGHTS:
        blend = {i: w * bot[i] + (1 - w) * crowd[i] for i in bot}
        b = {i: score.brier(blend[i], y[i]) for i in blend}
        row = {"brier": statistics.fmean(b.values())}
        for name, ref in (("vs_crowd", crowd), ("vs_bot", bot)):
            diffs = {i: score.brier(ref[i], y[i]) - b[i] for i in b}
            mean, lo, hi, p = score.group_bootstrap(diffs, groups)
            row[name] = {"gain": mean, "ci95": [lo, hi], "p": p}
        out["weights"][str(w)] = row
    text = json.dumps(out, indent=1)
    (score.OUT / "explore-blend.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
