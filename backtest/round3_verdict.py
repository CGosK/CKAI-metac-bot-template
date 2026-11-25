"""
Round-3 verdict (benchmarks/PLAN-refine-round3.md), no model calls.

    uv run python -m backtest.round3_verdict

Recomputes the three primary comparisons from the saved batches with score.compare (the same code and seed as
score.py), applies Holm over E1-E3, checks the log-score condition, and adds the plan's exploratory splits:
the non-election questions (election rule: explore_errors.TOPICS), and for E2 election and re-sampled market
group questions. Writes backtest/out/round3-verdict.json.

No decision is issued while any batch is partial (fewer than 325 questions with 3 predictions, or no utc_end in
its launch.txt): the output then says PARTIAL and gives numbers only. E2's 325 result is supporting evidence;
the plan's decision for E2 is the held-out harm check. Run backtest.model_check before reporting E3.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict

from backtest import score

BASELINE = "wiki"
K = 3
TOTAL = 325
TESTS = {
    "E1_effort_max": {"compare": "r3-effort-max"},
    "E2_prompt_v2": {"compare": "r3-v2"},
    "E3_three_models": {"ensemble": ["wiki", "r3-sonnet", "r3-fable"]},
}
EXPLORATORY_SINGLE = ["r3-sonnet", "r3-fable"]


def election_rule():
    """explore_errors.topic(q) == 'elections', taken from that file's source without running it (running it
    rewrites out/errors-wiki.json)."""
    tree = ast.parse((score.HERE / "explore_errors.py").read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "TOPICS"
                                                                       for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "topic")]
    space = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "explore_errors.py", "exec"), space)
    return lambda q: space["topic"](q) == "elections"


def resampled_groups(questions: list[dict]) -> set[str]:
    """Exploratory stand-in for the plan's "recurring-series" split (the plan names it without a rule): a market
    group sampled at two or more freeze dates in this set. Mostly elections; the output reports the overlap."""
    freezes = defaultdict(set)
    for q in questions:
        freezes[q["group"]].add(q["freeze_datetime"])
    return {q["id"] for q in questions if len(freezes[q["group"]]) > 1}


def compared_preds(spec: dict) -> tuple[dict, int]:
    if "compare" in spec:
        return score.load([spec["compare"]])[2], K
    batch_preds = [score.load([b])[2] for b in spec["ensemble"]]
    return score.ensemble(batch_preds, K), len(spec["ensemble"])


def summary(result: dict) -> dict:
    return {key: result[key] for key in ("questions", "clusters", "brier", "gain", "log_gain", "mean_forecast",
                                         "by_outcome") if key in result}


def batch_state(batch: str) -> dict:
    preds = score.load([batch])[2]
    n = sum(len(v) >= K for v in preds.values())
    finished = "utc_end:" in (score.RUNS / batch / "launch.txt").read_text()
    # Two Sonnet questions failed parsing on every retry; a finished batch may be short by a few.
    return {"questions_with_3": n, "of": TOTAL, "finished": finished, "partial": not finished}


def main() -> None:
    meta, questions, base = score.load([BASELINE])
    is_election = election_rule()
    resampled = resampled_groups(meta["questions"])
    result = {"baseline": f"{BASELINE}, expected median of {K} of its {K} saved predictions", "tests": {}}
    pvalues = {}
    for name, spec in TESTS.items():
        other, k = compared_preds(spec)
        full = score.compare(questions, base, other, k, None, K)
        pvalues[name] = full["gain"]["p"]
        states = {b: batch_state(b) for b in spec.get("ensemble", [spec.get("compare")])}
        entry = {"spec": spec, "batches": states, "partial": any(s["partial"] for s in states.values()),
                 "all": summary(full)}
        entry["exploratory_non_election"] = summary(
            score.compare([q for q in questions if not is_election(q)], base, other, k, None, K))
        if name.startswith("E2"):
            entry["exploratory_election"] = summary(
                score.compare([q for q in questions if is_election(q)], base, other, k, None, K))
            entry["exploratory_resampled_group"] = summary(
                score.compare([q for q in questions if q["id"] in resampled], base, other, k, None, K))
            entry["exploratory_not_resampled_group"] = summary(
                score.compare([q for q in questions if q["id"] not in resampled], base, other, k, None, K))
        result["tests"][name] = entry
    partial = any(e["partial"] for e in result["tests"].values())
    result["partial"] = partial
    adjusted = score.holm(pvalues)
    for name, entry in result["tests"].items():
        gain, log_gain = entry["all"]["gain"]["mean"], entry["all"]["log_gain"]["mean"]
        entry["holm_p"] = adjusted[name]
        entry["passes_brier"] = gain >= 0.01 and adjusted[name] < 0.05
        entry["log_not_worse"] = log_gain >= 0
        if partial:
            entry["decision"] = "PARTIAL: no decision until every batch is finished"
        elif name.startswith("E2"):
            entry["decision"] = ("supporting evidence only ("
                                 + ("passes" if entry["passes_brier"] and entry["log_not_worse"] else "fails")
                                 + " the 325 rule); decision = held-out harm check")
        else:
            entry["decision"] = "adopt" if entry["passes_brier"] and entry["log_not_worse"] else "not adopted"
    if partial:
        result["E1_decision"] = "PARTIAL: no decision"
    else:
        e1 = result["tests"]["E1_effort_max"]
        result["E1_decision"] = ("max" if e1["decision"] == "adopt"
                                 else "highest affordable OpenRouter level (point gain >= -0.005)"
                                 if e1["all"]["gain"]["mean"] >= -0.005 else "high")
    # Exploratory single models, on the questions every single-model batch covers, so they compare.
    singles = {b: score.load([b])[2] for b in EXPLORATORY_SINGLE}
    common = [q for q in questions if all(len(p.get(q["id"], [])) >= K for p in singles.values())]
    result["exploratory_single_models"] = {
        b: summary(score.compare(common, base, p, K, None, K)) for b, p in singles.items()}
    result["definitions"] = {
        "election": "explore_errors.topic(q) == 'elections' (first matching keyword list on the title)",
        "resampled_group": "market group sampled at two or more freeze dates in questions.json (exploratory)",
        "resampled_group_questions": len(resampled),
        "resampled_group_also_election": sum(1 for q in meta["questions"]
                                             if q["id"] in resampled and is_election(q)),
    }
    score.OUT.mkdir(exist_ok=True)
    path = score.OUT / "round3-verdict.json"
    path.write_text(json.dumps(result, indent=1))
    for name, entry in result["tests"].items():
        a = entry["all"]
        done = ", ".join(f"{b} {s['questions_with_3']}/{TOTAL}{'' if s['finished'] else ' RUNNING'}"
                         for b, s in entry["batches"].items())
        print(f"{name}: n={a['questions']} gain {a['gain']['mean']:+.4f} {[round(x, 4) for x in a['gain']['ci95']]} "
              f"holm p {entry['holm_p']:.3f} log {a['log_gain']['mean']:+.4f} | {entry['decision']} | {done}")
    print(f"E1 decision: {result['E1_decision']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
