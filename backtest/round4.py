"""
Round-4 derived batches (benchmarks/PLAN-refine-round4.md).

    # X1: combine a batch's P(Yes) with a --reverse batch's P(No), forecast by forecast (no model calls)
    uv run python -m backtest.round4 combine --yes wiki --no r4-reverse --out r4-both

    # X2: reconcile related questions (same market group, same freeze date), one call per group and k
    uv run python -m backtest.round4 reconcile --batch wiki --out r4-reconciled

Each writes a new batch folder (never overwriting one) with its own launch.txt, so score.py --compare reads it
like any other batch. Reconcile resumes: groups already done for a k are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backtest import run

PREDICTIONS = 3


def _stops() -> tuple:
    from claude_cli_llm import ClaudeCliRefused

    return (run.MetaculusBlocked, ClaudeCliRefused)


STOPS: tuple = ()


def rows_by_key(batch: str) -> dict[tuple[str, int], dict]:
    """Saved predictions keyed by (id, n): the n-th non-null row for that id, the order score.load uses."""
    out, seen = {}, defaultdict(int)
    path = run.RUNS / batch / "predictions.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("prediction") is not None:
            out[(row["id"], seen[row["id"]])] = row
            seen[row["id"]] += 1
    return out


def reconciled_done(batch: str) -> set[tuple[str, int]]:
    """For reconcile resumes: (id, k) already written with a prediction (k is the source forecast's n)."""
    out = set()
    path = run.RUNS / batch / "predictions.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("prediction") is not None:
                out.add((row["id"], row["k"]))
    return out


def first_launch(batch: str) -> dict[str, str]:
    text = (run.RUNS / batch / "launch.txt").read_text().split("\n\n")[0]
    return dict(ln.split(": ", 1) for ln in text.splitlines() if ": " in ln)


def new_batch(name: str, resume: bool = False) -> Path:
    path = run.RUNS / name
    if path.exists() and not resume:
        sys.exit(f"{path} exists; batches are never overwritten.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_launch(path: Path, lines: dict) -> None:
    base = {
        "command": " ".join(sys.argv),
        "git_head": run._git("rev-parse", "HEAD"),
        "git_dirty": bool(run._git("status", "--porcelain")),
        "utc_start": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": run.ROUND4_PLAN.name,
        "plan_sha256": run._sha256(run.ROUND4_PLAN),
        "claude_code_version": run.subprocess.run(["claude", "--version"], capture_output=True,
                                                  text=True).stdout.strip(),
        "prompts_r4_sha256": run._sha256(run.ROOT / "prompts_r4.py"),
    }
    with open(path / "launch.txt", "a") as f:
        f.write("".join(f"{k}: {v}\n" for k, v in {**lines, **base}.items()) + "\n")


def combine(yes: str, no: str, out: str) -> None:
    from prompts_r4 import combine_both_directions

    ly, ln = first_launch(yes), first_launch(no)
    if ly["questions_sha256"] != ln["questions_sha256"]:
        sys.exit("the two batches used different question lists")
    if ln.get("reverse", "off") == "off" or ly.get("reverse", "off") != "off":
        sys.exit("--no must be a --reverse batch and --yes must not be")
    defaults = {"effort": f"{run.DEFAULT_EFFORT} (forecaster, claude --effort)", "prompt_version": "template",
                "prompt_sha256": "template (main.py)", "prompt_evidence": "off"}
    for key in ("forecaster", "effort", "prompt_version", "prompt_sha256", "prompt_evidence"):
        if ly.get(key, defaults.get(key)) != ln.get(key, defaults.get(key)):
            sys.exit(f"the two batches differ in {key}")
    same_research = ly.get("research") == ln.get("research") and ly.get("joined_from") == ln.get("joined_from")
    reuses_yes = ln.get("research") == "joined" and ln.get("joined_from", "").startswith(f"runs/{yes}/research.jsonl ")
    if not (same_research or reuses_yes):
        sys.exit("the reverse batch must use the same research as the yes batch (--research joined --sources YES)")
    ys, ns = rows_by_key(yes), rows_by_key(no)
    pairs = sorted(set(ys) & set(ns))
    odd = [key for key in pairs if abs(ys[key]["prediction"] + ns[key]["prediction"] - 1) > 0.5]
    path = new_batch(out)
    with open(path / "predictions.jsonl", "w") as f:
        for key in pairs:
            p = combine_both_directions(ys[key]["prediction"], ns[key]["prediction"])
            f.write(json.dumps({"id": key[0], "k": key[1], "prediction": p,
                                "p_yes": ys[key]["prediction"], "p_no": ns[key]["prediction"]}) + "\n")
    alone = new_batch(out + "-no-only")  # exploratory: 1 - P(No) alone, a batch of its own for --compare
    with open(alone / "predictions.jsonl", "w") as f:
        for key in pairs:
            f.write(json.dumps({"id": key[0], "k": key[1], "prediction": 1 - ns[key]["prediction"]}) + "\n")
    (alone / "question_ids.json").write_text(json.dumps(sorted({i for i, _ in pairs})))
    write_launch(alone, {"derived": f"X1 exploratory: 1 - P(No) from {no} alone",
                         "questions_sha256": ly["questions_sha256"], "forecaster": ln.get("forecaster")})
    ids = sorted({i for i, _ in pairs})
    (path / "question_ids.json").write_text(json.dumps(ids))
    write_launch(path, {"derived": f"X1 both directions: 0.5*(P(Yes) from {yes} + 1 - P(No) from {no}), "
                                   "paired by question and k",
                        "questions_sha256": ly["questions_sha256"], "forecaster": ly.get("forecaster"),
                        "pairs": len(pairs),
                        "pairs_far_from_complementary": f"{len(odd)} (|P(Yes)+P(No)-1| > 0.5): "
                                                        + ", ".join(f"{i}/{k}" for i, k in odd[:20]),
                        "claude_code_versions": f"yes {ly.get('claude_code_version')}; no {ln.get('claude_code_version')}"})
    print(f"wrote {len(pairs)} combined forecasts for {len(ids)} questions to {path}; "
          f"{len(odd)} pairs far from complementary")


def groups(questions: list[dict]) -> list[list[dict]]:
    """Related questions: same market group and same freeze (a later freeze could leak into an earlier one)."""
    by = defaultdict(list)
    for q in questions:
        by[(q["group"], q["freeze_datetime"])].append(q)
    return [v for _, v in sorted(by.items()) if len(v) > 1]


async def reconcile(batch: str, out: str, resume: bool, parallel: int) -> None:
    from claude_cli_llm import ClaudeCliLlm
    from prompts_r4 import parse_reconciled, reconcile_prompt

    meta = json.loads(run.QUESTIONS.read_text())
    lb = first_launch(batch)
    if lb["questions_sha256"] != meta["sha256"]:
        sys.exit("the batch used a different question list")
    if lb.get("reverse", "off") != "off":
        sys.exit("reconcile needs a P(Yes) batch, not a --reverse one")
    # The baseline's forecaster and effort, so reconciled forecasts differ from it only by reconciling.
    run.FORECASTER = lb["forecaster"]
    effort = lb.get("effort", run.DEFAULT_EFFORT).split(" ")[0]
    run.configure_env(effort)
    rows = rows_by_key(batch)
    sets = [g for g in groups(meta["questions"])
            if all((q["id"], k) in rows for q in g for k in range(PREDICTIONS))]
    path = new_batch(out, resume)
    derived = f"X2 reconcile related questions from {batch}, one call per group and k"
    if resume and first_launch(out).get("derived") != derived:
        sys.exit(f"{out} was not reconciled from {batch}")
    write_launch(path, {"derived": derived, "questions_sha256": meta["sha256"], "forecaster": run.FORECASTER,
                        "effort": f"{effort} (forecaster, claude --effort)",
                        "system_prompt": run.BACKTEST_SYSTEM_PROMPT, "groups": len(sets),
                        "questions": sum(len(g) for g in sets), "resume": resume})
    (path / "question_ids.json").write_text(json.dumps(sorted(q["id"] for g in sets for q in g)))
    done = reconciled_done(out) if resume else set()
    llm = ClaudeCliLlm(run.FORECASTER, allowed_tries=2)
    lock, slots = asyncio.Lock(), asyncio.Semaphore(parallel)

    async def one(g: list[dict], k: int) -> None:
        if all((q["id"], k) in done for q in g):
            return
        async with slots:
            today = g[0]["freeze_datetime"][:10]
            prompt = reconcile_prompt([{"id": q["id"], "question": q["question"], "criteria": q["criteria"],
                                        "forecast": rows[(q["id"], k)]["prediction"],
                                        "reasoning": rows[(q["id"], k)]["reasoning"]} for q in g], today)
            try:
                reply = await llm.invoke(prompt)
                new = parse_reconciled(reply, [q["id"] for q in g])
                out_rows = [{"id": i, "k": k, "prediction": p, "before": rows[(i, k)]["prediction"],
                             "reasoning": reply} for i, p in new.items()]
            except STOPS:
                raise
            except Exception as exc:  # recorded; retried on resume
                out_rows = [{"id": q["id"], "k": k, "prediction": None, "error": f"{type(exc).__name__}: {exc}"[:500]}
                            for q in g]
            async with lock:
                with open(path / "predictions.jsonl", "a") as f:
                    for r in out_rows:
                        f.write(json.dumps(r) + "\n")

    await asyncio.gather(*(one(g, k) for g in sets for k in range(PREDICTIONS)))
    print(f"reconciled {len(sets)} groups x {PREDICTIONS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--yes", required=True)
    c.add_argument("--no", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("reconcile")
    r.add_argument("--batch", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    if args.cmd == "combine":
        combine(args.yes, args.no, args.out)
        return
    from claude_cli_llm import refuse_in_ci

    refuse_in_ci()
    import ckai_bot  # noqa: F401  (loads .env, including RKIT_PATH)

    global STOPS
    STOPS = _stops()
    run.block_metaculus()
    asyncio.run(reconcile(args.batch, args.out, args.resume, args.parallel))


if __name__ == "__main__":
    main()
