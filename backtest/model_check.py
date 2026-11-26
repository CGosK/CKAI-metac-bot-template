"""
Which model answered (benchmarks/PLAN-refine-round3.md, E3), no model calls.

    uv run python -m backtest.model_check wiki r3-sonnet r3-fable

For each batch, reads the call-log folders named on its launch.txt `call_logs:` lines and sums output tokens per
model over the forecaster's calls (files named after the forecaster; the Haiku parser's files separately). Lists
forecaster call files with no result line or with tokens from another model. Writes
backtest/out/model-check-<batches>.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from backtest import score


def check(batch: str) -> dict:
    launch = (score.RUNS / batch / "launch.txt").read_text()
    first = dict(ln.split(": ", 1) for ln in launch.split("\n\n")[0].splitlines() if ": " in ln)
    model = first["forecaster"].split("/")[-1]
    dirs = [Path(ln.split(": ", 1)[1].strip()) for ln in launch.splitlines() if ln.startswith("call_logs: ")]
    tokens, other_tokens = Counter(), Counter()
    files = no_result = 0
    substituted = []
    for d in dirs:
        for f in sorted((d / "calls").glob("*.jsonl")):
            forecaster = f.name.endswith(f"-{model}.jsonl")
            here, results = Counter(), 0
            for line in f.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("type") == "result":
                    results += 1
                    for m, u in (row.get("model_usage_since_last_result") or {}).items():
                        here[m] += u.get("outputTokens", 0)
            if forecaster:
                files += 1
                tokens.update(here)
                no_result += results == 0
                if any(m != model and n for m, n in here.items()):
                    substituted.append(f.name)
            else:
                other_tokens.update(here)
    return {"forecaster": model, "call_log_dirs": len(dirs), "forecaster_files": files,
            "forecaster_output_tokens": dict(tokens), "files_without_result": no_result,
            "files_with_another_model": substituted, "other_calls_output_tokens": dict(other_tokens)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("batches", nargs="+")
    args = parser.parse_args()
    result = {b: check(b) for b in args.batches}
    score.OUT.mkdir(exist_ok=True)
    path = score.OUT / f"model-check-{'+'.join(args.batches)}.json"
    path.write_text(json.dumps(result, indent=1))
    for b, r in result.items():
        print(f"{b}: {r['forecaster_files']} forecaster calls, tokens {r['forecaster_output_tokens']}, "
              f"no result {r['files_without_result']}, another model {len(r['files_with_another_model'])}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
