"""
Round-5 X2, helper picks the series (benchmarks/PLAN-refine-round5.md, exploratory).

    uv run python backtest/round5_x2.py run --batch r5-x2      # 300 Haiku calls, subscription, rkit runner
    uv run python -m backtest.round5_x2 score --batch r5-x2    # no model calls

Haiku 5.5, no tools, sees the question title, criteria and T, and returns one series id from an allowlist or
`none`. The allowlist is the 268 fred and dbnomics series of the round-5 items, named as the D1/P0 data blocks
named them. Items: 300 of the round-5 items ranked by sha256("round5-x2:" + id). Added here, fixed before the
run: every third item (rank index % 3 == 2, 100 items) gets the allowlist without its gold series, so `none`
versus a wrong series can be measured. Validation of a pick (plan): in the allowlist, first row a year or more
before T, a row before T, and a word shared by the series name and the title.

Gate for live use (plan): 90% usable picks (gold in list: pick == gold and valid), and `none` rather than a
wrong series, read here as no validated wrong pick in either condition. Also reported: wrong picks before
validation, golds that fail validation (gold_fails_validation; 5 station names don't share a word with their
titles, e.g. GILLOT-AEROPORT vs Roland Garros Airport, capping the raw usable share at 97.5%) and the usable share
of the achievable items. Picks are re-parsed from the saved responses at score time; a partial or duplicated batch
is not scored. Resume needs the first launch's x2_fingerprint (script, plan, items, allowlist hashes). Writes
runs/<batch>/picks.jsonl and backtest/out/round5-x2-<batch>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import round5, score  # noqa: E402

N_ITEMS = 300
SEED = "round5-x2:"
NAME_SOURCES = ("r5-d1-ours", "r5-p0-ours")
STOP = {"daily", "weekly", "monthly", "value", "values", "rate", "rates", "index", "level", "degree", "celsius",
        "will", "have", "than", "higher", "increased", "compared", "what", "probability", "that", "from", "with"}
PROMPT = """Today is {t}. A forecasting question needs one data series whose values decide it.

Question: {title}
Resolution criteria: {criteria}

Series you may choose from (id: name):
{allowlist}

Pick the single series from this list whose values decide the question. If no series in the list is the one
the question is about, answer none. Don't pick a merely similar series (another station, country, tenor or
index). End with one line in exactly this form:
SERIES: <id from the list, or none>"""


def key(q: dict) -> str:
    return f"{q['source']}:{q['series_id']}"


def names() -> dict[str, str]:
    """Series key -> name, from the data-block text the round-5 A2 arms showed the forecaster. Names shared by
    two series (FRED daily and weekly twins) get the frequency from the block's note."""
    by_id = {q["id"]: q for q in round5.load_items(round5.DEFAULT_ITEMS)["questions"]}
    out, freq = {}, {}
    for batch in NAME_SOURCES:
        for line in (score.RUNS / batch / "research.jsonl").read_text().splitlines():
            row = json.loads(line)
            m = re.search(r"\n- (.*?) \((?:FRED|DBnomics)", row.get("text", ""))
            if m and row["id"] in by_id:
                out[key(by_id[row["id"]])] = m.group(1)
                note = re.search(r"Note: ([A-Za-z]+)", row["text"])
                freq[key(by_id[row["id"]])] = note.group(1) if note else ""
    count = Counter(out.values())
    return {k: (f"{v} ({freq[k]})" if count[v] > 1 and freq[k] else v) for k, v in out.items()}


def chosen_items() -> list[dict]:
    items = round5.load_items(round5.DEFAULT_ITEMS)["questions"]
    ranked = sorted(items, key=lambda q: hashlib.sha256((SEED + q["id"]).encode()).hexdigest())[:N_ITEMS]
    return [dict(q, gold_removed=(i % 3 == 2)) for i, q in enumerate(ranked)]


def allowlist_text(series: dict[str, str], drop: str | None) -> str:
    return "\n".join(f"{k}: {v}" for k, v in sorted(series.items()) if k != drop)


def parse_pick(text: str, allowed: set[str]) -> str:
    """The id on the last SERIES: line (any case, markdown stripped); 'none', an allowed id, or 'invalid'.
    A bare id without its source prefix maps to the one allowed id it ends, if exactly one."""
    lines = re.findall(r"^\s*series:\s*(\S+)", re.sub(r"[*`]", "", text or ""), re.IGNORECASE | re.MULTILINE)
    if not lines:
        return "invalid"
    pick = lines[-1].strip().rstrip(".,;")
    if pick.lower() == "none":
        return "none"
    if pick in allowed:
        return pick
    bare = [k for k in allowed if k.split(":", 1)[1] == pick]
    return bare[0] if len(bare) == 1 else "invalid"


def words(text: str) -> set[str]:
    # Drop accents only (Désirade -> Desirade); other non-ASCII marks such as en dashes still split words.
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")
    return {w for w in re.findall(r"[a-z]{4,}", plain.lower()) if w not in STOP}


def validate(pick: str, q: dict, series: dict[str, str], http) -> dict:
    import data_research as dr

    source, sid = pick.split(":", 1)
    t = datetime.fromisoformat(q["freeze_datetime"])
    out = {"in_list": pick in series, "overlap": bool(words(series.get(pick, "")) & words(q["question"]))}
    try:
        s = dr.series_for(source, sid, t, http)
        out["row_before_t"] = bool(s.rows) and s.rows[-1][0] < t.date()
        out["year_of_history"] = bool(s.rows) and s.rows[0][0] <= t.date() - timedelta(days=365)
    except dr.DataUnavailable as exc:
        out.update(row_before_t=False, year_of_history=False, error=dr.redact(str(exc))[:200])
    out["valid"] = all(out[k] for k in ("in_list", "overlap", "row_before_t", "year_of_history"))
    return out


async def run(batch: str, parallel: int) -> None:
    from backtest import run as bt

    batch_dir = score.RUNS / batch
    picks_path = batch_dir / "picks.jsonl"
    series = names()
    items = chosen_items()
    plan = bt.ROUND5_PLAN
    if not plan.exists():
        sys.exit(f"{plan} is missing; X2 launches record its hash.")
    marker = (f"x2_fingerprint: script {bt._sha256(Path(__file__))[:16]} plan {bt._sha256(plan)[:16]} "
              f"items {hashlib.sha256(json.dumps([q['id'] for q in items]).encode()).hexdigest()[:16]} "
              f"allowlist {hashlib.sha256(allowlist_text(series, None).encode()).hexdigest()[:16]}")
    done = set()
    if batch_dir.exists():
        launch = (batch_dir / "launch.txt").read_text() if (batch_dir / "launch.txt").exists() else ""
        first = re.search(r"^x2_fingerprint: .*$", launch, re.MULTILINE)
        if not first:
            sys.exit(f"{batch_dir} exists and is not an X2 batch; batches are never overwritten. Use a new name.")
        if first.group(0) != marker:
            sys.exit("script, plan, items or allowlist differ from this batch's first launch.")
        done = {json.loads(ln)["id"] for ln in picks_path.read_text().splitlines()} if picks_path.exists() else set()
    batch_dir.mkdir(parents=True, exist_ok=True)
    with open(batch_dir / "launch.txt", "a") as f:
        f.write("\n".join([
            f"command: {' '.join(sys.argv)}", f"helper: {bt.HELPER} (plain mode, no tools)",
            f"claude_code_version: {bt.subprocess.run(['claude', '--version'], capture_output=True, text=True).stdout.strip()}",
            f"git_head: {bt._git('rev-parse', 'HEAD')}", f"git_dirty: {bool(bt._git('status', '--porcelain'))}",
            f"utc_start: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"items_sha256: {round5.load_items(round5.DEFAULT_ITEMS)['sha256']}",
            f"items: {len(items)} (gold removed from the list for {sum(q['gold_removed'] for q in items)})",
            f"allowlist: {len(series)} series", f"plan: {plan.name}",
            f"plan_sha256: {bt._sha256(plan)}", f"script_sha256: {bt._sha256(Path(__file__))}", marker,
            "exploratory: yes", ""]))
    from claude_cli_llm import PLAIN, ClaudeCliLlm, get_backend

    with open(batch_dir / "launch.txt", "a") as f:
        try:
            f.write(f"call_logs: {get_backend().run_dir}\n")
        except Exception as exc:
            f.write(f"call_logs: unavailable ({exc})\n")
    helper = ClaudeCliLlm(bt.HELPER, mode=PLAIN, allowed_tries=2)
    gate = asyncio.Semaphore(parallel)

    async def one(q: dict) -> None:
        drop = key(q) if q["gold_removed"] else None
        allowed = {k for k in series if k != drop}
        prompt = PROMPT.format(t=q["freeze_datetime"][:10], title=q["question"], criteria=q["criteria"],
                               allowlist=allowlist_text(series, drop))
        async with gate:
            try:
                text = await helper.invoke(prompt)
            except Exception as exc:
                print(f"failed {q['id']}: {type(exc).__name__}", flush=True)
                return
        row = {"id": q["id"], "gold": key(q), "gold_removed": q["gold_removed"],
               "pick": parse_pick(text, allowed), "response": text,
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(picks_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"done {q['id']} {row['pick']}", flush=True)

    try:
        await asyncio.gather(*(one(q) for q in items if q["id"] not in done))
    finally:
        with open(batch_dir / "launch.txt", "a") as f:
            f.write(f"utc_end: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n")


def score_batch(batch: str) -> dict:
    import data_research as dr

    series = names()
    by_id = {q["id"]: q for q in chosen_items()}
    rows = [json.loads(ln) for ln in (score.RUNS / batch / "picks.jsonl").read_text().splitlines()]
    ids = [r["id"] for r in rows]
    want = Counter(q["gold_removed"] for q in by_id.values())
    have = Counter(r["gold_removed"] for r in rows)
    if len(ids) != len(set(ids)) or set(ids) != set(by_id) or have != want:
        sys.exit(f"{batch}: {len(set(ids))} unique of {len(by_id)} items ({dict(have)} vs {dict(want)}); "
                 "no score on a partial or duplicated batch")
    http = dr.PoliteHttp(cache=dr.CACHE_DIR)
    out = {"present": {"n": 0, "correct": 0, "usable": 0, "none": 0, "wrong": 0, "wrong_valid": 0, "invalid": 0,
                       "gold_fails_validation": 0},
           "removed": {"n": 0, "none": 0, "wrong": 0, "wrong_valid": 0, "invalid": 0}, "wrong_examples": []}
    for r in rows:
        q, part = by_id[r["id"]], out["removed" if r["gold_removed"] else "present"]
        part["n"] += 1
        drop = r["gold"] if r["gold_removed"] else None
        pick = parse_pick(r["response"], {k for k in series if k != drop})  # re-parsed from the saved response
        if not r["gold_removed"] and not validate(r["gold"], q, series, http)["valid"]:
            part["gold_fails_validation"] += 1  # a ceiling on the usable share, whatever the pick
        if pick in ("none", "invalid"):
            part[pick] += 1
            continue
        v = validate(pick, q, series, http)
        if pick == r["gold"]:
            part["correct"] += 1
            part["usable"] += v["valid"]
        else:
            part["wrong"] += 1
            part["wrong_valid"] += v["valid"]
            if len(out["wrong_examples"]) < 15:
                out["wrong_examples"].append({"title": q["question"][:160], "gold": r["gold"], "pick": pick,
                                              "pick_name": series.get(pick), "valid": v["valid"]})
    p = out["present"]
    out["usable_share"] = p["usable"] / p["n"] if p["n"] else None
    achievable = p["n"] - p["gold_fails_validation"]
    out["usable_share_of_achievable"] = p["usable"] / achievable if achievable else None
    out["wrong_before_validation"] = p["wrong"] + out["removed"]["wrong"]
    out["passes"] = bool(p["n"]) and out["usable_share"] >= 0.9 and (
        p["wrong_valid"] + out["removed"]["wrong_valid"] == 0)
    out["gate"] = ("usable share >= 0.9 (gold in list) and no validated wrong pick in either condition; "
                   "wrong picks before validation reported alongside, since validation does part of the work")
    out["requests"], out["cache_hits"] = http.requests, http.cache_hits
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["run", "score"])
    parser.add_argument("--batch", required=True)
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args()
    if args.step == "run":
        from backtest import run as bt
        from claude_cli_llm import refuse_in_ci

        refuse_in_ci()
        import ckai_bot  # noqa: F401  (loads .env, including RKIT_PATH)

        bt.configure_env()
        bt.block_metaculus()
        asyncio.run(run(args.batch, args.parallel))
    else:
        import dotenv

        dotenv.load_dotenv()  # FRED_API_KEY for validation fetches; never printed
        result = score_batch(args.batch)
        path = round5.write(f"x2-{args.batch}", result)
        print(json.dumps({k: v for k, v in result.items() if k != "wrong_examples"}, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
