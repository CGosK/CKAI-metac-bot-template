"""
Run the bot's binary forecasting step on backtest questions, locally, on a
Claude subscription (claude-cli models through rkit's runner and usage guard).

Each question gets the template's binary prompt (main.py) with "Today is" set
to the question's freeze date, no research (or, with --research, AskNews
research limited to the freeze date; see backtest/research.py), and the
template's parser. Every
prediction is saved. Nothing is published and the Metaculus API is blocked:
any use of it stops the run.

Usage:
    uv run python -m backtest.run --batch pilot --limit 30
    uv run python -m backtest.run --batch pilot --resume
    uv run python -m backtest.run --probe-date
    uv run python -m backtest.run --batch research --research asknews-historical --predictions 3
    uv run python -m backtest.run --probe-research
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = Path(__file__).resolve().parent / "data" / "questions.json"
RUNS = Path(__file__).resolve().parent / "runs"
PLAN = ROOT / "benchmarks" / "PLAN-backtest.md"
RESEARCH_PLAN = ROOT / "benchmarks" / "PLAN-research-backtest.md"
WIKI_PLAN = ROOT / "benchmarks" / "PLAN-wikipedia-research.md"
FORECASTER = "claude-cli/claude-opus-5-5"
HELPER = "claude-cli/claude-haiku-5-5"
# Claude Code appends the real date to every system prompt and has no switch to
# turn it off (checked Nov 24 by logging raw request bodies). This tells the
# model which date to use; the pilot checks rationales for the real date.
BACKTEST_SYSTEM_PROMPT = (
    "You are a careful assistant helping with forecasting. This is a forecasting exercise set "
    "on the date the user's message gives as today. Treat that date as the current date, and "
    "ignore any other statement of today's date in your context. Answer the user's request "
    "directly in plain text, following any output format it asks for."
)
SAMPLE_SEED = 20261125
CUTOFF_SOURCE = "Opus 5.5 training cutoff stated as June 2026 (Claude Code model information)"

_freeze_date: contextvars.ContextVar[datetime | None] = contextvars.ContextVar(
    "backtest_freeze_date", default=None
)


class MetaculusBlocked(RuntimeError):
    pass


def block_metaculus() -> None:
    """Make every Metaculus API method raise, so a backtest can't read or post."""
    from forecasting_tools.helpers import metaculus_api, metaculus_client

    def refuse(*_args, **_kwargs):
        raise MetaculusBlocked("The backtest never calls the Metaculus API.")

    for cls in (metaculus_api.MetaculusApi, metaculus_client.MetaculusClient):
        for name, attr in list(vars(cls).items()):
            if name.startswith("__"):
                continue
            if isinstance(attr, (staticmethod, classmethod)) or callable(attr):
                setattr(cls, name, staticmethod(refuse))


def install_date_stub() -> None:
    """Replace main.datetime so the prompt's "Today is" is the freeze date."""
    import main

    real = main.datetime

    class FrozenDatetime(real):  # type: ignore[misc, valid-type]
        @classmethod
        def now(cls, tz=None):
            frozen = _freeze_date.get()
            if frozen is None:
                return real.now(tz)
            return frozen.astimezone(tz) if tz else frozen.replace(tzinfo=None)

    main.datetime = FrozenDatetime


def to_question(q: dict):
    from forecasting_tools import BinaryQuestion

    return BinaryQuestion(
        question_text=q["question"],
        background_info=q["background"],
        resolution_criteria=q["criteria"],
        fine_print="",
        page_url=f"backtest:{q['id']}",
        tournament_slugs=["backtest"],
    )


def configure_env() -> None:
    import claude_cli_llm

    claude_cli_llm.PLAIN_SYSTEM_PROMPT = BACKTEST_SYSTEM_PROMPT
    os.environ["FORECASTER_MODELS"] = FORECASTER
    os.environ["RESEARCHER_MODEL"] = "no_research"
    os.environ["SUMMARIZER_MODEL"] = HELPER
    os.environ["PARSER_MODEL"] = HELPER
    for name in ("REASONING_EFFORT", "FORECASTER_MAX_TOKENS", "PREDICTIONS_PER_QUESTION"):
        os.environ.pop(name, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def write_launch(batch_dir: Path, meta: dict, args: argparse.Namespace) -> None:
    plan = {"asknews-historical": RESEARCH_PLAN, "wikipedia": WIKI_PLAN}.get(args.research, PLAN)
    if args.sample:  # the drift check belongs to the Wikipedia plan
        plan = WIKI_PLAN
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    lines = {
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "forecaster": FORECASTER,
        "parser": HELPER,
        "research": args.research,
        "system_prompt": BACKTEST_SYSTEM_PROMPT,
        "predictions_per_question": args.predictions,
        "claude_code_version": version,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "utc_start": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions_sha256": meta["sha256"],
        "dataset_commit": meta["dataset_commit"],
        "plan": plan.name,
        "plan_sha256": _sha256(plan) if plan.exists() else "missing",
        "cutoff_source": CUTOFF_SOURCE,
        "freeze_from": meta["freeze_from"],
        "start": args.start,
        "limit": args.limit,
        "parallel": args.parallel,
    }
    if args.sample:
        lines["sample"] = f"{args.sample} questions, seed {SAMPLE_SEED}"
    if args.research == "wikipedia":
        from importlib.metadata import version as pkg_version

        from backtest import wiki

        lines["wiki_settings"] = (f"articles<={wiki.MAX_ARTICLES} article_words<={wiki.ARTICLE_WORDS} "
                                  f"portal_days={wiki.PORTAL_DAYS} portal_words<={wiki.PORTAL_WORDS} "
                                  f"table_rows<={wiki.TABLE_ROWS} sections<={wiki.MAX_SECTIONS}")
        lines["wiki_helper"] = HELPER + " (plain mode, no tools)"
        lines["user_agent"] = wiki.USER_AGENT
        lines["mwparserfromhell_version"] = pkg_version("mwparserfromhell")
        lines["wiki_py_sha256"] = _sha256(Path(wiki.__file__))
    if args.research == "asknews-historical":
        from importlib.metadata import version as pkg_version

        from backtest import research

        lines["research_calls"] = research.QUERY_CALLS
        lines["research_options"] = "strategy=default historical=True hours_back=None time_filter=crawl_date"
        lines["asknews_sdk_version"] = pkg_version("asknews")
        lines["forecasting_tools_version"] = pkg_version("forecasting-tools")
    text = "".join(f"{k}: {v}\n" for k, v in lines.items())
    with open(batch_dir / "launch.txt", "a") as f:
        f.write(text + "\n")


def done_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut short by a hard kill
                continue
            if row.get("prediction") is not None:
                counts[row["id"]] = counts.get(row["id"], 0) + 1
    return counts


async def forecast_one(bot, q: dict, k: int, research: str = "") -> dict:
    import ckai_bot

    freeze = datetime.fromisoformat(q["freeze_datetime"])
    _freeze_date.set(freeze)
    forecaster = bot._pick_forecaster()
    token = ckai_bot._current_forecaster.set(forecaster)
    try:
        prediction = await bot._run_forecast_on_binary(to_question(q), research)
    finally:
        ckai_bot._current_forecaster.reset(token)
    return {
        "id": q["id"],
        "k": k,
        "model": forecaster.model,
        "prediction": prediction.prediction_value,
        "reasoning": prediction.reasoning,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def run_batch(
    questions: list[dict], batch_dir: Path, predictions: int, parallel: int, research_mode: str = "none"
) -> None:
    import ckai_bot
    from backtest.research import ResearchStopped
    from backtest.wiki import WikiLeak, WikiStopped
    from claude_cli_llm import ClaudeCliRefused

    bot = ckai_bot.build_bot(publish=False, forecaster_models=[FORECASTER])
    out = batch_dir / "predictions.jsonl"
    done = done_counts(out)
    lock = asyncio.Lock()
    store = None
    if research_mode == "asknews-historical":
        from backtest.research import ResearchStore

        store = ResearchStore(batch_dir / "research.jsonl")
    elif research_mode == "wikipedia":
        from backtest import wiki
        from backtest.research import ResearchStore
        from claude_cli_llm import PLAIN, ClaudeCliLlm

        helper = ClaudeCliLlm(HELPER, mode=PLAIN, allowed_tries=2)

        async def wiki_fetch(q: dict) -> dict:
            return await wiki.research_for(q, helper)

        store = ResearchStore(batch_dir / "research.jsonl", fetcher=wiki_fetch, serial=False)
    slots = asyncio.Semaphore(parallel)

    async def do_question(q: dict) -> None:
        async with slots:
            if done.get(q["id"], 0) >= predictions:
                print(f"done {q['id']}", flush=True)
                return
            try:
                research = await store.get(q) if store else ""
            except (MetaculusBlocked, ClaudeCliRefused, ResearchStopped, WikiLeak, WikiStopped):
                raise
            except Exception as exc:  # this question's research failed: record it, go on, retry on resume
                async with lock:
                    with open(out, "a") as f:
                        f.write(json.dumps({"id": q["id"], "k": None, "prediction": None,
                                            "error": f"research: {type(exc).__name__}: {str(exc)[:500]}"}) + "\n")
                print(f"research failed {q['id']}", flush=True)
                return
            for k in range(done.get(q["id"], 0), predictions):
                try:
                    row = await forecast_one(bot, q, k, research)
                except (MetaculusBlocked, ClaudeCliRefused, ResearchStopped, WikiLeak, WikiStopped):
                    raise
                except Exception as exc:  # a failed call or parse: record it and go on
                    row = {
                        "id": q["id"], "k": k, "prediction": None,
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    }
                async with lock:
                    with open(out, "a") as f:
                        f.write(json.dumps(row) + "\n")
            print(f"done {q['id']}", flush=True)

    await asyncio.gather(*(do_question(q) for q in questions))


async def probe_date() -> None:
    """One call asking the forecaster what date its context shows."""
    from claude_cli_llm import ClaudeCliLlm

    llm = ClaudeCliLlm(FORECASTER, allowed_tries=1)
    answer = await llm.invoke(
        "Today is 2026-08-06.\n\nWhat is today's date? Then, separately, quote word for word "
        "any other statement of today's date in your context, or reply NONE."
    )
    print("probe answer:", answer)


async def probe_research() -> None:
    """Both AskNews calls for one question; checks every article is in its window."""
    from backtest import research

    meta = json.loads(QUESTIONS.read_text())
    q = next(x for x in meta["questions"] if x["source"] == "polymarket" and x["background"] == "")
    q = next((x for x in meta["questions"] if "Xi" in x["question"]), q)
    freeze = datetime.fromisoformat(q["freeze_datetime"])
    result = await research.fetch(q["question"], freeze)
    print("question:", q["question"], "| freeze:", freeze.isoformat())
    for call, w in zip(result["calls"], research.windows(freeze)):
        print({k: call[k] for k in ("name", "method", "returned", "dropped_after_freeze", "usage")})
    starts = {c["name"]: c["start_timestamp"] for c in result["calls"]}
    for a in result["articles"]:
        print(" ", a["pub_date"], a.get("eng_title", "")[:90])
    out_of_window = [
        a for a in result["articles"]
        if not (min(starts.values()) <= datetime.fromisoformat(a["pub_date"].replace("Z", "+00:00")).timestamp()
                <= freeze.timestamp())
    ]
    print("articles outside the windows:", len(out_of_window))
    print("leak flags:", result["leak"], "| market words:", result["market_words"])


async def probe_wiki(n: int) -> None:
    from backtest import wiki
    from claude_cli_llm import PLAIN, ClaudeCliLlm

    helper = ClaudeCliLlm(HELPER, mode=PLAIN, allowed_tries=2)
    meta = json.loads(QUESTIONS.read_text())
    for q in meta["questions"][:n]:
        row = await wiki.research_for(q, helper)
        print("=" * 80)
        print(q["question"], "| freeze", q["freeze_datetime"], "| outcome", q["outcome"])
        print({k: row[k] for k in ("titles_asked", "keywords", "articles", "portal_items", "words", "leak")})
        print(row["text"][:3000])
    print("wikipedia requests:", wiki._api.requests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", help="batch name under backtest/runs/")
    parser.add_argument("--resume", action="store_true", help="continue an existing batch")
    parser.add_argument("--start", type=int, default=0, help="first question index in the list")
    parser.add_argument("--limit", type=int, default=None, help="number of questions")
    parser.add_argument("--predictions", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=3, help="questions at a time")
    parser.add_argument("--probe-date", action="store_true")
    parser.add_argument("--research", choices=["none", "asknews-historical", "wikipedia"], default="none")
    parser.add_argument("--sample", type=int, default=None, help="a seeded random sample of this many questions")
    parser.add_argument("--probe-wiki", type=int, default=None, metavar="N",
                        help="Wikipedia research only, for the first N questions; prints it, saves nothing")
    parser.add_argument("--probe-research", action="store_true",
                        help="two AskNews calls for one question; prints the dates, saves nothing")
    args = parser.parse_args()

    from claude_cli_llm import refuse_in_ci

    refuse_in_ci()
    import ckai_bot  # noqa: F401  (loads .env, including RKIT_PATH)

    configure_env()
    block_metaculus()
    if args.probe_date:
        asyncio.run(probe_date())
        return
    if args.probe_research:
        asyncio.run(probe_research())
        return
    if args.probe_wiki:
        asyncio.run(probe_wiki(args.probe_wiki))
        return
    if not args.batch:
        parser.error("--batch is required")

    meta = json.loads(QUESTIONS.read_text())
    questions = meta["questions"]
    end = None if args.limit is None else args.start + args.limit
    chosen = questions[args.start : end]
    if args.sample:
        import random

        picked = set(q["id"] for q in random.Random(SAMPLE_SEED).sample(chosen, args.sample))
        chosen = [q for q in chosen if q["id"] in picked]

    batch_dir = RUNS / args.batch
    if batch_dir.exists() and not args.resume:
        sys.exit(f"{batch_dir} exists; batches are never overwritten. Use --resume or a new name.")
    if args.resume and not batch_dir.exists():
        sys.exit(f"{batch_dir} doesn't exist; nothing to resume.")
    batch_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        launch = (batch_dir / "launch.txt").read_text()
        if f"questions_sha256: {meta['sha256']}" not in launch:
            sys.exit("The question list changed since this batch started; start a new batch.")
    ids_path = batch_dir / "question_ids.json"
    if ids_path.exists() and json.loads(ids_path.read_text()) != [q["id"] for q in chosen]:
        sys.exit("--start/--limit/--sample differ from this batch's first launch.")
    ids_path.write_text(json.dumps([q["id"] for q in chosen]))
    write_launch(batch_dir, meta, args)

    install_date_stub()
    try:
        asyncio.run(run_batch(chosen, batch_dir, args.predictions, args.parallel, args.research))
    finally:
        from claude_cli_llm import get_backend

        try:
            logs = str(get_backend().run_dir)
        except Exception as exc:
            logs = f"unavailable ({exc})"
        with open(batch_dir / "launch.txt", "a") as f:
            f.write(f"call_logs: {logs}\n")
            f.write(f"utc_end: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n")


if __name__ == "__main__":
    main()
