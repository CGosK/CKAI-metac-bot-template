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
LIVE_WIKI_PLAN = ROOT / "benchmarks" / "PLAN-live-wikipedia.md"
COMBO_IDS = ROOT / "benchmarks" / "combo-ids.json"
COMBO_IDS_SHA256 = "30a26a105ea5640472d0bd478c29176cc48ced47643cd66c65ee1bfeab239b25"
COMBO_SOURCES = ("research2", "wiki")  # stored AskNews and Wikipedia research
ROUND2_PLAN = ROOT / "benchmarks" / "PLAN-refine-round2.md"
ROUND2_AMEND = ROOT / "benchmarks" / "PLAN-refine-round2-amend1.md"
ROUND3_PLAN = ROOT / "benchmarks" / "PLAN-refine-round3.md"
ROUND4_PLAN = ROOT / "benchmarks" / "PLAN-refine-round4.md"
ROUND5_PLAN = ROOT / "benchmarks" / "PLAN-refine-round5.md"
DATA_BLOCKS = ("off", "fb", "ours")  # round 5: A0 no value, A1 ForecastBench's value, A2 our dated block
# Forecasters the backtest may use, with their stated training cutoffs. Freezes start Aug 1, 2026.
FORECASTERS = {
    "claude-cli/claude-opus-5-5": "Opus 5.5 training cutoff stated as June 2026 (Claude Code model information)",
    "claude-cli/claude-sonnet-5-5": "Sonnet 5.5 training cutoff June 2026 (Anthropic models overview, read Nov 25, 2026)",
    "claude-cli/claude-fable-5-2": "Fable 5.2 training cutoff June 2026 (Anthropic models overview, read Nov 25, 2026)",
}
DEFAULT_EFFORT = "high"  # what the Claude Code settings file gave every batch before round 3
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
PILOT_SEED = "round5-pilot"


def pilot_ids(items: list[dict], n: int) -> list[str]:
    """Round 5 P0: n items spread across clusters. Clusters and the items within each are ordered by a seeded
    hash, then taken round-robin, so every cluster gets one before any gets two."""
    import hashlib

    def h(x: str) -> str:
        return hashlib.sha256(f"{PILOT_SEED}:{x}".encode()).hexdigest()

    by: dict[str, list[str]] = {}
    for q in items:
        by.setdefault(q["cluster"], []).append(q["id"])
    queues = [sorted(ids, key=h) for _, ids in sorted(by.items(), key=lambda kv: h(kv[0]))]
    out: list[str] = []
    while len(out) < n and any(queues):
        for qu in queues:
            if qu and len(out) < n:
                out.append(qu.pop(0))
    if len(out) < n:
        raise ValueError(f"only {len(out)} items for a pilot of {n}")
    return out

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


def configure_env(effort: str = DEFAULT_EFFORT, prompt: str = "template", evidence: bool = False) -> None:
    import claude_cli_llm

    claude_cli_llm.PLAIN_SYSTEM_PROMPT = BACKTEST_SYSTEM_PROMPT
    # Effort is set for the forecaster only; helpers keep the settings file's level.
    claude_cli_llm.PLAIN_EFFORT = {FORECASTER.removeprefix("claude-cli/"): effort}
    if effort in ("xhigh", "max"):
        claude_cli_llm.PLAIN_TIMEOUT = 25 * 60
    os.environ["FORECASTER_MODELS"] = FORECASTER
    os.environ["RESEARCHER_MODEL"] = "no_research"
    os.environ["SUMMARIZER_MODEL"] = HELPER
    os.environ["PARSER_MODEL"] = HELPER
    os.environ["PROMPT_VERSION"] = prompt
    os.environ["PROMPT_EVIDENCE"] = "on" if evidence else "off"
    for name in ("REASONING_EFFORT", "FORECASTER_MAX_TOKENS", "PREDICTIONS_PER_QUESTION"):
        os.environ.pop(name, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def write_launch(batch_dir: Path, meta: dict, args: argparse.Namespace) -> None:
    plan = {"asknews-historical": RESEARCH_PLAN, "wikipedia": WIKI_PLAN,
            "combined": LIVE_WIKI_PLAN, "gdelt": ROUND2_PLAN, "joined": ROUND2_PLAN}.get(args.research, PLAN)
    if args.dataset or args.freeze_on:
        plan = ROUND2_PLAN
    if args.sample and args.research == "none" and not (args.dataset or args.freeze_on):  # first drift check
        plan = WIKI_PLAN
    if round3(args):
        plan = ROUND3_PLAN
    if args.evidence or args.reverse:
        plan = ROUND4_PLAN
    if args.data_block:
        plan = ROUND5_PLAN
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
        "dataset": QUESTIONS.name,
        "questions_sha256": meta["sha256"],
        "dataset_commit": meta["dataset_commit"],
        "plan": plan.name,
        "plan_sha256": _sha256(plan) if plan.exists() else "missing",
        "cutoff_source": FORECASTERS[FORECASTER],
        "effort": f"{args.effort} (forecaster, claude --effort)",
        "prompt_version": args.prompt,
        "prompt_sha256": _sha256(ROOT / "prompts_v2.py") if args.prompt == "v2" else "template (main.py)",
        "plain_timeout_s": __import__("claude_cli_llm").PLAIN_TIMEOUT,
        "prompt_evidence": "on" if args.evidence else "off",
        "reverse": "P(No) asked (round 4, X1)" if args.reverse else "off",
        "prompts_r4_sha256": _sha256(ROOT / "prompts_r4.py") if (args.evidence or args.reverse) else "unused",
        "freeze_from": meta["freeze_from"],
        "data_block": args.data_block or "unused",
        "data_research_sha256": _sha256(ROOT / "data_research.py") if args.data_block else "unused",
        "start": args.start,
        "limit": args.limit,
        "parallel": args.parallel,
    }
    if args.sample:
        lines["sample"] = f"{args.sample} questions, seed {SAMPLE_SEED}"
    if args.pilot:
        lines["pilot"] = f"{args.pilot} items across clusters, seed {PILOT_SEED} (round 5 P0)"
    if args.exclude_pilot:
        lines["exclude_pilot"] = f"all items but the {args.exclude_pilot}-item pilot, seed {PILOT_SEED} (round 5 D1)"
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
    if args.freeze_on:
        lines["freeze_on"] = args.freeze_on
    if args.research == "gdelt":
        import gdelt_research

        lines["plan_amendment"] = ROUND2_AMEND.name
        lines["plan_amendment_sha256"] = _sha256(ROUND2_AMEND) if ROUND2_AMEND.exists() else "missing"
        lines["gdelt_lookback_days"] = args.lookback_days
        lines["gdelt_settings"] = (f"queries<={gdelt_research.MAX_QUERIES} sub_windows={gdelt_research.SUB_WINDOWS} "
                                   f"per_window={gdelt_research.PER_WINDOW} kept<={gdelt_research.MAX_HEADLINES} "
                                   f"gap={gdelt_research.GAP_SECONDS}s")
        lines["gdelt_helper"] = HELPER + " (plain mode, no tools)"
        lines["user_agent"] = gdelt_research.USER_AGENT
        lines["gdelt_research_py_sha256"] = _sha256(Path(gdelt_research.__file__))
    if args.research == "joined":
        lines["joined_from"] = ", ".join(
            f"runs/{b}/research.jsonl (sha256 {_sha256(RUNS / b / 'research.jsonl')})" for b in args.sources.split(","))
        lines["joined_with"] = "wiki_research.join_research, texts that found nothing left out"
    if args.research == "combined":
        lines["combined_from"] = ", ".join(f"runs/{b}/research.jsonl" for b in COMBO_SOURCES)
        lines["combined_join"] = "wiki_research.join_research (the live bot's join)"
        lines["combo_ids_sha256"] = COMBO_IDS_SHA256
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


def round3(args: argparse.Namespace) -> bool:
    return (args.effort != DEFAULT_EFFORT or args.prompt != "template"
            or args.forecaster != "claude-cli/claude-opus-5-5")


def check_resume_settings(first_launch: str, args: argparse.Namespace) -> str | None:
    """The reason a resume would change this batch's forecaster, effort or prompt, or None.
    Batches from before round 3 have no effort or prompt line: they ran at high with the template."""
    lines = dict(ln.split(": ", 1) for ln in first_launch.splitlines() if ": " in ln)
    if lines.get("forecaster", args.forecaster) != args.forecaster:
        return "--forecaster differs from this batch's first launch."
    if lines.get("effort", DEFAULT_EFFORT).split(" ")[0] != args.effort:
        return "--effort differs from this batch's first launch."
    if lines.get("prompt_version", "template") != args.prompt:
        return "--prompt differs from this batch's first launch."
    if args.prompt == "v2" and lines.get("prompt_sha256") != _sha256(ROOT / "prompts_v2.py"):
        return "prompts_v2.py changed since this batch's first launch."
    if lines.get("prompt_evidence", "off") != ("on" if args.evidence else "off"):
        return "--evidence differs from this batch's first launch."
    if (lines.get("reverse", "off") != "off") != args.reverse:
        return "--reverse differs from this batch's first launch."
    if (args.evidence or args.reverse) and lines.get("prompts_r4_sha256") != _sha256(ROOT / "prompts_r4.py"):
        return "prompts_r4.py changed since this batch's first launch."
    if lines.get("data_block", "unused") != (args.data_block or "unused"):
        return "--data-block differs from this batch's first launch."
    if args.data_block and lines.get("data_research_sha256") != _sha256(ROOT / "data_research.py"):
        return "data_research.py changed since this batch's first launch."
    if args.data_block and lines.get("plan_sha256") != _sha256(ROUND5_PLAN):
        return "the round-5 plan changed since this batch's first launch."
    return None


def _number_forms(x: float) -> list[str]:
    """Ways a value may be written in background text: as stored, without a trailing .0, with commas."""
    forms = {repr(float(x)), f"{x:g}", f"{x:,}", f"{x:,.2f}", f"{x:.2f}", f"{x:.3f}", f"{x:,.3f}"}
    forms |= {f.replace("-", "\u2212") for f in forms if f.startswith("-")}
    if float(x).is_integer():
        forms |= {str(int(x)), f"{int(x):,}"}
    return sorted((f for f in forms if len(f.replace(",", "").replace(".", "")) >= 2), key=len, reverse=True)


def scrub_value(text: str, value: float) -> tuple[str, int]:
    """Remove ForecastBench's freeze value from a text (round 5, all arms): returns the text and the count."""
    import re

    count = 0
    for form in _number_forms(value):
        # Not part of a longer number or a date (2026-09-20), and not followed by a unit of time ("10 years").
        pattern = (rf"(?<![\d.,\-/]){re.escape(form)}(?![\d\-/]|\.\d)"
                   + (r"(?!\s*(?:years?|months?|weeks?|days?)\b)" if "." not in form and "," not in form else ""))
        text, n = re.subn(pattern, "[value removed]", text)
        count += n
    return text, count


def round5_item(q: dict) -> dict:
    """A dataset item as the forecaster sees it: the freeze value removed from the background."""
    background, removed = scrub_value(q["background"], q["freeze_value"])
    return {**q, "background": background, "value_mentions_removed": removed}


def data_block_text(q: dict, mode: str, http=None) -> dict:
    """The research slot for a round-5 arm. off: nothing; fb: ForecastBench's single value;
    ours: the dated block from data_research (rows strictly before T after the publication-lag rules)."""
    if mode == "off":
        return {"text": "", "arm": "A0"}
    if mode == "fb":
        what = q.get("freeze_value_explanation") or "The latest value"
        return {"text": f"{what.rstrip('.')}: {q['freeze_value']:g} (as of {q['freeze_datetime'][:10]}).",
                "arm": "A1"}
    import data_research as dr

    freeze = datetime.fromisoformat(q["freeze_datetime"])
    try:
        series = [dr.series_for(q["source"], q["series_id"], freeze, http)]
    except (dr.DataAuthError, dr.DataTransient):
        raise  # a transient failure fails this question (retried on resume) rather than storing "no data"
    except dr.DataUnavailable as exc:
        return {"text": dr.format_block([], freeze), "arm": "A2", "error": dr.redact(str(exc))[:300]}
    return {"text": dr.format_block(series, freeze), "arm": "A2", "rows": len(series[0].rows),
            "last": series[0].rows[-1][0].isoformat()}


def done_ks(path: Path) -> dict[str, set[int]]:
    """The k values with a saved prediction, per question."""
    out: dict[str, set[int]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("prediction") is not None and row.get("k") is not None:
                out.setdefault(row["id"], set()).add(row["k"])
    return out


def missing_ks(have: set[int], predictions: int) -> list[int]:
    """The k values still to run: the unused ones, so a failed middle k is retried under its own number."""
    need = predictions - len(have)
    return [k for k in range(predictions + len(have)) if k not in have][:max(need, 0)]


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


def stored_research(batch: str) -> dict[str, dict]:
    rows = {}
    for line in (RUNS / batch / "research.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "text" in row and not row.get("error"):
            rows[row["id"]] = row
    return rows


def combined_row(qid: str, news: dict, wiki_rows: dict) -> dict:
    """The stored AskNews text, then the stored Wikipedia text, joined the way
    the live bot joins them. No fetching."""
    from wiki_research import join_research

    if qid not in news or qid not in wiki_rows:
        raise ValueError(f"{qid}: both stored texts are needed for the combined arm")
    return {"text": join_research(news[qid]["text"], wiki_rows[qid]["text"]),
            "sources": list(COMBO_SOURCES)}


def joined_row(qid: str, stores: list[dict]) -> dict:
    """Stored research texts for one question, joined in order with the live
    bot's join; a text that found nothing is left out."""
    from wiki_research import NO_ARTICLES, join_research

    texts = []
    for rows in stores:
        if qid not in rows:
            raise ValueError(f"{qid}: a stored text is missing for the joined arm")
        if rows[qid]["text"].strip() != NO_ARTICLES:
            texts.append(rows[qid]["text"])
    text = ""
    for t in texts:
        text = join_research(text, t)
    return {"text": text or NO_ARTICLES}


def combo_ids() -> list[str]:
    """The pre-registered ids; the hash is of the ids joined by newlines."""
    ids = json.loads(COMBO_IDS.read_text())
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != COMBO_IDS_SHA256:
        sys.exit("benchmarks/combo-ids.json changed since it was pre-registered.")
    return ids


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
    questions: list[dict], batch_dir: Path, predictions: int, parallel: int, research_mode: str = "none",
    sources: list[str] | None = None, lookback_days: int | None = None, reverse: bool = False,
    data_block: str | None = None,
) -> None:
    """predictions=0 with a research mode fetches and stores research only."""
    import ckai_bot
    from backtest.research import ResearchStopped
    from backtest.wiki import WikiLeak, WikiStopped
    from claude_cli_llm import ClaudeCliRefused
    from gdelt_research import GdeltStopped

    stops = (MetaculusBlocked, ClaudeCliRefused, ResearchStopped, WikiLeak, WikiStopped, GdeltStopped)

    bot = ckai_bot.build_bot(publish=False, forecaster_models=[FORECASTER])
    bot._reverse_binary = reverse
    out = batch_dir / "predictions.jsonl"
    done = done_counts(out)
    have_ks = done_ks(out)
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
    elif research_mode == "combined":
        from backtest.research import ResearchStore

        news, wiki_rows = (stored_research(b) for b in COMBO_SOURCES)

        async def combined_fetch(q: dict) -> dict:
            return combined_row(q["id"], news, wiki_rows)

        store = ResearchStore(batch_dir / "research.jsonl", fetcher=combined_fetch, serial=False)
    elif research_mode == "gdelt":
        import gdelt_research
        from backtest.research import ResearchStore
        from claude_cli_llm import PLAIN, ClaudeCliLlm

        helper = ClaudeCliLlm(HELPER, mode=PLAIN, allowed_tries=2)

        async def gdelt_fetch(q: dict) -> dict:
            return await gdelt_research.research_for(q, helper, lookback_days)

        store = ResearchStore(batch_dir / "research.jsonl", fetcher=gdelt_fetch, serial=False)
    elif research_mode == "joined":
        from backtest.research import ResearchStore

        stored = [stored_research(b) for b in sources]

        async def joined_fetch(q: dict) -> dict:
            return joined_row(q["id"], stored)

        store = ResearchStore(batch_dir / "research.jsonl", fetcher=joined_fetch, serial=False)
    if data_block:
        import data_research as dr
        from backtest.research import ResearchStore

        stops = stops + (dr.DataAuthError, dr.DataLeak)
        http = dr.PoliteHttp(cache=dr.CACHE_DIR)
        fetch_lock = asyncio.Lock()  # one data request at a time

        async def block_fetch(q: dict) -> dict:
            async with fetch_lock:
                return await asyncio.to_thread(data_block_text, q, data_block, http)

        store = ResearchStore(batch_dir / "research.jsonl", fetcher=block_fetch, serial=False)
    slots = asyncio.Semaphore(parallel)

    async def do_question(q: dict) -> None:
        async with slots:
            if predictions and done.get(q["id"], 0) >= predictions:
                print(f"done {q['id']}", flush=True)
                return
            try:
                research = await store.get(q) if store else ""
            except stops:
                raise
            except Exception as exc:  # this question's research failed: record it, go on, retry on resume
                async with lock:
                    with open(out, "a") as f:
                        f.write(json.dumps({"id": q["id"], "k": None, "prediction": None,
                                            "error": f"research: {type(exc).__name__}: {str(exc)[:500]}"}) + "\n")
                print(f"research failed {q['id']}", flush=True)
                return
            for k in missing_ks(have_ks.get(q["id"], set()), predictions):
                try:
                    row = await forecast_one(bot, q, k, research)
                except stops:
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
    parser.add_argument("--dataset", help="question list file (default backtest/data/questions.json)")
    parser.add_argument("--parallel", type=int, default=3, help="questions at a time")
    parser.add_argument("--probe-date", action="store_true")
    parser.add_argument("--forecaster", default="claude-cli/claude-opus-5-5", choices=sorted(FORECASTERS))
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"],
                        help="the forecaster's Claude Code effort level")
    parser.add_argument("--prompt", default="template", choices=["template", "v2"],
                        help="binary prompt: the template's or prompts_v2.py")
    parser.add_argument("--evidence", action="store_true", help="add the round-4 evidence-and-window step")
    parser.add_argument("--reverse", action="store_true",
                        help="ask for P(No) (round 4, X1); predictions are P(No), combined later")
    parser.add_argument("--data-block", choices=DATA_BLOCKS, default=None,
                        help="round 5: off (A0), fb (A1, ForecastBench's value) or ours (A2, dated block); "
                             "needs a --dataset of round-5 items")
    parser.add_argument("--research", default="none",
                        choices=["none", "asknews-historical", "wikipedia", "combined", "gdelt", "joined"])
    parser.add_argument("--sources", help="joined: batches whose stored research texts are joined, in order")
    parser.add_argument("--lookback-days", type=int, help="gdelt: look-back before the freeze, in days")
    parser.add_argument("--freeze-on", help="only questions frozen on these dates (comma-separated YYYY-MM-DD)")
    parser.add_argument("--sample", type=int, default=None, help="a seeded random sample of this many questions")
    parser.add_argument("--pilot", type=int, default=None, help="round 5 P0: this many items spread across clusters")
    parser.add_argument("--exclude-pilot", type=int, default=None, metavar="N",
                        help="round 5 D1: every item except the N-item pilot")
    parser.add_argument("--probe-wiki", type=int, default=None, metavar="N",
                        help="Wikipedia research only, for the first N questions; prints it, saves nothing")
    parser.add_argument("--probe-research", action="store_true",
                        help="two AskNews calls for one question; prints the dates, saves nothing")
    args = parser.parse_args()
    if args.evidence and args.reverse:
        parser.error("--evidence and --reverse can't be combined (the step's 'No' would be the reversed No)")
    global FORECASTER
    FORECASTER = args.forecaster
    if args.dataset:  # a held-out list instead of questions.json
        global QUESTIONS
        QUESTIONS = Path(args.dataset).resolve()

    from claude_cli_llm import refuse_in_ci

    refuse_in_ci()
    import ckai_bot  # noqa: F401  (loads .env, including RKIT_PATH)

    configure_env(args.effort, args.prompt, args.evidence)
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
    if args.freeze_on:
        days = set(args.freeze_on.split(","))
        chosen = [q for q in chosen if q["freeze_datetime"][:10] in days]
    if args.data_block:
        if not all("series_id" in q and "freeze_value" in q for q in chosen):
            parser.error("--data-block needs round-5 dataset items (backtest.fetch --datasets)")
        if args.research != "none" or args.evidence or args.reverse:
            parser.error("--data-block runs alone: no --research, --evidence or --reverse")
        chosen = [round5_item(q) for q in chosen]
    if args.research == "joined" and not args.sources:
        parser.error("--research joined needs --sources")
    if args.research == "gdelt" and not args.lookback_days:
        parser.error("--research gdelt needs --lookback-days")
    if args.sample:
        import random

        picked = set(q["id"] for q in random.Random(SAMPLE_SEED).sample(chosen, args.sample))
        chosen = [q for q in chosen if q["id"] in picked]
    if args.pilot or args.exclude_pilot:
        if not args.data_block or args.sample or (args.pilot and args.exclude_pilot):
            parser.error("--pilot / --exclude-pilot: round-5 items with --data-block, one of them, no --sample")
        pilot = set(pilot_ids(chosen, args.pilot or args.exclude_pilot))
        chosen = [q for q in chosen if (q["id"] in pilot) == bool(args.pilot)]
    if args.research == "combined":  # only the pre-registered questions with both texts
        wanted = set(combo_ids())
        chosen = [q for q in chosen if q["id"] in wanted]
        if len(chosen) != len(wanted):
            sys.exit("Some pre-registered combined-arm questions aren't in the chosen range.")

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
        first = launch.split("\n\n")[0]
        if problem := check_resume_settings(first, args):
            sys.exit(problem)
        for key, value in (("gdelt_lookback_days", args.lookback_days), ("joined_from", args.sources)):
            line = next((ln for ln in first.splitlines() if ln.startswith(key + ": ")), None)
            if line and key == "gdelt_lookback_days" and line != f"{key}: {value}":
                sys.exit("--lookback-days differs from this batch's first launch.")
            if line and key == "joined_from" and [x.split("/")[1] for x in line[len(key) + 2:].split(", ")] != (
                    (value or "").split(",")):
                sys.exit("--sources differs from this batch's first launch.")
    ids_path = batch_dir / "question_ids.json"
    if ids_path.exists() and json.loads(ids_path.read_text()) != [q["id"] for q in chosen]:
        sys.exit("--start/--limit/--sample differ from this batch's first launch.")
    ids_path.write_text(json.dumps([q["id"] for q in chosen]))
    write_launch(batch_dir, meta, args)
    # Recorded at start: a run that is stopped (killed) never reaches the finally block below.
    from claude_cli_llm import get_backend

    try:
        logs = str(get_backend().run_dir)
    except Exception as exc:
        logs = f"unavailable ({exc})"
    with open(batch_dir / "launch.txt", "a") as f:
        f.write(f"call_logs: {logs}\n")

    install_date_stub()
    try:
        asyncio.run(run_batch(chosen, batch_dir, args.predictions, args.parallel, args.research,
                              args.sources.split(",") if args.sources else None, args.lookback_days,
                              args.reverse, args.data_block))
    finally:
        with open(batch_dir / "launch.txt", "a") as f:
            f.write(f"utc_end: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n")


if __name__ == "__main__":
    main()
