"""
Date-limited AskNews research for the backtest.

Mirrors the live bot's research (RESEARCHER_MODEL=asknews/news-summaries, which
calls forecasting-tools' AskNewsSearcher.get_formatted_news_async with the
question text): two news searches, 6 and 10 articles, formatted by the library.
Here both searches are limited to what AskNews had crawled by the question's
freeze time T, using the windows the SDK documents for its two presets
("latest news": method nl, past 24 hours; "news knowledge": method kw, past
60 days).

Each question's research is fetched once, saved with its raw articles and the
call parameters, and reused for every prediction and on resume. The text is
also scanned for signs of information from after T and for quoted market
prices; both flags are reported by the scorer.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

QUERY_CALLS = (
    # (name, articles, method, look-back)
    ("latest", 6, "nl", timedelta(hours=24)),
    ("knowledge", 10, "kw", timedelta(days=60)),
)
PAUSE_SECONDS = 12  # the library's pause for AskNews's free-tier rate limit
NO_ARTICLES = "No articles were found.\n\n"
HEADER = "Here are the relevant news articles:\n\n"

MONTHS = ("january february march april may june july august september october november december").split()
MARKET_WORDS = re.compile(
    r"\b(polymarket|kalshi|manifold|metaculus|predictit|betting (?:market|odds)|prediction market|odds)\b",
    re.IGNORECASE,
)
OUTCOME_WORDS = re.compile(
    r"\b(was elected|won the|wins the|has won|was confirmed|was announced as|final result|"
    r"resolved (?:yes|no)|officially (?:declared|announced))\b",
    re.IGNORECASE,
)


class ResearchStopped(RuntimeError):
    """Auth, quota or repeated rate-limit errors: stop the run rather than
    record empty research."""


def windows(freeze: datetime) -> list[dict]:
    end = int(freeze.timestamp())
    return [
        {
            "name": name,
            "n_articles": n,
            "method": method,
            "start_timestamp": int((freeze - back).timestamp()),
            "end_timestamp": end,
        }
        for name, n, method, back in QUERY_CALLS
    ]


def _as_utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def later_than(articles: list[dict], freeze: datetime) -> list[dict]:
    return [a for a in articles if _as_utc(a["pub_date"]) > freeze]


def leak_flags(text: str, freeze: datetime) -> dict:
    """Signs in the research text of information from after the freeze:
    a named month and year after T's month, a year after T's, or words that
    report an outcome. A flag is a prompt to read, not proof of a leak."""
    lower = text.lower()
    later_months = []
    for m, name in enumerate(MONTHS, start=1):
        for year in re.findall(rf"\b{name}\s+(?:\d{{1,2}},?\s+)?(20\d\d)\b", lower):
            if (int(year), m) > (freeze.year, freeze.month):
                later_months.append(f"{name} {year}")
    later_years = sorted({y for y in re.findall(r"\b(20\d\d)\b", lower) if int(y) > freeze.year})
    outcome = sorted({w.lower() for w in OUTCOME_WORDS.findall(text)})
    score = 3 * len(later_months) + 2 * len(later_years) + len(outcome)
    return {
        "later_months": sorted(set(later_months)),
        "later_years": later_years,
        "outcome_words": outcome,
        "score": score,
    }


def market_flag(text: str) -> list[str]:
    return sorted({w.lower() for w in MARKET_WORDS.findall(text)})


def format_research(per_call: list[list]) -> str:
    """The live bot's formatting: each call's articles formatted by the
    library's own formatter (newest first within a call), joined in call order,
    with the library's text when nothing was found."""
    from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher

    if not any(per_call):
        return HEADER + NO_ARTICLES
    return HEADER + "".join(
        AskNewsSearcher._format_articles(None, articles)  # type: ignore[arg-type]
        for articles in per_call
        if articles
    )


async def fetch(question: str, freeze: datetime) -> dict:
    """Both searches for one question, one at a time with the library's pause."""
    from asknews_sdk import AsyncAskNewsSDK
    from asknews_sdk.errors import (
        ConcurrencyLimitExceededError,
        ForbiddenError,
        RateLimitExceededError,
        RequestTimeoutError,
        ServiceUnavailableError,
        UnauthorizedError,
    )

    os.environ["ASKNEWS_CACHE_MODE"] = "no_cache"
    api_key = os.getenv("ASKNEWS_API_KEY")
    client_id, secret = os.getenv("ASKNEWS_CLIENT_ID"), os.getenv("ASKNEWS_SECRET")
    if client_id and secret:
        auth = {"client_id": client_id, "client_secret": secret}
    elif api_key:
        auth = {"api_key": api_key}
    else:
        raise ResearchStopped("No AskNews credentials in the environment.")

    calls, kept, all_articles = [], [], []
    transient = (RateLimitExceededError, ConcurrencyLimitExceededError, RequestTimeoutError,
                 ServiceUnavailableError)
    async with AsyncAskNewsSDK(**auth, scopes={"news"}) as ask:
        for i, params in enumerate(windows(freeze)):
            if i:
                await asyncio.sleep(PAUSE_SECONDS)
            request = {k: v for k, v in params.items() if k != "name"}
            for attempt in (1, 2):
                try:
                    response = await ask.news.search_news(
                        query=question,
                        return_type="both",
                        strategy="default",
                        historical=True,
                        hours_back=None,
                        **request,
                    )
                    break
                except transient as exc:
                    if attempt == 2:
                        raise ResearchStopped(f"AskNews failed twice: {exc}") from exc
                    await asyncio.sleep(60)
                except (UnauthorizedError, ForbiddenError) as exc:
                    raise ResearchStopped(f"AskNews refused the call: {exc}") from exc
            articles = list(response.as_dicts or [])
            ok = [a for a in articles if _as_utc(a.pub_date) <= freeze]
            kept.append(ok)
            all_articles += [a.model_dump(mode="json") for a in articles]
            usage = getattr(response, "usage", None)
            calls.append({
                **params,
                "returned": len(articles),
                "dropped_after_freeze": len(articles) - len(ok),
                "usage": usage.model_dump(mode="json") if hasattr(usage, "model_dump") else usage,
            })
    text = format_research(kept)
    return {
        "freeze": freeze.isoformat(),
        "query": question,
        "calls": calls,
        "articles": all_articles,
        "text": text,
        "leak": leak_flags(text, freeze),
        "market_words": market_flag(text),
    }


class ResearchStore:
    """research.jsonl in a batch: one line per question, written once."""

    def __init__(self, path: Path, fetcher=None, serial: bool = True):
        """`fetcher(q)` returns the row for a question; by default AskNews.
        `serial` runs one question's research at a time (AskNews's rate limit);
        otherwise only the file write is locked."""
        self.path = path
        self.fetcher = fetcher
        self.serial = serial
        self.write_lock = asyncio.Lock()
        self.rows: dict[str, dict] = {}
        if path.exists():
            raw = path.read_text()
            if raw and not raw.endswith("\n"):  # a hard kill cut the last line; start a fresh one
                with open(path, "a") as f:
                    f.write("\n")
            for line in raw.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # a line cut short by a hard kill
                    continue
                self.rows[row["id"]] = row
        self.lock = asyncio.Lock()  # one AskNews question at a time

    async def get(self, q: dict) -> str:
        if self.fetcher is not None and not self.serial:
            if q["id"] not in self.rows:
                row = {"id": q["id"], **await self.fetcher(q)}
                async with self.write_lock:
                    if q["id"] not in self.rows:
                        with open(self.path, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        self.rows[q["id"]] = row
            return self.rows[q["id"]]["text"]
        async with self.lock:
            if q["id"] not in self.rows:
                if self.fetcher is not None:
                    row = {"id": q["id"], **await self.fetcher(q)}
                else:
                    row = {"id": q["id"], **await fetch(q["question"], datetime.fromisoformat(q["freeze_datetime"]))}
                with open(self.path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                self.rows[q["id"]] = row
                if self.fetcher is None:
                    await asyncio.sleep(PAUSE_SECONDS)
            return self.rows[q["id"]]["text"]
