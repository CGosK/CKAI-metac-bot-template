"""
News headlines from GDELT, limited to what GDELT had seen before a time T.

For one question: a helper model (Haiku, plain mode, no tools) writes up to 2
search queries from the question and resolution criteria. Each query runs on
GDELT's DOC 2.0 API over three equal sub-windows of [T - look-back, T), so the
headlines span the look-back instead of bunching just before T. Articles GDELT
saw at or after T are dropped and counted. Only headlines, dates and domains
are used: no article is fetched (a page fetched today may have changed since T).

GDELT asks for at most one request every 5 seconds and reports errors and
throttling as plain text. Requests go one at a time with a gap; a text reply
is an error, never "no articles". Results are cached on disk by query and
window, so a stopped run resumes where it was.

Plan: benchmarks/PLAN-refine-round2.md, section 3 (exploratory).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wiki_research import NO_ARTICLES, leak_scan

API = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "CKAI-metac-bot/0.1 (https://github.com/CGosK/CKAI-metac-bot-template)"
CACHE = Path(__file__).resolve().parent / "backtest" / "data" / "gdelt-cache"
GAP_SECONDS = 60.0  # GDELT asks for 5 s, but refused 10-15 s gaps after early bursts; 60 s worked (amendment 1)
THROTTLE_WAITS = (120, 600)  # seconds to wait after a throttle reply, then stop
MAX_QUERIES = 1  # amendment 1 (was 2): halves the requests
SUB_WINDOWS = 3
PER_WINDOW = 25
MAX_HEADLINES = 30
THROTTLE = re.compile(r"limit requests|rate limit|too many requests", re.I)


class GdeltStopped(RuntimeError):
    """GDELT kept throttling or failing: stop the run (it resumes from the cache)."""


class GdeltQueryError(RuntimeError):
    """GDELT rejected one query (for example a keyword too short): this question only."""


class Api:
    def __init__(self, cache: Path | None = CACHE, gap: float = GAP_SECONDS,
                 throttle_waits: tuple[float, ...] = THROTTLE_WAITS):
        self._lock = threading.Lock()
        self._last = 0.0
        self.requests = 0
        self.cache, self.gap, self.throttle_waits = cache, gap, throttle_waits

    def _fetch(self, url: str) -> tuple[int, str]:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            return 0, f"network error: {exc}"

    def get(self, params: dict) -> list[dict]:
        """The articles for one request. Empty JSON means none; an empty body or text is an error."""
        key = json.dumps(params, sort_keys=True)
        path = self.cache / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json") if self.cache else None
        if path and path.exists():
            return json.loads(path.read_text())
        url = API + "?" + urllib.parse.urlencode({**params, "mode": "artlist", "format": "json"})
        with self._lock:
            for attempt in range(len(self.throttle_waits) + 1):
                wait = self._last + self.gap - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                self.requests += 1
                status, body = self._fetch(url)
                parsed = None
                if status == 200 and body.strip():
                    try:
                        parsed = json.loads(body)
                    except json.JSONDecodeError:
                        parsed = None
                # Network errors and 5xx are waited out like throttling (a deviation from the plan's
                # "429 or a throttle notice", noted there); a notice is only looked for in non-JSON text.
                throttled = parsed is None and (status in (0, 429) or status >= 500
                                                or (status == 200 and THROTTLE.search(body[:300])))
                if throttled:
                    if attempt < len(self.throttle_waits):
                        time.sleep(self.throttle_waits[attempt])
                        continue
                    raise GdeltStopped(f"GDELT throttled or failed {attempt + 1} times: {status} {body[:200]}")
                if status != 200:
                    raise GdeltQueryError(f"GDELT {status}: {body[:200]}")
                if not isinstance(parsed, dict):  # text or an empty body: an error, never "no articles"
                    raise GdeltQueryError(f"GDELT replied without JSON: {body[:200]!r}")
                articles: list[dict] = parsed.get("articles", [])
                break
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(articles))
            tmp.replace(path)
        return articles


_api = Api()


def _stamp(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def seen_at(article: dict) -> datetime:
    """GDELT's seendate, e.g. 20260917T123000Z, as a UTC datetime."""
    return datetime.strptime(article["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def sub_windows(freeze: datetime, lookback_days: int, parts: int = SUB_WINDOWS) -> list[tuple[datetime, datetime]]:
    start = freeze - timedelta(days=lookback_days)
    step = (freeze - start) / parts
    return [(start + step * i, start + step * (i + 1)) for i in range(parts)]


def _json_in(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in helper reply: {text[:200]}")
    return json.loads(m.group(0))


async def pick_queries(llm, question: str, criteria: str) -> list[str]:
    prompt = (
        "A forecaster wants recent news headlines about this question from the GDELT news search. Write "
        + ("one short search query" if MAX_QUERIES == 1 else f"up to {MAX_QUERIES} short search queries")
        + ". Use the main names, places and terms; put multi-word names in "
        "double quotes; every word at least 4 characters; no OR, no parentheses, no dates.\n\n"
        f"Question: {question}\n\nResolution criteria: {criteria[:1500]}\n\n"
        'Reply with JSON only: {"queries": [...]}'
    )
    for attempt in (1, 2):
        reply = await llm.invoke(prompt)
        try:
            data = _json_in(reply)
            break
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise
    queries = [str(x).strip() for x in data.get("queries", []) if str(x).strip()]
    return queries[:MAX_QUERIES]


def headlines(queries: list[str], freeze: datetime, lookback_days: int, api: Api = _api) -> dict:
    """Articles for the queries over the sub-windows, seen before T, deduplicated."""
    kept: dict[str, dict] = {}
    returned = dropped = 0
    for query in queries:
        for start, end in sub_windows(freeze, lookback_days):
            articles = api.get({"query": f"{query} sourcelang:english", "maxrecords": str(PER_WINDOW),
                                "sort": "datedesc", "startdatetime": _stamp(start), "enddatetime": _stamp(end)})
            for a in articles:
                returned += 1
                if seen_at(a) >= freeze:
                    dropped += 1
                    continue
                title = " ".join((a.get("title") or "").split())
                norm = re.sub(r"\W+", " ", title.lower()).strip()
                if norm and norm not in kept:
                    kept[norm] = {"seen": a["seendate"], "domain": a.get("domain", ""), "title": title}
    rows = sorted(kept.values(), key=lambda a: a["seen"], reverse=True)[:MAX_HEADLINES]
    return {"returned": returned, "dropped_after_freeze": dropped, "articles": rows}


def format_headlines(rows: list[dict], freeze: datetime, lookback_days: int) -> str:
    if not rows:
        return NO_ARTICLES
    start = (freeze - timedelta(days=lookback_days)).date().isoformat()
    lines = [f"- {datetime.strptime(a['seen'], '%Y%m%dT%H%M%SZ').date().isoformat()}, {a['domain']}: {a['title']}"
             for a in rows]
    return f"News headlines from {start} to {freeze.date().isoformat()} (GDELT):\n" + "\n".join(lines)


async def research_for(q: dict, llm, lookback_days: int, api: Api = _api) -> dict:
    import asyncio

    freeze = datetime.fromisoformat(q["freeze_datetime"])
    queries = await pick_queries(llm, q["question"], q.get("criteria", ""))
    found = await asyncio.to_thread(headlines, queries, freeze, lookback_days, api)
    text = format_headlines(found["articles"], freeze, lookback_days)
    return {
        "freeze": freeze.isoformat(),
        "lookback_days": lookback_days,
        "queries": queries,
        "returned": found["returned"],
        "dropped_after_freeze": found["dropped_after_freeze"],
        "articles": found["articles"],
        "text": text,
        "leak": leak_scan(text, freeze),
    }
