"""
Free, date-limited research from Wikipedia for the backtest.

For a question frozen at time T, the research text holds:
- up to 4 Wikipedia articles as they stood at T. Haiku 5.5 picks the titles
  from the question alone (Wikipedia's search index is today's, so it isn't
  used) and, per article, the sections worth reading;
- items from Wikipedia's daily Current events portal for the 60 days before
  T, each page as it stood at T, kept when they link to a chosen article or
  share keywords with the question.

Every revision used is checked to be at or before T. Wikitext is converted
locally: no server-side template expansion, which could pull today's values
from Wikidata. The helper model calls run in the claude-cli backend's plain
mode, which has no tools, so no web search.
"""

from __future__ import annotations

import asyncio
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

import mwparserfromhell as mwp

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "backtest-research/0.1 (python-urllib)"
CACHE = Path(__file__).resolve().parent / "data" / "wiki-cache"
GAP_SECONDS = 2.0
TRIES = 5
MAX_ARTICLES = 4
ARTICLE_WORDS = 1500
PORTAL_WORDS = 1500
PORTAL_DAYS = 60
TABLE_ROWS = 25
MAX_SECTIONS = 4
ALWAYS_SECTIONS = re.compile(r"poll|opinion|forecast|candidate|result|standings|schedule", re.IGNORECASE)
SKIP_SECTIONS = re.compile(r"(see also|references|external links|further reading|notes|bibliography|sources)$",
                           re.IGNORECASE)
RENDER = {"start date", "end date", "date", "birth date and age", "birth date", "death date and age",
          "convert", "cvt", "as of", "percentage", "nowrap", "small", "abbr", "lang", "efn"}
NO_ARTICLES = "No articles were found."


class WikiLeak(RuntimeError):
    """A revision later than the freeze time: stop the run."""


class WikiStopped(RuntimeError):
    """Wikipedia refused or kept failing: stop rather than record empty research."""


# ---------------------------------------------------------------- the API


class Api:
    """One request at a time, a gap between requests, maxlag, one retry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last = 0.0
        self.requests = 0

    def get(self, params: dict) -> dict:
        query = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
        url = API + "?" + urllib.parse.urlencode(query)
        with self._lock:
            for attempt in range(1, TRIES + 1):
                wait = self._last + GAP_SECONDS - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                self.requests += 1
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        data = json.load(response)
                except urllib.error.HTTPError as exc:
                    if attempt < TRIES and (exc.code == 429 or exc.code >= 500):
                        retry_after = exc.headers.get("Retry-After", "")
                        time.sleep(max(float(retry_after) if retry_after.isdigit() else 0, 15 * attempt))
                        continue
                    raise WikiStopped(f"Wikipedia HTTP {exc.code} after {attempt} tries") from exc
                except (urllib.error.URLError, TimeoutError) as exc:
                    if attempt < TRIES:
                        time.sleep(15 * attempt)
                        continue
                    raise WikiStopped(f"Wikipedia unreachable: {exc}") from exc
                if data.get("error", {}).get("code") == "maxlag" and attempt < TRIES:
                    time.sleep(10)
                    continue
                if "error" in data:
                    raise WikiStopped(f"Wikipedia error: {data['error']}")
                return data
        raise WikiStopped(f"Wikipedia failed {TRIES} times")


_api = Api()
_portal_lock = threading.Lock()


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check(rev: dict, freeze: datetime) -> None:
    ts = datetime.fromisoformat(rev["timestamp"].replace("Z", "+00:00"))
    if ts > freeze:
        raise WikiLeak(f"revision {rev.get('revid')} at {rev['timestamp']} is after {freeze.isoformat()}")


def _cache_path(kind: str, key: str) -> Path:
    return CACHE / kind / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def _cached(kind: str, key: str, make):
    path = _cache_path(kind, key)
    if path.exists():
        return json.loads(path.read_text())
    value = make()
    _write(path, value)
    return value


def revision_at(title: str, freeze: datetime, api: Api = _api) -> dict | None:
    """The page's latest revision at or before T, without today's redirects.
    Returns {"title", "revid", "timestamp", "content"} or None."""

    def make():
        data = api.get({
            "action": "query", "prop": "revisions", "titles": title, "rvprop": "ids|timestamp|content",
            "rvslots": "main", "rvstart": _iso(freeze), "rvdir": "older", "rvlimit": "1",
        })
        page = data["query"]["pages"][0]
        revs = page.get("revisions") or []
        if not revs:
            return None
        rev = revs[0]
        content = rev["slots"]["main"].get("content")
        if content is None:  # text hidden by an admin
            return {"title": page["title"], "revid": rev["revid"], "timestamp": rev["timestamp"],
                    "content": None}
        return {"title": page["title"], "revid": rev["revid"], "timestamp": rev["timestamp"],
                "content": content}

    rev = _cached("rev", f"{title}|{_iso(freeze)}", make)
    if rev is not None:
        _check(rev, freeze)
    return rev


REDIRECT = re.compile(r"^\s*#REDIRECT\s*\[\[([^\]|#]+)", re.IGNORECASE)


def moved_after(title: str, freeze: datetime, api: Api = _api) -> str | None:
    """If the page at `title` was moved away after T, where it went (its
    history moved with it). Otherwise None: the title wasn't a page at T."""

    def make():
        data = api.get({"action": "query", "list": "logevents", "letype": "move", "letitle": title,
                        "lestart": _iso(freeze), "ledir": "newer", "lelimit": "1", "leprop": "timestamp|details"})
        events = data["query"].get("logevents") or []
        if not events:
            return None
        return (events[0].get("params") or {}).get("target_title")

    return _cached("moved", f"{title}|{_iso(freeze)}", make)


def article_at(title: str, freeze: datetime, api: Api = _api) -> dict | None:
    """Follow redirects as they stood at T. If the title has no revision before
    T because its page was moved after T (the move log says so), take the moved
    page's revision at T: page history moves with the page. The title shown is
    always the one the page had at T, never today's."""
    seen = set()
    current, shown, moved = title, title, False
    for _ in range(4):
        if current in seen:
            return None
        seen.add(current)
        rev = revision_at(current, freeze, api)
        if rev is None:
            target = moved_after(current, freeze, api)
            if target and target not in seen:
                current, moved = target, True  # `shown` stays the title it had at T
                continue
            return None
        if rev["content"] is None:
            return None
        m = REDIRECT.match(rev["content"])
        if not m:
            return {**rev, "title": shown if moved else rev["title"]}
        current = shown = m.group(1).strip()
        moved = False
    return None


# ---------------------------------------------------------------- wikitext to text


def _param(t, name, default=""):
    try:
        return t.get(name).value.strip_code().strip()
    except ValueError:
        return default


def _render_template(t) -> str:
    name = str(t.name).strip().lower().replace("_", " ")
    pos = [p.value.strip_code().strip() for p in t.params if not p.showkey]
    if name in ("start date", "end date", "date", "birth date", "birth date and age", "death date and age"):
        return "-".join(x for x in pos[:3] if x)
    if name in ("convert", "cvt"):
        return " ".join(pos[:2])
    if name == "as of":
        return "As of " + " ".join(pos[:3])
    if name == "percentage" and len(pos) >= 2:
        try:
            return f"{100 * float(pos[0].replace(',', '')) / float(pos[1].replace(',', '')):.1f}%"
        except (ValueError, ZeroDivisionError):
            return ""
    if name in ("nowrap", "small", "abbr"):
        return pos[0] if pos else ""
    if name == "lang":
        return pos[1] if len(pos) > 1 else ""
    return ""


def _infobox(t) -> str:
    lines = []
    for p in t.params:
        if not p.showkey:
            continue
        key = str(p.name).strip()
        value = to_text(str(p.value))
        value = " ".join(value.split())
        if value and not key.lower().startswith(("image", "caption", "logo", "map", "alt", "signature", "pushpin",
                                                      "coordinates", "screenshot", "website", "embed")):
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


CELL_ATTR = re.compile(r'^\s*[a-z\-]+\s*=\s*("[^"]*"|\S+)(\s+[a-z\-]+\s*=\s*("[^"]*"|\S+))*\s*\|(?!\|)', re.I)


def _table(wikitext: str) -> str:
    """A wikitable as tab-separated rows: header rows, then the first data rows."""
    rows: list[tuple[bool, list[str]]] = []
    for raw_row in re.split(r"^\s*\|-.*$", wikitext, flags=re.M):
        cells, header, data = [], False, False
        for line in raw_row.splitlines():
            line = line.strip()
            if line.startswith(("{|", "|}", "|+")) or not line:
                continue
            if line.startswith("!"):
                header = True
                parts = re.split(r"!!|\|\|", line[1:])
            elif line.startswith("|"):
                data = True
                parts = line[1:].split("||")
            else:
                if cells:
                    cells[-1] = " ".join((cells[-1] + " " + to_text(line)).split())
                continue
            for part in parts:
                part = CELL_ATTR.sub("", part, count=1)
                cells.append(" ".join(to_text(part).split()))
        if cells:
            rows.append((header and not data, cells))
    headers = [r for h, r in rows if h][:3]
    data = [r for h, r in rows if not h][:TABLE_ROWS]
    return "\n".join("\t".join(r) for r in headers + data)


TABLE = re.compile(r"^\{\|.*?^\|\}", re.S | re.M)


def to_text(wikitext: str) -> str:
    """Wikitext to plain text, locally: infoboxes as key: value lines, a few
    templates rendered from their own parameters, tables kept as rows, every
    other template, reference, file and category dropped."""
    tables: list[str] = []
    token = uuid.uuid4().hex[:8]

    def keep_table(m):
        tables.append(_table(m.group(0)))
        return f"\n@@TABLE{token}x{len(tables) - 1}@@\n"

    wikitext = TABLE.sub(keep_table, wikitext)
    code = mwp.parse(wikitext)
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).lower() in ("ref", "gallery", "references"):
            try:
                code.remove(tag)
            except ValueError:
                pass
    for link in code.filter_wikilinks(recursive=True):
        if str(link.title).strip().lower().startswith(("file:", "image:", "category:")):
            try:
                code.remove(link)
            except ValueError:
                pass
    for t in code.filter_templates(recursive=False):
        name = str(t.name).strip().lower().replace("_", " ")
        try:
            if name.startswith("infobox"):
                code.replace(t, "\n" + _infobox(t) + "\n")
            elif name in RENDER:
                code.replace(t, _render_template(t))
            else:
                code.remove(t)
        except ValueError:
            pass
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(rf"@@TABLE{token}x(\d+)@@", lambda m: tables[int(m.group(1))], text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sections(wikitext: str) -> list[tuple[str, str]]:
    """(heading, wikitext) for the lead and each level-2 or level-3 section."""
    parts = re.split(r"^(={2,3})\s*(.+?)\s*\1\s*$", wikitext, flags=re.M)
    out = [("Lead", parts[0])]
    for i in range(1, len(parts) - 2, 3):
        out.append((mwp.parse(parts[i + 1]).strip_code().strip(), parts[i + 2]))
    return out


def _words(text: str) -> int:
    return len(text.split())


def _cap(text: str, words: int) -> str:
    """Cut after `words` words, counting words separated by any whitespace."""
    if words <= 0:
        return ""
    count = 0
    for m in re.finditer(r"\S+", text):
        count += 1
        if count == words:
            rest = text[m.end():]
            return text if not rest.strip() else text[: m.end()] + " [...]"
    return text


# ---------------------------------------------------------------- the portal


def portal_title(day: datetime) -> str:
    return f"Portal:Current events/{day.year} {day.strftime('%B')} {day.day}"


def portal_revisions(freeze: datetime, api: Api = _api) -> dict[str, dict]:
    """Each day page for the 60 days before T, as it stood at T."""

    def make():
        chosen: dict[str, int] = {}
        for d in range(1, PORTAL_DAYS + 1):
            title = portal_title(freeze - timedelta(days=d))
            data = api.get({"action": "query", "prop": "revisions", "titles": title,
                            "rvprop": "ids|timestamp", "rvstart": _iso(freeze), "rvdir": "older",
                            "rvlimit": "1"})
            revs = data["query"]["pages"][0].get("revisions") or []
            if revs:
                chosen[title] = revs[0]["revid"]
        out: dict[str, dict] = {}
        ids = list(chosen.values())
        for i in range(0, len(ids), 50):
            data = api.get({"action": "query", "prop": "revisions", "revids": "|".join(map(str, ids[i:i + 50])),
                            "rvprop": "ids|timestamp|content", "rvslots": "main"})
            for page in data["query"]["pages"]:
                for rev in page.get("revisions", []):
                    if rev["slots"]["main"].get("content") is None:
                        continue
                    out[page["title"]] = {"revid": rev["revid"], "timestamp": rev["timestamp"],
                                          "content": rev["slots"]["main"]["content"]}
        return out

    with _portal_lock:  # questions sharing a freeze time fetch the pages once
        pages = _cached("portal", _iso(freeze), make)
    for rev in pages.values():
        _check(rev, freeze)
    return pages


def portal_items(content: str) -> list[dict]:
    """Leaf bullet items with their parent chain and their link targets."""
    items, stack = [], []
    lines = [ln for ln in content.splitlines() if ln.startswith("*")]
    for i, line in enumerate(lines):
        depth = len(line) - len(line.lstrip("*"))
        body = line[depth:].strip()
        stack = stack[: depth - 1] + [body]
        next_depth = (len(lines[i + 1]) - len(lines[i + 1].lstrip("*"))) if i + 1 < len(lines) else 0
        if next_depth > depth:
            continue  # a parent topic, shown with its leaves
        chain = [to_text(s) for s in stack[:-1]]
        links = set()
        for s in stack:
            for link in mwp.parse(s).filter_wikilinks():
                links.add(_norm(str(link.title)))
        items.append({"chain": [c for c in chain if c], "text": to_text(body), "links": links})
    return items


def _norm(title: str) -> str:
    t = title.strip().replace("_", " ")
    return (t[:1].upper() + t[1:]) if t else t


def keyword_hits(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for k in keywords if k and k.lower() in lower)


# ---------------------------------------------------------------- helper model calls


def _json_in(text: str):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON in helper reply: {text[:200]}")
    return json.loads(m.group(0))


async def pick_titles(llm, question: str, criteria: str) -> dict:
    prompt = (
        "A forecaster needs background from English Wikipedia for this question. Using only your own "
        "knowledge (no browsing), name up to 4 existing English Wikipedia article titles most likely to "
        "hold relevant facts as of the question's date: the event itself, the main people or organisations, "
        "and for an election its \"Opinion polling for ...\" article if one likely exists. Also give up to 8 "
        "short keywords (names, places, terms) that news items about this question would contain.\n\n"
        f"Question: {question}\n\nResolution criteria: {criteria[:1500]}\n\n"
        'Reply with JSON only: {"titles": [...], "keywords": [...]}'
    )
    for attempt in (1, 2):
        reply = await llm.invoke(prompt)
        try:
            data = _json_in(reply)
            break
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise
    titles = [str(t).strip() for t in data.get("titles", []) if str(t).strip()][:MAX_ARTICLES]
    keywords = [str(k).strip() for k in data.get("keywords", []) if str(k).strip()][:8]
    return {"titles": titles, "keywords": keywords, "reply": reply}


async def pick_sections(llm, question: str, title: str, heads: list[tuple[str, int]]) -> list[int]:
    listing = "\n".join(f"{i}. {h} ({w} words)" for i, (h, w) in enumerate(heads))
    prompt = (
        f"A forecaster is answering: {question}\n\nThe Wikipedia article \"{title}\" has these sections:\n"
        f"{listing}\n\nWhich sections (up to {MAX_SECTIONS}, not counting 0, the lead) would most help? "
        "Reply with JSON only: a list of section numbers."
    )
    reply = await llm.invoke(prompt)
    picked = _json_in(reply)
    return [int(i) for i in picked if isinstance(i, (int, str)) and str(i).isdigit()][:MAX_SECTIONS]


# ---------------------------------------------------------------- research for one question


async def article_text(llm, question: str, rev: dict, keywords: list[str]) -> dict:
    secs = sections(rev["content"])
    heads = [(h, _words(mwp.parse(body).strip_code())) for h, body in secs]
    always = [i for i, (h, _) in enumerate(heads) if i and ALWAYS_SECTIONS.search(h)]
    skip = {i for i, (h, _) in enumerate(heads) if SKIP_SECTIONS.match(h)}
    path = _cache_path("sections", f"{rev['revid']}|{question}")
    if path.exists():
        picked = json.loads(path.read_text())
    else:
        try:
            picked = await pick_sections(llm, question, rev["title"], heads)
        except (ValueError, json.JSONDecodeError):  # no usable JSON: lead and the always-kept sections
            picked = []
        _write(path, picked)
    chosen = sorted({0, *[i for i in picked if 0 < i < len(secs)], *always[:3]} - skip)
    budget, parts = ARTICLE_WORDS, []
    for i in chosen:
        h, body = secs[i]
        text = await asyncio.to_thread(to_text, body)
        if not text:
            continue
        text = _cap(text, budget)
        if not text:
            break
        budget -= _words(text)
        parts.append(text if i == 0 else f"== {h} ==\n{text}")
        if budget <= 0:
            break
    return {"title": rev["title"], "revid": rev["revid"], "timestamp": rev["timestamp"],
            "sections": [secs[i][0] for i in chosen], "text": "\n\n".join(parts)}


async def research_for(q: dict, llm, api: Api = _api) -> dict:
    freeze = datetime.fromisoformat(q["freeze_datetime"])
    picks = await pick_titles(llm, q["question"], q.get("criteria", ""))
    articles = []
    for title in picks["titles"]:
        rev = await asyncio.to_thread(article_at, title, freeze, api)
        if rev is None:
            continue
        articles.append(await article_text(llm, q["question"], rev, picks["keywords"]))

    pages = await asyncio.to_thread(portal_revisions, freeze, api)
    chosen_titles = {_norm(a["title"]) for a in articles} | {_norm(t) for t in picks["titles"]}
    kept, budget = [], PORTAL_WORDS
    for d in range(1, PORTAL_DAYS + 1):
        day = freeze - timedelta(days=d)
        page = pages.get(portal_title(day))
        if not page:
            continue
        for item in await asyncio.to_thread(portal_items, page["content"]):
            full = " ".join(item["chain"] + [item["text"]])
            if item["links"] & chosen_titles or keyword_hits(full, picks["keywords"]) >= 2:
                line = f"- {day.date().isoformat()}: " + " > ".join(item["chain"] + [item["text"]])
                if budget - _words(line) < 0:
                    break
                budget -= _words(line)
                kept.append(line)

    date = freeze.date().isoformat()
    blocks = []
    if articles:
        blocks.append(f"Wikipedia articles as they stood on {date}:\n\n" + "\n\n".join(
            f"### {a['title']}\n{a['text']}" for a in articles))
    if kept:
        start = (freeze - timedelta(days=PORTAL_DAYS)).date().isoformat()
        blocks.append(f"News items from Wikipedia's Current events portal, {start} to {date}:\n" + "\n".join(kept))
    text = "\n\n".join(blocks) if blocks else NO_ARTICLES
    return {
        "freeze": freeze.isoformat(),
        "titles_asked": picks["titles"],
        "keywords": picks["keywords"],
        "articles": [{k: a[k] for k in ("title", "revid", "timestamp", "sections")} for a in articles],
        "portal_items": len(kept),
        "words": _words(text),
        "text": text,
        "leak": leak_scan(text, freeze),
    }


# ---------------------------------------------------------------- leak scan


MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
FULL_DATE = re.compile(r"\b(?:(\d{1,2})\s+([A-Z][a-z]+)|([A-Z][a-z]+)\s+(\d{1,2}),?)\s+(20\d\d)\b")
MARKETS = re.compile(r"\b(polymarket|kalshi|manifold|predictit|metaculus|betting odds|prediction market)\b", re.I)


def leak_scan(text: str, freeze: datetime) -> dict:
    """Full dates after T and market mentions. Text at T can name future
    scheduled dates, so a flag means read it, not a leak."""
    later = set()
    for m in FULL_DATE.finditer(text):
        day, month = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        mi = MONTHS.get(month.lower())
        if not mi:
            continue
        try:
            when = datetime(int(m.group(5)), mi, int(day), tzinfo=timezone.utc)
        except ValueError:
            continue
        if when > freeze:
            later.add(when.date().isoformat())
    return {"later_dates": sorted(later), "markets": sorted({w.lower() for w in MARKETS.findall(text)})}
