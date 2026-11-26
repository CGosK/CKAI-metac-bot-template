"""
Dated current values for a question, as they stood before its freeze time T
(round 5, benchmarks/PLAN-refine-round5.md; not wired into the live bot).

Each fetcher returns a Series: dated (date, value) rows with every row dated
strictly before T, plus a note on what the cut can and can't guarantee.
`format_block` turns one or more Series into a short, deterministic text
block for the forecaster ("Latest values before T: ...").

Sources:
- FRED through ALFRED (key FRED_API_KEY, read from the environment at call time,
  never written or logged): observations as published on the day before T
  (realtime_start = realtime_end = T - 1 day), so revisions and releases after T
  can't leak;
- World Bank Open Data API (annual indicators by country, keyless);
- US Treasury daily par yield curve CSV (one file per year);
- DBnomics API (ECB, Eurostat, IMF, national statistics, Meteo-France, ...).

What the cut means. These APIs serve today's values, not values as they were
published at T. Cutting by observation date stops later observations from
leaking, but not (a) revisions made after T to earlier observations, or
(b) observations dated before T that were only published after T. `lag_days`
handles (b) roughly: rows must be dated before T minus the lag. For revising
macro series, ALFRED vintages (FRED key needed) are the proper fix.

DBnomics rows are cut with a conservative publication lag per frequency
(DBNOMICS_LAG_DAYS, counted from the period's start date), since DBnomics gives
no release dates.

No model calls. Requests are one at a time with a gap and a public User-Agent;
responses are cached on disk by URL (the key is never part of the cache key or
any message).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

USER_AGENT = "CKAI-metac-bot/0.1 (https://github.com/CGosK/CKAI-metac-bot-template)"
GAP_SECONDS = 2.0
WORLD_BANK = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
TREASURY = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
            "&field_tdr_date_value={year}&page&_format=csv")
DBNOMICS = "https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}"
FRED_OBS = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = "https://api.stlouisfed.org/fred/series"
# Days from a period's start date before its value is assumed published (plan, Step 0).
DBNOMICS_LAG_DAYS = {"daily": 2, "weekly": 14, "monthly": 60, "quarterly": 120, "annual": 400}
# Datasets published later than their frequency suggests (code review, Nov 26): the larger lag applies.
# ECB CISS is a daily index released weekly.
DBNOMICS_DATASET_LAG_DAYS = {("ECB", "CISS"): 14}
CACHE_DIR = Path(os.environ.get("DATA_RESEARCH_CACHE", Path.home() / ".cache" / "ckai-bot-data"))
_KEY_PARAM = re.compile(r"(api_key=)[^&\s]+")
TREASURY_TENORS = ("1 Mo", "1.5 Month", "2 Mo", "3 Mo", "4 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr",
                   "7 Yr", "10 Yr", "20 Yr", "30 Yr")

Http = Callable[[str], bytes]


class DataLeak(RuntimeError):
    """A row dated at or after T reached the formatter: stop."""


class DataUnavailable(RuntimeError):
    """The source refused, failed, or has no rows before T."""


class DataTransient(DataUnavailable):
    """A network or server failure (timeout, 5xx, 429): retry later; never stored as "no data"."""


class DataAuthError(DataUnavailable):
    """The source refused the key (FRED): stop, nothing else will work."""


def redact(text: str) -> str:
    """Remove an api_key value from a URL or message."""
    return _KEY_PARAM.sub(r"\1REDACTED", text)


@dataclass
class Series:
    source: str
    name: str
    units: str
    rows: list[tuple[date, float]]  # sorted by date, every date < cutoff
    cutoff: date  # rows are strictly before this date
    vintage: str  # "observation-date cut" or "vintage"
    notes: list[str] = field(default_factory=list)
    url: str = ""


class PoliteHttp:
    """One request at a time, GAP_SECONDS apart, public User-Agent. With `cache`, a
    response already fetched for the same URL (key removed) is read from disk."""

    def __init__(self, gap: float = GAP_SECONDS, timeout: float = 60, cache: Path | None = None):
        self.gap, self.timeout, self._last, self.cache = gap, timeout, 0.0, cache
        self.requests = self.cache_hits = 0

    def _cache_path(self, url: str) -> Path | None:
        if self.cache is None:
            return None
        return self.cache / (hashlib.sha256(redact(url).encode()).hexdigest() + ".bin")

    def __call__(self, url: str) -> bytes:
        path = self._cache_path(url)
        if path is not None and path.exists():
            self.cache_hits += 1
            return path.read_bytes()
        wait = self._last + self.gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        self.requests += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            cls = (DataAuthError if "api_key" in detail.lower() or exc.code in (401, 403)
                   else DataTransient if exc.code == 429 or exc.code >= 500 else DataUnavailable)
            raise cls(redact(f"{url}: HTTP {exc.code}: {detail}")) from None
        except Exception as exc:
            raise DataTransient(redact(f"{url}: {exc}")) from None
        finally:
            self._last = time.monotonic()
        if body.lstrip()[:1] == b"<":  # an HTML error or maintenance page served with 200: not data, not cached
            raise DataTransient(redact(f"{url}: HTML page instead of data"))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return body


def _as_date(t: datetime | date) -> date:
    return t.date() if isinstance(t, datetime) else t


def _cut(rows: list[tuple[date, float]], freeze: datetime | date, lag_days: int) -> tuple[list, date]:
    cutoff = _as_date(freeze) - timedelta(days=lag_days)
    kept = sorted((d, v) for d, v in rows if d < cutoff)
    return kept, cutoff


# ---------------------------------------------------------------- fetchers


def world_bank(indicator: str, country: str, freeze: datetime | date, http: Http,
               lag_days: int = 180) -> Series:
    """Annual World Bank indicator. A year counts as observed on Dec 31; with the
    default lag a year is used only if it ended 180+ days before T, because
    annual figures are published months later. No vintages: values are today's."""
    url = WORLD_BANK.format(country=urllib.parse.quote(country), indicator=urllib.parse.quote(indicator))
    url += "?" + urllib.parse.urlencode({"format": "json", "per_page": 100, "date": f"1960:{_as_date(freeze).year}"})
    payload = json.loads(http(url))
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        message = payload[0].get("message") if isinstance(payload, list) and payload else payload
        raise DataUnavailable(f"World Bank {indicator}/{country}: {message}")
    meta, records = payload[0], payload[1]
    rows = [(date(int(r["date"]), 12, 31), float(r["value"])) for r in records
            if r.get("value") is not None and str(r.get("date", "")).isdigit()]
    kept, cutoff = _cut(rows, freeze, lag_days)
    if not kept:
        raise DataUnavailable(f"World Bank {indicator}/{country}: no values before {cutoff}")
    first = records[0]
    name = f"{first['indicator']['value']}, {first['country']['value']}"
    notes = [f"annual; a year is dated Dec 31 and kept only if it ended {lag_days}+ days before T",
             f"today's revised values (dataset last updated {meta.get('lastupdated', '?')}), not a vintage"]
    return Series("World Bank", name, first.get("unit") or "", kept, cutoff, "observation-date cut", notes, url)


def treasury_yield(tenor: str, freeze: datetime | date, http: Http, years: int = 2) -> Series:
    """US Treasury daily par yield (percent) for one tenor, e.g. "10 Yr". Reads the
    CSV for T's year and the years before it. Published the same afternoon,
    rarely corrected, so an observation-date cut is close to a vintage."""
    if tenor not in TREASURY_TENORS:
        raise ValueError(f"unknown tenor {tenor!r}; one of {TREASURY_TENORS}")
    t = _as_date(freeze)
    rows: list[tuple[date, float]] = []
    urls = []
    for year in range(t.year - years + 1, t.year + 1):
        url = TREASURY.format(year=year)
        urls.append(url)
        text = http(url).decode("utf-8-sig")
        for record in csv.DictReader(io.StringIO(text)):
            value = (record.get(tenor) or "").strip()
            if not value:
                continue
            month, day, yr = record["Date"].split("/")
            rows.append((date(int(yr), int(month), int(day)), float(value)))
    kept, cutoff = _cut(rows, freeze, 0)
    if not kept:
        raise DataUnavailable(f"Treasury {tenor}: no values before {cutoff}")
    notes = ["daily, business days; published after the close on the same day"]
    return Series("US Treasury", f"{tenor} par yield curve rate", "percent", kept, cutoff,
                  "observation-date cut", notes, urls[-1])


def dbnomics(provider: str, dataset: str, series: str, freeze: datetime | date, http: Http,
             lag_days: int | None = None) -> Series:
    """One DBnomics series. Dates are period start days; "NA" values are skipped.
    DBnomics mirrors the provider's current data, so revision risk is the provider's.
    lag_days None: the plan's lag for the series' frequency (DBNOMICS_LAG_DAYS)."""
    url = DBNOMICS.format(provider=urllib.parse.quote(provider), dataset=urllib.parse.quote(dataset),
                          series=urllib.parse.quote(series))
    url += "?" + urllib.parse.urlencode({"observations": 1, "metadata": "false", "format": "json"})
    payload = json.loads(http(url))
    docs = (payload.get("series") or {}).get("docs") or []
    if not docs:
        raise DataUnavailable(f"DBnomics {provider}/{dataset}/{series}: {payload.get('errors') or 'not found'}")
    doc = docs[0]
    rows = dbnomics_rows(doc)
    if lag_days is None:
        frequency = doc.get("@frequency", "")
        if frequency not in DBNOMICS_LAG_DAYS:
            raise DataUnavailable(f"DBnomics {provider}/{dataset}/{series}: no lag rule for frequency {frequency!r}")
        lag_days = max(DBNOMICS_LAG_DAYS[frequency], DBNOMICS_DATASET_LAG_DAYS.get((provider, dataset), 0))
    kept, cutoff = _cut(rows, freeze, lag_days)
    if not kept:
        raise DataUnavailable(f"DBnomics {provider}/{dataset}/{series}: no values before {cutoff}")
    notes = [f"{doc.get('@frequency', '?')}; dated by period start",
             f"provider's current values (indexed {doc.get('indexed_at', '?')[:10]}), not a vintage"]
    if lag_days:
        notes.append(f"rows kept only if dated {lag_days}+ days before T")
    return Series("DBnomics", doc.get("series_name") or series, "", kept, cutoff, "observation-date cut", notes, url)


def dbnomics_rows(doc: dict) -> list[tuple[date, float]]:
    """Every numeric (period start day, value) row of a DBnomics series doc, uncut."""
    return sorted((date.fromisoformat(day), float(value))
                  for day, value in zip(doc.get("period_start_day", []), doc.get("value", []))
                  if isinstance(value, (int, float)))


def dbnomics_doc(provider: str, dataset: str, series: str, http: Http) -> dict:
    url = DBNOMICS.format(provider=urllib.parse.quote(provider), dataset=urllib.parse.quote(dataset),
                          series=urllib.parse.quote(series))
    url += "?" + urllib.parse.urlencode({"observations": 1, "metadata": "false", "format": "json"})
    docs = (json.loads(http(url)).get("series") or {}).get("docs") or []
    if not docs:
        raise DataUnavailable(f"DBnomics {provider}/{dataset}/{series}: not found")
    return docs[0]


def fred_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise DataAuthError("FRED_API_KEY is not set in the environment")
    return key


FRED_HISTORY_DAYS = 3 * 366  # observations fetched back this far from the vintage


def fred_vintage_rows(series_id: str, vintage: date | None, http: Http,
                      start: date | None = None) -> list[tuple[date, float]]:
    """ALFRED: the series as published on `vintage` (realtime_start = realtime_end), uncut by date;
    vintage None gives today's values (for resolving self-made questions after the fact)."""
    params = {"series_id": series_id, "file_type": "json", "api_key": fred_key()}
    if vintage is not None:
        params.update(realtime_start=vintage.isoformat(), realtime_end=vintage.isoformat())
    first = start or ((vintage or date.today()) - timedelta(days=FRED_HISTORY_DAYS))
    params["observation_start"] = first.isoformat()
    payload = json.loads(http(FRED_OBS + "?" + urllib.parse.urlencode(params)))
    if "error_message" in payload:
        raise DataUnavailable(f"FRED {series_id}: {redact(str(payload['error_message']))}")
    return sorted((date.fromisoformat(o["date"]), float(o["value"]))
                  for o in payload.get("observations", []) if o.get("value") not in (".", "", None))


def fred_info(series_id: str, http: Http) -> dict:
    params = {"series_id": series_id, "file_type": "json", "api_key": fred_key()}
    payload = json.loads(http(FRED_SERIES + "?" + urllib.parse.urlencode(params)))
    seriess = payload.get("seriess") or []
    if not seriess:
        raise DataUnavailable(f"FRED {series_id}: {redact(str(payload.get('error_message', 'not found')))}")
    return seriess[0]


def fred_vintage(freeze: datetime | date) -> date:
    """ALFRED's realtime dates are US Central days, and a vintage holds what was posted by the end of that day.
    The last full Central day before T: a 00:00 UTC freeze is 18:00-19:00 the previous Central day, so the
    vintage is two calendar days before T's UTC date (an evening posting on T-1 Central is after T)."""
    if isinstance(freeze, datetime) and freeze.tzinfo is not None:
        from zoneinfo import ZoneInfo

        return freeze.astimezone(ZoneInfo("America/Chicago")).date() - timedelta(days=1)
    return _as_date(freeze) - timedelta(days=2)  # no time zone: assume 00:00 UTC, as ForecastBench freezes


def fred(series_id: str, freeze: datetime | date, http: Http) -> Series:
    """A FRED series as it was published by the last full US Central day before T (ALFRED), rows dated
    before T."""
    vintage = fred_vintage(freeze)
    info = fred_info(series_id, http)
    kept, cutoff = _cut(fred_vintage_rows(series_id, vintage, http), freeze, 0)
    if not kept:
        raise DataUnavailable(f"FRED {series_id}: no values published by {vintage}")
    notes = [f"{info.get('frequency', '?')}", f"values as published on {vintage.isoformat()} (ALFRED vintage)"]
    return Series("FRED", info.get("title") or series_id, info.get("units") or "", kept, cutoff,
                  "vintage", notes, f"{FRED_OBS}?series_id={series_id}")


def dbnomics_id(forecastbench_id: str) -> tuple[str, str, str]:
    """ForecastBench's dbnomics id 'provider_dataset_series' (series may hold underscores)."""
    provider, dataset, series = forecastbench_id.split("_", 2)
    return provider, dataset, series


def series_for(source: str, series_id: str, freeze: datetime | date, http: Http) -> Series:
    """Registry from a ForecastBench source to its fetcher."""
    if source == "fred":
        return fred(series_id, freeze, http)
    if source == "dbnomics":
        return dbnomics(*dbnomics_id(series_id), freeze, http)
    raise DataUnavailable(f"no fetcher for source {source!r}")


# ---------------------------------------------------------------- formatter


def _num(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-3:
        return f"{x:,.6g}" if abs(x) >= 1e6 else f"{x:.4g}"
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def _days(delta: timedelta) -> str:
    return "1 day" if delta.days == 1 else f"{delta.days} days"


def _summary(s: Series, freeze: date, recent: int) -> list[str]:
    for d, _ in s.rows:
        if d >= freeze or d >= s.cutoff:
            raise DataLeak(f"{s.source} {s.name}: row dated {d} is not before {min(freeze, s.cutoff)}")
    last_d, last_v = s.rows[-1]
    units = f" {s.units}" if s.units else ""
    lines = [f"- {s.name} ({s.source}{', ' + s.units if s.units else ''}): latest {_num(last_v)}{units} "
             f"on {last_d.isoformat()} ({_days(freeze - last_d)} before T)."]
    year = [v for d, v in s.rows if d >= last_d - timedelta(days=365)]
    if len(year) >= 3:
        lines.append(f"  Past 12 months: low {_num(min(year))}, high {_num(max(year))}, {len(year)} observations.")
    earlier = [(d, v) for d, v in s.rows if d <= last_d - timedelta(days=365)]
    if earlier:
        d0, v0 = earlier[-1]
        lines.append(f"  A year earlier: {_num(v0)} on {d0.isoformat()}.")
    tail = s.rows[-recent:]
    lines.append("  Recent (date, value): " + ", ".join(f"({d.isoformat()}, {_num(v)})" for d, v in tail))
    lines.append("  Note: " + "; ".join(s.notes) + ".")
    return lines


def format_block(series: list[Series], freeze: datetime | date, recent: int = 8) -> str:
    """Short text block for the forecaster. Raises DataLeak if any row is not
    strictly before T. An empty list gives an explicit 'no data' line, so the
    forecaster is told the value is missing rather than left to guess."""
    t = _as_date(freeze)
    head = f"Latest values before {t.isoformat()} (T). Values on or after T are not shown."
    if not series:
        return head + "\n- No data series was found for this question. Do not assume a current value."
    lines = [head]
    for s in series:
        lines.extend(_summary(s, t, recent))
    return "\n".join(lines)
