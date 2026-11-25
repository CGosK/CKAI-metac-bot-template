"""Offline tests of GDELT headline research (fake API, no network, no model calls)."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gdelt_research as g  # noqa: E402

T = datetime.fromisoformat("2026-09-23T00:00:00+00:00")


def art(seen, title, domain="news.example"):
    return {"seendate": seen, "title": title, "domain": domain, "url": f"https://{domain}/x"}


class FakeApi:
    def __init__(self, by_window):
        self.by_window, self.calls = by_window, []

    def get(self, params):
        self.calls.append(params)
        return self.by_window.get(params["startdatetime"], [])


def test_sub_windows_cover_the_lookback():
    w = g.sub_windows(T, 27)
    assert len(w) == 3 and w[0][0] == datetime.fromisoformat("2026-08-27T00:00:00+00:00") and w[-1][1] == T
    assert all(a[1] == b[0] for a, b in zip(w, w[1:]))


def test_headlines_drop_after_freeze_dedupe_and_sort():
    starts = [g._stamp(s) for s, _ in g.sub_windows(T, 30)]
    api = FakeApi({
        starts[0]: [art("20260825T100000Z", "Old story")],
        starts[2]: [art("20260922T235959Z", "Late story"), art("20260923T000000Z", "At T: dropped"),
                    art("20260924T010000Z", "After T: dropped"), art("20260921T080000Z", "Late  story!")],
    })
    found = g.headlines(["\"Lee\""], T, 30, api)
    assert [a["title"] for a in found["articles"]] == ["Late story", "Old story"]
    assert found["dropped_after_freeze"] == 2 and found["returned"] == 5
    assert all(c["enddatetime"] <= "20260923000000" for c in api.calls)
    assert all("sourcelang:english" in c["query"] for c in api.calls)


def test_format_and_nothing_found():
    rows = [{"seen": "20260922T120000Z", "domain": "a.example", "title": "Lee leads"}]
    text = g.format_headlines(rows, T, 30)
    assert text.startswith("News headlines from 2026-08-24 to 2026-09-23 (GDELT):")
    assert "- 2026-09-22, a.example: Lee leads" in text
    assert g.format_headlines([], T, 30) == g.NO_ARTICLES


class Replies(g.Api):
    """A real Api whose network replies are scripted."""

    def __init__(self, replies, tmp_path):
        super().__init__(cache=tmp_path, gap=0, throttle_waits=(0, 0))
        self.replies = list(replies)

    def _fetch(self, url):
        return self.replies.pop(0)


def test_text_reply_is_an_error_not_no_articles(tmp_path):
    api = Replies([(200, "Your search contained a keyword that was too short.")], tmp_path)
    with pytest.raises(g.GdeltQueryError):
        api.get({"query": "ab"})


def test_throttling_waits_then_stops(tmp_path):
    notice = (200, "Please limit requests to one every 5 seconds")
    api = Replies([notice, (429, ""), (200, json.dumps({"articles": [art("20260920T000000Z", "x")]}))], tmp_path)
    assert len(api.get({"query": "lee"})) == 1 and api.requests == 3
    api = Replies([notice, notice, notice], tmp_path)
    with pytest.raises(g.GdeltStopped):
        api.get({"query": "other"})


def test_empty_json_is_no_articles_and_cached(tmp_path):
    api = Replies([(200, "{}")], tmp_path)
    assert api.get({"query": "lee"}) == []
    assert api.get({"query": "lee"}) == [] and api.requests == 1  # second read from the cache


def test_empty_body_is_an_error_and_not_cached(tmp_path):
    api = Replies([(200, ""), (200, "{}")], tmp_path)
    with pytest.raises(g.GdeltQueryError):
        api.get({"query": "lee"})
    assert api.get({"query": "lee"}) == [] and api.requests == 2


def test_headline_saying_rate_limit_is_not_throttling(tmp_path):
    body = json.dumps({"articles": [art("20260920T000000Z", "Fed signals rate limit on hikes")]})
    api = Replies([(200, body)], tmp_path)
    assert len(api.get({"query": "fed"})) == 1 and api.requests == 1


def test_network_error_and_5xx_are_waited_out(tmp_path):
    api = Replies([(0, "network error"), (503, ""), (200, "{}")], tmp_path)
    assert api.get({"query": "lee"}) == [] and api.requests == 3


def test_research_for_end_to_end_with_helper_without_tools(tmp_path):
    starts = [g._stamp(s) for s, _ in g.sub_windows(T, 30)]
    api = FakeApi({starts[1]: [art("20260910T000000Z", "Lee gains in polls")]})

    class Helper:
        async def invoke(self, prompt):
            assert "Question: Will Lee win?" in prompt
            return '{"queries": ["\\"Lee\\" election", "Hamilton vote"]}'

    q = {"id": "x", "question": "Will Lee win?", "criteria": "C", "freeze_datetime": T.isoformat()}
    row = asyncio.run(g.research_for(q, Helper(), 30, api))
    assert row["queries"] == ['"Lee" election'][: g.MAX_QUERIES]
    assert len(api.calls) == 3 * g.MAX_QUERIES and "Lee gains in polls" in row["text"]
    assert row["leak"] == {"later_dates": [], "markets": []}


def test_gdelt_helper_runs_without_tools():
    from backtest import run as r

    src = Path(r.__file__).read_text()
    gdelt_part = src.split('research_mode == "gdelt"')[1].split("elif")[0]
    assert "mode=PLAIN" in gdelt_part and "RESEARCH" not in gdelt_part


def test_joined_row_leaves_out_empty_texts():
    from backtest import run as r

    wiki = {"q": {"text": "Wikipedia text"}, "n": {"text": g.NO_ARTICLES}}
    gd = {"q": {"text": "Headlines"}, "n": {"text": g.NO_ARTICLES}}
    assert r.joined_row("q", [wiki, gd])["text"] == "Wikipedia text\n\nHeadlines"
    assert r.joined_row("n", [wiki, gd])["text"] == g.NO_ARTICLES
    assert r.joined_row("q", [wiki])["text"] == "Wikipedia text"
    with pytest.raises(ValueError):
        r.joined_row("missing", [wiki])


def test_clusters_join_nearby_resolution_dates():
    from backtest import score

    qs = [{"id": "a", "group": "ga", "resolution_date": "2026-11-03"},
          {"id": "b", "group": "gb", "resolution_date": "2026-11-04"},
          {"id": "c", "group": "gc", "resolution_date": "2026-11-09"}]
    plain = score.clusters(qs)
    joined = score.clusters(qs, join_days=1)
    assert plain["a"] != plain["b"]
    assert joined["a"] == joined["b"] != joined["c"]


def test_clusters_chain_three_dates():
    from backtest import score

    qs = [{"id": c, "group": "g" + c, "resolution_date": d}
          for c, d in (("a", "2026-11-03"), ("b", "2026-11-04T12:00:00Z"), ("c", "2026-11-05"), ("e", "2026-11-04"))]
    joined = score.clusters(qs, join_days=1)
    assert joined["a"] == joined["b"] == joined["c"] == joined["e"]
    assert len(set(score.clusters(qs).values())) == 4
