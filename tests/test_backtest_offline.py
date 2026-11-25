"""
Offline tests of the backtest: question selection, the prompt date stub, the
Metaculus block, batch protection and the scorer. No network, no model calls.

Run with:  uv run --with pytest python -m pytest tests -q
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ckai_bot  # noqa: E402,F401  (its .env load happens before the fixtures)
from backtest import fetch, run, score  # noqa: E402

TEMPLATE = "Resolves to the outcome of the question found at https://example.com/m."


def market(qid, source="polymarket", freeze="2026-08-06T00:00:00+00:00", **extra):
    q = {
        "id": qid,
        "source": source,
        "question": f"Will {qid} happen?",
        "resolution_criteria": TEMPLATE,
        "market_info_resolution_criteria": "N/A",
        "background": f"Rules for {qid}: resolves Yes if it happens. See https://example.com/x",
        "freeze_datetime": freeze,
        "freeze_datetime_value": "0.3",
        "url": "https://example.com/m",
    }
    q.update(extra)
    return q


def resolution(qid, source="polymarket", resolved=True, to=1.0):
    return {"id": qid, "source": source, "resolved": resolved, "resolved_to": to,
            "resolution_date": "2026-09-01"}


# Selection


def test_select_filters_and_joins():
    questions = [
        market("ok"),
        market("early", freeze="2026-07-23T00:00:00+00:00"),
        market("open"),
        market("price"),
        market("noRules", background="N/A"),
        market("acled1", source="acled"),
        market(["a", "b"], source="acled"),
        market("meta", source="metaculus", market_info_resolution_criteria="Real rules here."),
        market("gone"),
    ]
    resolutions = [
        resolution("ok"),
        resolution("early"),
        resolution("open", resolved=False, to=1.0),  # unresolved: resolved_to is the price
        resolution("price", to=0.37),
        resolution("noRules"),
        resolution("acled1", source="acled"),
        resolution(["a", "b"], source="acled"),
        resolution("meta", source="metaculus", to=0.0),
    ]
    chosen, dropped = fetch.select(
        [("2026-08-16", {"questions": questions}, {"resolutions": resolutions})], "2026-08-01"
    )
    assert sorted(q["id"] for q in chosen) == ["meta", "ok"]
    assert dropped["frozen_too_early"] == 1
    assert dropped["unresolved"] == 1
    assert dropped["not_0_or_1"] == 1
    assert dropped["no_criteria"] == 1
    assert dropped["not_market"] == 2
    assert dropped["no_resolution"] == 1
    by_id = {q["id"]: q for q in chosen}
    assert by_id["meta"]["criteria"] == "Real rules here."
    assert by_id["meta"]["outcome"] == 0
    assert "example.com" not in by_id["ok"]["criteria"]  # links removed
    assert by_id["ok"]["background"] == ""  # not repeated when it is the criteria
    assert by_id["ok"]["crowd"] == 0.3


def test_earliest_freeze_kept():
    first = ("2026-08-16", {"questions": [market("x")]}, {"resolutions": [resolution("x")]})
    later = (
        "2026-08-30",
        {"questions": [market("x", freeze="2026-08-20T00:00:00+00:00")]},
        {"resolutions": [resolution("x")]},
    )
    chosen, dropped = fetch.select([first, later], "2026-08-01")
    assert len(chosen) == 1 and chosen[0]["set"] == "2026-08-16"
    assert dropped["repeat"] == 1


def test_polymarket_groups_share_background():
    shared = "Who will win the event? This market resolves to the winner. " * 3
    a = market("a", background=shared)
    b = market("b", background=shared)
    c = market("c", source="manifold", background="Own text")
    assert fetch.group_for(a) == fetch.group_for(b)
    assert fetch.group_for(c) == "manifold:c"


# The run


def test_question_object_has_no_metaculus_tag():
    q = {"id": "x", "question": "Q?", "background": "B", "criteria": "C"}
    bq = run.to_question(q)
    assert bq.question_text == "Q?" and bq.resolution_criteria == "C"
    assert bq.tournament_slugs == ["backtest"]


def test_date_stub_sets_today_in_prompt(monkeypatch):
    import main

    monkeypatch.setattr(main, "datetime", main.datetime)  # restored after the test
    run.install_date_stub()
    captured = {}

    async def capture(self, question, prompt):
        captured["prompt"] = prompt

    monkeypatch.setattr(main.FallTemplateBot2026, "_binary_prompt_to_forecast", capture)
    bot = object.__new__(main.FallTemplateBot2026)

    async def go():
        run._freeze_date.set(datetime.fromisoformat("2026-08-06T00:00:00+00:00"))
        await main.FallTemplateBot2026._run_forecast_on_binary(
            bot, run.to_question({"id": "x", "question": "Q?", "background": "", "criteria": "C"}), ""
        )

    asyncio.run(go())
    assert "Today is 2026-08-06." in captured["prompt"]
    # Outside a backtest call the real date is used.
    assert main.datetime.now().year == datetime.now().year


def test_metaculus_is_blocked(monkeypatch):
    from forecasting_tools.helpers import metaculus_api, metaculus_client

    for cls in (metaculus_api.MetaculusApi, metaculus_client.MetaculusClient):
        for name, attr in list(vars(cls).items()):
            if not name.startswith("__"):
                monkeypatch.setattr(cls, name, attr)  # restored after the test
    run.block_metaculus()
    with pytest.raises(run.MetaculusBlocked):
        metaculus_api.MetaculusApi.get_question_by_url("https://www.metaculus.com/questions/1/")
    with pytest.raises(run.MetaculusBlocked):
        metaculus_client.MetaculusClient().get_question_by_url("https://www.metaculus.com/questions/1/")


def test_existing_batch_is_not_overwritten(monkeypatch, tmp_path):
    import claude_cli_llm

    import json

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"sha256": "x", "questions": []}))
    (tmp_path / "pilot").mkdir()
    monkeypatch.setattr(run, "QUESTIONS", questions)  # the real list is local data, not in the repo
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(claude_cli_llm, "refuse_in_ci", lambda: None)
    monkeypatch.setattr(run, "configure_env", lambda *a: None)
    monkeypatch.setattr(run, "block_metaculus", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run", "--batch", "pilot"])
    with pytest.raises(SystemExit, match="never overwritten"):
        run.main()


# Scoring


def scored_questions(n=40):
    qs, preds = [], {}
    for i in range(n):
        y = i % 2
        qid = f"q{i}"
        qs.append({"id": qid, "group": f"g{i // 2}", "outcome": y, "crowd": 0.5})
        # Under-confident but right: extremizing should help.
        preds[qid] = [0.6 if y else 0.4] * 5
    return qs, preds


def test_scores_and_comparisons():
    qs, preds = scored_questions()
    result = score.analyse(qs, preds, predictions=5)
    assert result["questions_scored"] == 40
    assert result["brier"]["bot_median"] == pytest.approx(0.16)
    assert result["brier"]["crowd"] == pytest.approx(0.25)
    a = result["comparisons"]["a_extremize_cv"]
    assert a["gain"] > 0.05 and a["passes"]
    # Identical predictions: fewer of them or another average changes nothing.
    for name in ("b_1_prediction", "b_3_predictions", "c_mean", "c_trimmed_mean"):
        assert result["comparisons"][name]["gain"] == pytest.approx(0)
        assert not result["comparisons"][name]["passes"]


def test_questions_with_too_few_predictions_are_left_out():
    qs, preds = scored_questions(10)
    preds["q0"] = preds["q0"][:3]
    result = score.analyse(qs, preds, predictions=5)
    assert result["questions_scored"] == 9


def test_holm():
    adjusted = score.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


def test_subsample_brier():
    assert score.subsample_brier([0.2, 0.8, 0.8], 1, 1) == pytest.approx((0.64 + 0.04 + 0.04) / 3)


def test_extremize():
    assert score.extremize(0.5, 2.0) == pytest.approx(0.5)
    assert score.extremize(0.7, 1.0) == pytest.approx(0.7)
    assert score.extremize(0.7, 2.0) > 0.8


def test_clusters_join_groups_and_resolution_dates():
    qs = [
        {"id": "a", "group": "polymarket:x", "resolution_date": "2026-11-04"},
        {"id": "b", "group": "polymarket:x", "resolution_date": "2026-11-05"},
        {"id": "c", "group": "manifold:c", "resolution_date": "2026-11-05"},
        {"id": "d", "group": "manifold:d", "resolution_date": "2026-09-01"},
    ]
    c = score.clusters(qs)
    assert c["a"] == c["b"] == c["c"]
    assert c["d"] != c["a"]


def test_short_questions_are_listed():
    qs, preds = scored_questions(10)
    preds["q3"] = preds["q3"][:2]
    assert score.analyse(qs, preds, predictions=5)["questions_with_fewer_predictions"] == ["q3"]


# Research (no AskNews calls)


def test_research_windows_end_at_freeze():
    from backtest import research

    freeze = datetime.fromisoformat("2026-08-06T00:00:00+00:00")
    w = {x["name"]: x for x in research.windows(freeze)}
    end = int(freeze.timestamp())
    assert w["latest"]["end_timestamp"] == w["knowledge"]["end_timestamp"] == end
    assert end - w["latest"]["start_timestamp"] == 24 * 3600
    assert end - w["knowledge"]["start_timestamp"] == 60 * 86400
    assert (w["latest"]["method"], w["latest"]["n_articles"]) == ("nl", 6)
    assert (w["knowledge"]["method"], w["knowledge"]["n_articles"]) == ("kw", 10)


def test_articles_after_freeze_are_found():
    from backtest import research

    freeze = datetime.fromisoformat("2026-08-06T00:00:00+00:00")
    arts = [{"pub_date": "2026-08-05T10:00:00Z"}, {"pub_date": "2026-08-07T10:00:00"}]
    assert research.later_than(arts, freeze) == [arts[1]]


def test_leak_and_market_flags():
    from backtest import research

    freeze = datetime.fromisoformat("2026-08-06T00:00:00+00:00")
    clean = research.leak_flags("In July 2026 talks stalled. Polls close in 2026.", freeze)
    assert clean["score"] == 0
    leaky = research.leak_flags("On September 3, 2026 she was elected. Plans for 2027.", freeze)
    assert leaky["later_months"] == ["september 2026"]
    assert leaky["later_years"] == ["2027"]
    assert leaky["outcome_words"] == ["was elected"]
    assert research.market_flag("Polymarket odds put it at 60%") == ["odds", "polymarket"]
    assert research.market_flag("No markets here") == []


def test_empty_research_matches_the_library_text():
    from backtest import research

    assert research.format_research([]) == "Here are the relevant news articles:\n\nNo articles were found.\n\n"


def test_research_is_fetched_once_and_reused(monkeypatch, tmp_path):
    from backtest import research

    calls = []

    async def fake_fetch(question, freeze):
        calls.append(question)
        return {"text": f"news for {question}", "calls": [], "leak": {"score": 0}, "market_words": []}

    monkeypatch.setattr(research, "fetch", fake_fetch)
    monkeypatch.setattr(research, "PAUSE_SECONDS", 0)
    q = {"id": "x", "question": "Q?", "freeze_datetime": "2026-08-06T00:00:00+00:00"}
    path = tmp_path / "research.jsonl"

    async def go(store):
        return await store.get(q), await store.get(q)

    assert asyncio.run(go(research.ResearchStore(path))) == ("news for Q?", "news for Q?")
    assert asyncio.run(go(research.ResearchStore(path)))[0] == "news for Q?"  # read back on resume
    assert calls == ["Q?"]


def test_compare_is_paired_and_matched():
    qs, base = scored_questions(20)
    better = {q["id"]: ([0.9] * 3 if q["outcome"] else [0.1] * 3) for q in qs}
    result = score.compare(qs, base, better, 3)
    assert result["questions"] == 20
    assert result["brier"]["baseline_median_of_k"] == pytest.approx(0.16)
    assert result["brier"]["compared_median"] == pytest.approx(0.01)
    assert result["gain"]["mean"] == pytest.approx(0.15) and result["passes"]
    same = score.compare(qs, base, {i: v[:3] for i, v in base.items()}, 3)
    assert same["gain"]["mean"] == pytest.approx(0) and not same["passes"]



def test_compare_reads_wikipedia_flags():
    qs, base = scored_questions(20)
    other = {i: v[:3] for i, v in base.items()}
    ids = [q["id"] for q in qs]
    research = {i: {"text": "x", "leak": {"later_dates": [], "markets": []}} for i in ids}
    research[ids[0]]["leak"]["later_dates"] = ["2026-11-03"]
    research[ids[1]]["leak"]["markets"] = ["kalshi"]
    result = score.compare(qs, base, other, 3, research)
    assert result["exploratory_by_leak_flag"]["flagged"]["n"] == 1
    assert result["exploratory_by_market_words"]["flagged"]["n"] == 1

def test_fetch_drops_articles_after_freeze(monkeypatch):
    """The filter works on the SDK's own article objects (pub_date, AnyUrl)."""
    from asknews_sdk.dto.news import SearchResponse

    from backtest import research

    def article(url, pub):
        return {"article_url": url, "pub_date": pub, "eng_title": "t", "summary": "s", "language": "en",
                "source_id": "x", "article_id": "00000000-0000-0000-0000-000000000000",
                "country": "US", "image_url": None, "title": "t", "keywords": [], "classification": "c",
                "authors": [], "entities": {}, "domain_url": url, "page_rank": 1, "provocative": "low",
                "reporting_voice": "Objective", "sentiment": 0, "continent": "Europe", "medium": "News", "as_string_key": "k"}

    class FakeNews:
        async def search_news(self, **kwargs):
            payload = {"as_dicts": [article("https://a.example/1", "2026-08-05T00:00:00Z"),
                                    article("https://a.example/2", "2026-08-09T00:00:00Z")],
                       "as_string": "", "offset": 0}
            return SearchResponse.model_validate(payload)  # real objects: AnyUrl, datetime

    class FakeSDK:
        def __init__(self, **kwargs):
            self.news = FakeNews()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    import asknews_sdk

    monkeypatch.setattr(asknews_sdk, "AsyncAskNewsSDK", FakeSDK)
    monkeypatch.setattr(research, "PAUSE_SECONDS", 0)
    monkeypatch.setattr(research, "format_research", lambda per_call: ",".join(str(len(x)) for x in per_call))
    monkeypatch.setenv("ASKNEWS_API_KEY", "test-not-a-key")
    freeze = datetime.fromisoformat("2026-08-06T00:00:00+00:00")
    result = asyncio.run(research.fetch("Q?", freeze))
    assert result["text"] == "1,1"  # one kept per call
    assert [c["dropped_after_freeze"] for c in result["calls"]] == [1, 1]


# Wikipedia research (fake API, no network, no model calls)


class FakeWikiApi:
    """Pages: title -> list of (timestamp, content), oldest first."""

    def __init__(self, pages, redirects=None):
        self.pages, self.redirects, self.requests = pages, redirects or {}, []

    def get(self, params):
        self.requests.append(params)
        if params.get("list") == "logevents":
            target = self.redirects.get(params["letitle"])
            events = [{"timestamp": "2026-09-01T00:00:00Z", "params": {"target_title": target}}] if target else []
            return {"query": {"logevents": events}}
        title = params["titles"]
        start = params["rvstart"]
        revs = [(i, ts, c) for i, (ts, c) in enumerate(self.pages.get(title, [])) if ts <= start]
        if not revs:
            return {"query": {"pages": [{"title": title, "missing": True}]}}
        i, ts, c = revs[-1]
        return {"query": {"pages": [{"title": title, "revisions": [
            {"revid": hash((title, i)) & 0xFFFFFF, "timestamp": ts, "slots": {"main": {"content": c}}}]}]}}


T = datetime.fromisoformat("2026-08-06T00:00:00+00:00")


@pytest.fixture
def wiki(monkeypatch, tmp_path):
    from backtest import wiki as w

    monkeypatch.setattr(w, "CACHE", tmp_path / "cache")
    return w


def test_wiki_revision_at_freeze(wiki):
    api = FakeWikiApi({"A": [("2026-07-01T00:00:00Z", "old"), ("2026-09-01T00:00:00Z", "new, after T")]})
    rev = wiki.article_at("A", T, api)
    assert rev["content"] == "old" and rev["timestamp"] <= "2026-08-06"


def test_wiki_later_revision_stops_the_run(wiki):
    with pytest.raises(wiki.WikiLeak):
        wiki._check({"revid": 1, "timestamp": "2026-08-07T00:00:00Z"}, T)


def test_wiki_redirect_followed_as_it_stood(wiki):
    api = FakeWikiApi(
        {"Old": [("2026-01-01T00:00:00Z", "#REDIRECT [[Then]]")],
         "Then": [("2026-01-01T00:00:00Z", "content then")],
         "Now": [("2026-01-01T00:00:00Z", "content now")]},
        redirects={"Old": "Now"},
    )
    assert wiki.article_at("Old", T, api)["content"] == "content then"


def test_wiki_moved_page_uses_todays_target_history(wiki):
    # "Renamed" was created after T as a redirect; its target's history goes back before T.
    # "Old name" was moved to "New name" after T; history moved with it. The title shown is the one at T.
    api = FakeWikiApi({"New name": [("2026-05-01T00:00:00Z", "text at T")]}, redirects={"Old name": "New name"})
    rev = wiki.article_at("Old name", T, api)
    assert rev["content"] == "text at T" and rev["title"] == "Old name"


def test_wiki_title_that_was_not_a_page_at_t_is_skipped(wiki):
    assert wiki.article_at("Created later", T, FakeWikiApi({"Elsewhere": [("2026-01-01T00:00:00Z", "x")]})) is None


def test_wiki_table_inside_infobox_and_caps(wiki):
    text = wiki.to_text("{{Infobox x|results=\n{| class=wikitable\n! A !! B\n|-\n| 1 || 2\n|}\n}}")
    assert "A\tB" in text and "1\t2" in text
    assert wiki._words(wiki._cap("a\tb\nc d e f", 3)) == 4  # 3 words plus the marker
    assert wiki._cap("a b", 5) == "a b"


def test_wiki_row_headers_are_data(wiki):
    text = wiki._table('{|\n! Pollster !! Lee\n|-\n! scope="row" | Abacus\n| 41%\n|}')
    assert text.splitlines() == ["Pollster\tLee", "Abacus\t41%"]


def test_wiki_missing_page_skipped(wiki):
    assert wiki.article_at("Nothing", T, FakeWikiApi({})) is None


def test_wiki_text_conversion(wiki):
    text = wiki.to_text(
        "{{Infobox election|election_date={{Start date|2026|10|27}}|candidate1=Ann Lee|image=x.png}}\n"
        "The vote is on {{Start date|2026|10|27}}, about {{convert|5|km}} away.<ref>Source</ref>\n"
        "{{Citation needed}}[[File:Map.png|thumb|A map]][[Category:Elections]]\n"
        "{| class=\"wikitable\"\n! Pollster !! Date !! Lee !! Kim\n|-\n| Abacus || Jul 2026 || 41% || 38%\n|}\n"
    )
    assert "election_date: 2026-10-27" in text and "candidate1: Ann Lee" in text
    assert "image" not in text
    assert "The vote is on 2026-10-27, about 5 km away." in text
    assert "Source" not in text and "Citation" not in text and "Map" not in text and "Elections" not in text
    assert "Pollster\tDate\tLee\tKim" in text and "Abacus\tJul 2026\t41%\t38%" in text


def test_wiki_sections_split(wiki):
    secs = wiki.sections("Lead text\n== Background ==\nB\n=== Polls ===\nP\n== See also ==\nS")
    assert [h for h, _ in secs] == ["Lead", "Background", "Polls", "See also"]


def test_wiki_portal_items_keep_their_chain(wiki):
    content = ("'''Armed conflicts'''\n*[[2026 Iran war]]\n**[[Strait of Hormuz]]\n"
               "***Iran and [[Oman]] reach an agreement.\n*[[Cricket]] news item.\n")
    items = wiki.portal_items(content)
    assert items[0]["chain"] == ["2026 Iran war", "Strait of Hormuz"]
    assert "Oman" in items[0]["text"] and "2026 Iran war" in items[0]["links"]
    assert items[1]["chain"] == [] and "Cricket" in items[1]["links"]


def test_wiki_leak_scan(wiki):
    flags = wiki.leak_scan("Set for October 27, 2026. Held on 3 July 2026. Polymarket odds.", T)
    assert flags["later_dates"] == ["2026-10-27"]
    assert flags["markets"] == ["polymarket"]


def test_wiki_research_end_to_end(wiki, monkeypatch):
    portal = wiki.portal_title(T - timedelta(days=1))
    api = FakeWikiApi({
        "Hamilton election": [("2026-07-30T00:00:00Z", "Lead about the vote.\n== Opinion polls ==\nLee leads.\n"
                                                       "== History ==\nOld stuff.")],
        portal: [("2026-08-05T20:00:00Z", "*[[Hamilton election]]\n**Lee holds a rally.\n*Unrelated item.")],
    })

    def portal_revisions(freeze, api_=None):
        rev = api.get({"titles": portal, "rvstart": wiki._iso(freeze)})["query"]["pages"][0]["revisions"][0]
        return {portal: {"revid": rev["revid"], "timestamp": rev["timestamp"],
                         "content": rev["slots"]["main"]["content"]}}

    monkeypatch.setattr(wiki, "portal_revisions", portal_revisions)

    class Helper:
        prompts = []

        async def invoke(self, prompt):
            self.prompts.append(prompt)
            if "article titles" in prompt:
                return '{"titles": ["Hamilton election", "Not a page"], "keywords": ["Lee", "Hamilton"]}'
            return "[]"

    q = {"id": "x", "question": "Will Lee win in Hamilton?", "criteria": "C",
         "freeze_datetime": T.isoformat()}
    row = asyncio.run(wiki.research_for(q, Helper(), api))
    assert "Wikipedia articles as they stood on 2026-08-06" in row["text"]
    assert "Lee leads." in row["text"]  # a poll section is always included
    assert "Old stuff" not in row["text"]
    assert "Lee holds a rally." in row["text"] and "Unrelated" not in row["text"]
    assert row["articles"][0]["timestamp"] == "2026-07-30T00:00:00Z"


def test_wiki_helper_runs_without_tools():
    """The title and section calls use plain mode, which the runner starts with no tools."""
    import claude_cli_llm
    from backtest import run as r

    src = Path(r.__file__).read_text()
    assert "ClaudeCliLlm(HELPER, mode=PLAIN" in src
    assert "RESEARCH" not in src.split("wiki_fetch")[0].split("wikipedia")[-1]
    assert claude_cli_llm.PLAIN != claude_cli_llm.RESEARCH


# Calibration test (PLAN-refine-round2.md, section 2)


def _synthetic(n, alpha, beta, seed=3):
    import math
    import random as rnd

    r = rnd.Random(seed)
    qs, preds = [], {}
    for i in range(n):
        q = r.uniform(0.03, 0.97)  # true probability
        y = int(r.random() < q)
        # The forecaster's p is miscalibrated so that sigmoid(alpha*logit(p)+beta) = q.
        p = 1 / (1 + math.exp(-((math.log(q / (1 - q)) - beta) / alpha)))
        qid = f"c{i}"
        qs.append({"id": qid, "group": f"g{i}", "resolution_date": f"d{i % 40}", "outcome": y, "crowd": 0.5})
        preds[qid] = [p] * 3
    return qs, preds


def test_calibration_recovers_known_parameters(monkeypatch):
    monkeypatch.setattr(score, "CAL_SEEDS", 2)
    qs, preds = _synthetic(3000, alpha=1.6, beta=0.4)
    result = score.calibrate(qs, preds, 3)
    fit = result["fitted_on_all"]["B_alpha_beta"]
    assert fit["alpha"] == pytest.approx(1.6, abs=0.25) and fit["beta"] == pytest.approx(0.4, abs=0.15)
    assert result["variants"]["B_alpha_beta"]["gain_mean_over_seeds"] > 0
    assert result["clusters"] == 40


def test_calibration_finds_nothing_on_calibrated_forecasts(monkeypatch):
    monkeypatch.setattr(score, "CAL_SEEDS", 2)
    qs, preds = _synthetic(3000, alpha=1.0, beta=0.0)
    result = score.calibrate(qs, preds, 3)
    assert result["adopted_pending_heldout"] is None
    assert abs(result["variants"]["A_alpha"]["gain_mean_over_seeds"]) < 0.003


def test_heldout_excludes_ids_groups_and_nearby_dates():
    def q(i, group, date):
        return {"id": i, "group": group, "resolution_date": date}

    existing = [q("a", "g1", "2026-11-04"), q("b", "g2", "2026-10-01")]
    new = [q("a", "gx", "2026-12-20"),        # same id
           q("c", "g1", "2026-12-20"),        # same group
           q("d", "g9", "2026-11-03"),        # a day before an existing resolution date
           q("e", "g9", "2026-11-06"),        # two days after: kept
           q("f", "g8", "2026-12-20")]        # unrelated: kept
    kept, dropped = fetch.exclude_overlap(new, existing)
    assert [x["id"] for x in kept] == ["e", "f"]
    assert dropped == {"same_id": 1, "same_group": 1, "resolution_date_within_1_day": 1}


def test_compare_reports_log_gain():
    qs, base = scored_questions(20)
    better = {q["id"]: ([0.9] * 3 if q["outcome"] else [0.1] * 3) for q in qs}
    result = score.compare(qs, base, better, 3, base_k=3)
    assert result["log_gain"]["mean"] > 0
    same = score.compare(qs, base, {i: v[:3] for i, v in base.items()}, 3, base_k=3)
    assert same["log_gain"]["mean"] == pytest.approx(0)


def test_ensemble_is_median_of_model_medians():
    a = {"x": [0.1, 0.2, 0.3], "y": [0.5, 0.5, 0.5]}
    b = {"x": [0.6, 0.7, 0.8], "y": [0.5]}
    c = {"x": [0.9, 0.9, 0.9], "y": [0.4, 0.4, 0.4]}
    out = score.ensemble([a, b, c], 3)
    assert out == {"x": [0.2, 0.7, 0.9]}  # y left out: b has too few predictions


def test_resume_refuses_changed_round3_settings():
    from backtest import run

    def ns(**kw):
        base = dict(forecaster="claude-cli/claude-opus-5-5", effort="high", prompt="template")
        base.update(kw)
        return argparse.Namespace(**base)

    old = "forecaster: claude-cli/claude-opus-5-5\nresearch: wikipedia\n"  # a batch from before round 3
    assert run.check_resume_settings(old, ns()) is None
    assert "effort" in run.check_resume_settings(old, ns(effort="max"))
    assert "prompt" in run.check_resume_settings(old, ns(prompt="v2"))
    sha = run._sha256(run.ROOT / "prompts_v2.py")
    new = ("forecaster: claude-cli/claude-fable-5-2\neffort: max (forecaster, claude --effort)\n"
           f"prompt_version: v2\nprompt_sha256: {sha}\n")
    assert run.check_resume_settings(new, ns(forecaster="claude-cli/claude-fable-5-2", effort="max", prompt="v2")) is None
    changed = new.replace(sha, "0" * 64)
    assert "changed" in run.check_resume_settings(changed, ns(forecaster="claude-cli/claude-fable-5-2", effort="max", prompt="v2"))
    assert "forecaster" in run.check_resume_settings(new, ns(effort="max", prompt="v2"))
    assert run.round3(ns(effort="max")) and not run.round3(ns())


def test_configure_env_sets_effort_for_forecaster_only(monkeypatch):
    import claude_cli_llm
    from backtest import run

    monkeypatch.setattr(claude_cli_llm, "PLAIN_EFFORT", {})
    monkeypatch.setattr(claude_cli_llm, "PLAIN_TIMEOUT", 600)
    monkeypatch.setattr(run, "FORECASTER", "claude-cli/claude-sonnet-5-5")
    for name in ("FORECASTER_MODELS", "RESEARCHER_MODEL", "SUMMARIZER_MODEL", "PARSER_MODEL", "PROMPT_VERSION"):
        monkeypatch.delenv(name, raising=False)
    run.configure_env("max", "v2")
    assert claude_cli_llm.PLAIN_EFFORT == {"claude-sonnet-5-5": "max"}
    assert claude_cli_llm.PLAIN_TIMEOUT == 25 * 60
    assert os.environ["PROMPT_VERSION"] == "v2"
    assert os.environ["FORECASTER_MODELS"] == "claude-cli/claude-sonnet-5-5"
