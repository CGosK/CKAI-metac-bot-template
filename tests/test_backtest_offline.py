"""
Offline tests of the backtest: question selection, the prompt date stub, the
Metaculus block, batch protection and the scorer. No network, no model calls.

Run with:  uv run --with pytest python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
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

    (tmp_path / "pilot").mkdir()
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(claude_cli_llm, "refuse_in_ci", lambda: None)
    monkeypatch.setattr(run, "configure_env", lambda: None)
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
