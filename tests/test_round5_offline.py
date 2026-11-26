"""Offline tests for round 5 (benchmarks/PLAN-refine-round5.md): fake HTTP, no network, no model calls."""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_research as dr  # noqa: E402
from backtest import fetch, round5, run, score  # noqa: E402

T = datetime(2026, 8, 6, tzinfo=timezone.utc)
FAKE_KEY = "abcdef0123456789abcdef0123456789"


class FakeHttp:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        return self.handler(url)


def fred_handler(obs: list[tuple[str, str]], title="Test series", units="Percent"):
    def handle(url: str) -> bytes:
        if url.startswith(dr.FRED_OBS):
            return json.dumps({"observations": [{"date": d, "value": v, "realtime_start": "x", "realtime_end": "x"}
                                                for d, v in obs]}).encode()
        if url.startswith(dr.FRED_SERIES):
            return json.dumps({"seriess": [{"id": "X", "title": title, "units": units,
                                            "frequency": "Daily"}]}).encode()
        raise dr.DataUnavailable(url)
    return handle


# ---------------------------------------------------------------- data_research


def test_redact_removes_key_from_urls_and_messages():
    url = f"https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key={FAKE_KEY}&file_type=json"
    assert FAKE_KEY not in dr.redact(url) and "api_key=REDACTED&file_type" in dr.redact(url)


def test_fred_uses_vintage_day_before_t_and_cuts(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", FAKE_KEY)
    http = FakeHttp(fred_handler([("2026-08-03", "4.1"), ("2026-08-04", "."), ("2026-08-05", "4.2"),
                                  ("2026-08-06", "9.9")]))
    s = dr.fred("DGS10", T, http)
    obs_url = next(u for u in http.calls if u.startswith(dr.FRED_OBS))
    assert "realtime_start=2026-08-05" in obs_url and "realtime_end=2026-08-05" in obs_url
    assert "observation_start=" in obs_url
    assert s.rows == [(date(2026, 8, 3), 4.1), (date(2026, 8, 5), 4.2)]  # "." skipped, T's row cut
    assert s.vintage == "vintage" and s.units == "Percent"


def test_fred_without_key_is_an_auth_stop(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(dr.DataAuthError):
        dr.fred("DGS10", T, FakeHttp(fred_handler([])))


def test_polite_http_caches_without_key_and_redacts_errors(monkeypatch, tmp_path):
    served = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        served.append(request.full_url)
        if "bad" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 400, "Bad", {},
                                         io.BytesIO(b'{"error_message":"Bad Request. The value for variable api_key is not registered."}'))
        return Response(b"payload")

    monkeypatch.setattr(dr.urllib.request, "urlopen", fake_urlopen)
    http = dr.PoliteHttp(gap=0, cache=tmp_path)
    url = f"https://example.org/x?api_key={FAKE_KEY}"
    assert http(url) == b"payload" and http(url) == b"payload"
    assert len(served) == 1 and http.cache_hits == 1
    assert all(FAKE_KEY not in p.name for p in tmp_path.iterdir())
    with pytest.raises(dr.DataAuthError) as caught:
        http(f"https://example.org/bad?api_key={FAKE_KEY}")
    assert FAKE_KEY not in str(caught.value) and "REDACTED" in str(caught.value)


def db_doc(freq: str, days: list[str], values: list) -> bytes:
    return json.dumps({"series": {"docs": [{"@frequency": freq, "period_start_day": days, "value": values,
                                            "series_name": "S", "indexed_at": "2026-11-26T00:00:00Z"}]}}).encode()


@pytest.mark.parametrize("freq,lag", sorted(dr.DBNOMICS_LAG_DAYS.items()))
def test_dbnomics_lag_follows_frequency(freq, lag):
    cutoff = T.date() - timedelta(days=lag)
    days = [(cutoff - timedelta(days=1)).isoformat(), cutoff.isoformat()]
    s = dr.dbnomics("P", "D", "S", T, FakeHttp(lambda u: db_doc(freq, days, [1.0, 2.0])))
    assert [d for d, _ in s.rows] == [cutoff - timedelta(days=1)]


def test_dbnomics_unknown_frequency_refuses():
    with pytest.raises(dr.DataUnavailable):
        dr.dbnomics("P", "D", "S", T, FakeHttp(lambda u: db_doc("hourly", ["2026-01-01"], [1.0])))


def test_dbnomics_id_and_registry(monkeypatch):
    assert dr.dbnomics_id("meteofrance_TEMPERATURE_celsius.07117.D") == ("meteofrance", "TEMPERATURE", "celsius.07117.D")
    assert dr.dbnomics_id("ECB_EXR_D.USD_X.EUR") == ("ECB", "EXR", "D.USD_X.EUR")
    with pytest.raises(dr.DataUnavailable):
        dr.series_for("yfinance", "AAPL", T, FakeHttp(lambda u: b""))


# ---------------------------------------------------------------- fetch --datasets


def fb_sets():
    def qset(due, freeze, value="4.2"):
        return {"forecast_due_date": due, "questions": [
            {"id": "DGS10", "source": "fred", "freeze_datetime": freeze, "freeze_datetime_value": value,
             "freeze_datetime_value_explanation": "The latest value.",
             "question": "Will DGS10 have increased by {resolution_date} as compared to its value on {forecast_due_date}?",
             "background": "See https://fred.stlouisfed.org/series/DGS10", "resolution_criteria": "Resolves at https://x"},
            {"id": "AAPL", "source": "yfinance", "freeze_datetime": freeze, "question": "?"}]}

    def rset(dates):
        return {"resolutions": [{"id": "DGS10", "source": "fred", "direction": None, "resolution_date": d,
                                 "resolved_to": 1.0, "resolved": True} for d in dates]}

    return [("2026-08-16", qset("2026-08-16", "2026-08-06T00:00:00+00:00"), rset(["2026-08-23", "2026-09-15"])),
            ("2026-08-30", qset("2026-08-30", "2026-08-06T00:00:00+00:00"), rset(["2026-09-06"])),  # same freeze
            ("2026-09-13", qset("2026-09-13", "2026-09-03T00:00:00+00:00", "N/A"), rset(["2026-09-20"]))]


def test_select_datasets_one_per_series_and_freeze():
    items, dropped = fetch.select_datasets(fb_sets(), "2026-08-01")
    assert len(items) == 1 and dropped["repeat_freeze"] == 1 and dropped["no_freeze_value"] == 1
    item = items[0]
    assert item["resolution_date"] in ("2026-08-23", "2026-09-15")
    assert "{" not in item["question"] and "2026-08-16" in item["question"] and item["resolution_date"] in item["question"]
    assert item["cluster"] == f"fred:2026-08-06:{item['resolution_date']}"
    assert "https://" not in item["background"] + item["criteria"]
    assert item["crowd"] is None and item["freeze_value"] == 4.2
    assert fetch.select_datasets(fb_sets(), "2026-08-01")[0] == items  # the seeded pick is stable


# ---------------------------------------------------------------- F1, F2, X3


def item(**kw):
    base = {"id": "i1", "source": "fred", "series_id": "DGS10", "freeze_datetime": "2026-08-06T00:00:00+00:00",
            "forecast_due_date": "2026-08-16", "resolution_date": "2026-09-15", "horizon_days": 30,
            "freeze_value": 4.2, "cluster": "c", "group": "g", "background": "", "criteria": ""}
    base.update(kw)
    return base


def test_f1_agree_disagree_and_later(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", FAKE_KEY)
    agree = round5.f1_item(item(), FakeHttp(fred_handler([("2026-08-04", "4.1"), ("2026-08-05", "4.2")])))
    assert agree["status"] == "agree" and agree["our_date"] == "2026-08-05"
    later = round5.f1_item(item(), FakeHttp(fred_handler([("2026-08-04", "4.2"), ("2026-08-05", "4.3")])))
    assert later["status"] == "later" and later["our_minus_fb_days"] == 1
    off = round5.f1_item(item(), FakeHttp(fred_handler([("2026-08-05", "5.0")])))
    assert off["status"] == "disagree" and off["fb_value_date"] is None
    summary = round5.f1_summary([agree, later, off])
    assert summary["all"]["agree"] == 1 and summary["all"]["later_than_fb"] == 1 and not summary["passes"]


def test_f1_precision_follows_reference():
    assert round5.decimals(4.2) == 1 and round5.decimals(910776.0) == 0 and round5.decimals(0.125) == 3
    assert round5.same_value(4.24, 4.2, 1) and not round5.same_value(4.26, 4.2, 1)


def test_p_up_and_weather_window():
    start = date(2024, 1, 1)
    rising = [(start + timedelta(days=i), float(i)) for i in range(900)]
    assert round5.p_up(rising, date(2026, 8, 16), 7, weather=False) == 1.0
    falling = [(d, -v) for d, v in rising]
    assert round5.p_up(falling, date(2026, 8, 16), 7, weather=False) == 0.0
    # Weather: only changes starting near the due date's calendar day in past years count.
    seasonal = [(start + timedelta(days=i), (start + timedelta(days=i)).timetuple().tm_yday) for i in range(940)]
    assert round5.p_up(seasonal, date(2026, 8, 16), 7, weather=True) == 1.0
    assert round5.p_up(seasonal[:5], date(2026, 8, 16), 7, weather=False) is None


def test_x3_question_threshold_and_resolution():
    rows = [(date(2025, 1, 1) + timedelta(days=i), float(i % 7)) for i in range(500)]
    q = round5.x3_question(item(), rows, 1, "Series")
    assert q["self_made"] and q["threshold"] > rows[-1][1] and q["outcome"] is None
    assert round5.x3_resolve(q, [(date(2026, 9, 14), q["threshold"] + 1)]) == 1
    assert round5.x3_resolve(q, [(date(2026, 9, 14), q["threshold"])]) == 0
    assert round5.x3_resolve(q, [(date(2026, 9, 16), 99.0)]) is None
    picks = round5.x3_pick([item(id=str(i)) for i in range(40)], 60)
    assert len(picks) == 60 and {m for _, m in picks} == set(round5.X3_MULTIPLES)


# ---------------------------------------------------------------- score


def test_cluster_t_matches_textbook_and_explicit_clusters():
    assert abs(score.t_quantile(0.975, 21) - 2.0796) < 1e-3
    assert abs(score.t_two_sided_p(2.0796, 21) - 0.05) < 1e-3
    diffs = {"a": 0.1, "b": 0.1, "c": 0.2, "d": 0.3}
    groups = {"a": "x", "b": "x", "c": "y", "d": "z"}
    mean, lo, hi, p = score.cluster_t(diffs, groups)
    assert abs(mean - 0.2) < 1e-12 and lo < mean < hi and 0 < p < 1
    qs = [{"id": "a", "cluster": "k1", "group": "g", "resolution_date": "d"},
          {"id": "b", "cluster": "k2", "group": "g", "resolution_date": "d"}]
    assert score.clusters(qs) == {"a": "c:k1", "b": "c:k2"}  # explicit clusters are not joined by group or date


def test_compare_cluster_t_keeps_question_mean_and_handles_no_crowd():
    qs = [{"id": str(i), "cluster": f"k{i % 4}", "outcome": i % 2, "crowd": None} for i in range(16)]
    base = {q["id"]: [0.5, 0.5, 0.5] for q in qs}
    other = {q["id"]: [0.6 if q["outcome"] else 0.4] * 3 for q in qs}
    r = score.compare(qs, base, other, 3, None, 3, test="cluster-t")
    assert abs(r["gain"]["mean"] - (0.25 - 0.16)) < 1e-9 and r["brier"]["crowd"] is None
    assert "t-test" in r["test"] and r["clusters"] == 4


# ---------------------------------------------------------------- run.py --data-block


def test_scrub_value_and_arms():
    text, n = run.scrub_value("Latest 910776.0, or 910,776; not 1910776 or 910776.05.", 910776.0)
    assert n == 2 and "1910776" in text and "910776.05" in text
    q = item(freeze_value_explanation="The latest value.", background="was 4.2 then")
    assert run.round5_item(q)["background"] == "was [value removed] then"
    assert run.data_block_text(q, "off") == {"text": "", "arm": "A0"}
    assert "4.2" in run.data_block_text(q, "fb")["text"]


def test_data_block_ours_formats_rows_before_t(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", FAKE_KEY)
    out = run.data_block_text(item(), "ours", FakeHttp(fred_handler([("2026-08-04", "4.1"), ("2026-08-05", "4.2")])))
    assert out["arm"] == "A2" and out["last"] == "2026-08-05" and "Latest values before 2026-08-06" in out["text"]
    missing = run.data_block_text(item(), "ours", FakeHttp(fred_handler([])))
    assert "No data series was found" in missing["text"] and "error" in missing


def test_resume_refuses_changed_data_block():
    ns = argparse.Namespace(forecaster="claude-cli/claude-opus-5-5", effort="high", prompt="template",
                            evidence=False, reverse=False, data_block="ours")
    sha = run._sha256(run.ROOT / "data_research.py")
    first = f"data_block: ours\ndata_research_sha256: {sha}\n"
    assert run.check_resume_settings(first, ns) is None
    assert "data-block" in run.check_resume_settings(first.replace("ours", "off", 1), ns)
    assert "changed" in run.check_resume_settings(first.replace(sha, "0" * 64), ns)
