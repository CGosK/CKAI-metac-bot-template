"""Offline tests of the dated current-value prototype (fake HTTP, no network, no model calls).
Payload shapes copied from real responses recorded on 2026-11-25."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_research as dr  # noqa: E402

T = datetime(2026, 8, 6, tzinfo=timezone.utc)


class FakeHttp:
    def __init__(self, by_prefix: dict[str, bytes]):
        self.by_prefix, self.calls = by_prefix, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        for prefix, body in self.by_prefix.items():
            if url.startswith(prefix):
                return body
        raise dr.DataUnavailable(f"no fake for {url}")


def wb_payload(values: dict[str, float | None]) -> bytes:
    records = [{"indicator": {"id": "FP.CPI.TOTL.ZG", "value": "Inflation, consumer prices (annual %)"},
                "country": {"id": "US", "value": "United States"}, "countryiso3code": "USA",
                "date": year, "value": v, "unit": "", "obs_status": "", "decimal": 1}
               for year, v in sorted(values.items(), reverse=True)]
    return json.dumps([{"page": 1, "pages": 1, "per_page": 100, "total": len(records),
                        "lastupdated": "2026-11-11"}, records]).encode()


def test_world_bank_drops_years_too_recent_to_be_published():
    http = FakeHttp({"https://api.worldbank.org/": wb_payload(
        {"2026": 9.9, "2025": 2.7, "2024": 2.9495, "2023": 4.1163, "2022": None})})
    s = dr.world_bank("FP.CPI.TOTL.ZG", "USA", T, http)
    # 2025 ends Dec 31 2025, 218 days before T: kept with the 180-day lag; 2026 is not before T.
    assert [d.year for d, _ in s.rows] == [2023, 2024, 2025]
    assert s.cutoff == date(2026, 2, 7) and "not a vintage" in s.notes[1]
    assert "date=1960%3A2026" in http.calls[0] and "api_key" not in http.calls[0]
    s2 = dr.world_bank("FP.CPI.TOTL.ZG", "USA", T, http, lag_days=240)
    assert [d.year for d, _ in s2.rows] == [2023, 2024]


def test_world_bank_error_message_is_unavailable():
    body = json.dumps([{"message": [{"id": "120", "value": "Invalid value"}]}]).encode()
    with pytest.raises(dr.DataUnavailable):
        dr.world_bank("NOPE", "USA", T, FakeHttp({"https://api.worldbank.org/": body}))


TREASURY_2026 = ('Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr",'
                 '"10 Yr","20 Yr","30 Yr"\n'
                 "08/07/2026,4.20,4.27,4.28,4.36,4.36,4.46,4.59,4.63,4.67,4.69,4.79,4.80,5.33,5.30\n"
                 "08/06/2026,4.20,4.27,4.28,4.36,4.36,4.46,4.59,4.63,4.67,4.69,4.79,4.77,5.33,5.30\n"
                 "08/05/2026,4.20,4.27,4.28,4.36,4.36,4.46,4.59,4.63,4.67,4.69,4.79,4.74,5.33,5.30\n"
                 "08/04/2026,4.20,4.27,4.28,4.36,4.36,4.46,4.59,4.63,4.67,4.69,4.79,,5.33,5.30\n"
                 "01/02/2026,4.20,4.27,4.28,4.36,4.36,4.46,4.59,4.63,4.67,4.69,4.79,4.10,5.33,5.30\n")
TREASURY_2025 = ('Date,"1 Mo","2 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"\n'
                 "08/05/2025,4.3,4.3,4.3,4.3,4.0,3.7,3.7,3.8,4.0,4.20,4.8,4.8\n")


def test_treasury_cuts_at_t_skips_blanks_and_reads_last_year():
    http = FakeHttp({dr.TREASURY.format(year=2026): TREASURY_2026.encode("utf-8-sig"),
                     dr.TREASURY.format(year=2025): TREASURY_2025.encode()})
    s = dr.treasury_yield("10 Yr", T, http)
    assert s.rows == [(date(2025, 8, 5), 4.2), (date(2026, 1, 2), 4.1), (date(2026, 8, 5), 4.74)]
    assert len(http.calls) == 2
    with pytest.raises(ValueError):
        dr.treasury_yield("11 Yr", T, http)


def db_payload(days, values) -> bytes:
    return json.dumps({"series": {"docs": [{
        "@frequency": "daily", "dataset_code": "CISS", "indexed_at": "2026-11-11T07:16:06.324Z",
        "period": days, "period_start_day": days, "series_code": "D.IT.Z0Z.4F.EC.SS_CIN.IDX",
        "series_name": "Daily - Italy - New Composite Indicator of Systemic Stress (CISS) - Index",
        "value": values}], "num_found": 1}, "errors": None}).encode()


def test_dbnomics_skips_na_and_cuts_with_lag():
    days = ["2026-07-30", "2026-07-31", "2026-08-03", "2026-08-05", "2026-08-06", "2026-08-07"]
    http = FakeHttp({"https://api.db.nomics.world/": db_payload(days, [0.05, "NA", 0.06, 0.07, 0.5, 0.6])})
    s = dr.dbnomics("ECB", "CISS", "D.IT.Z0Z.4F.EC.SS_CIN.IDX", T, http, lag_days=0)
    assert [d.isoformat() for d, _ in s.rows] == ["2026-07-30", "2026-08-03", "2026-08-05"]
    daily = dr.dbnomics("ECB", "CISS", "D.IT.Z0Z.4F.EC.SS_CIN.IDX", T, http)  # the plan's daily lag: 2 days
    assert [d.isoformat() for d, _ in daily.rows] == ["2026-07-30", "2026-08-03"]
    s2 = dr.dbnomics("ECB", "CISS", "D.IT.Z0Z.4F.EC.SS_CIN.IDX", T, http, lag_days=3)
    assert [d.isoformat() for d, _ in s2.rows] == ["2026-07-30"]
    assert "metadata=false" in http.calls[0]


def test_dbnomics_not_found():
    body = json.dumps({"series": {"docs": []}, "errors": [{"message": "Series not found"}]}).encode()
    with pytest.raises(dr.DataUnavailable):
        dr.dbnomics("X", "Y", "Z", T, FakeHttp({"https://api.db.nomics.world/": body}))


def test_format_block_has_latest_range_and_no_rows_at_or_after_t():
    http = FakeHttp({dr.TREASURY.format(year=2026): TREASURY_2026.encode(),
                     dr.TREASURY.format(year=2025): TREASURY_2025.encode()})
    text = dr.format_block([dr.treasury_yield("10 Yr", T, http)], T, recent=2)
    assert text.startswith("Latest values before 2026-08-06 (T).")
    assert "latest 4.74 percent on 2026-08-05 (1 day before T)" in text
    assert "A year earlier: 4.2 on 2025-08-05." in text
    assert "(2026-01-02, 4.1), (2026-08-05, 4.74)" in text
    assert "2026-08-06," not in text and "4.77" not in text and "4.8," not in text


def test_format_block_refuses_a_leaked_row():
    s = dr.Series("Fake", "x", "", [(date(2026, 8, 6), 1.0)], date(2026, 8, 7), "observation-date cut")
    with pytest.raises(dr.DataLeak):
        dr.format_block([s], T)


def test_format_block_with_no_series_says_so():
    assert "Do not assume a current value" in dr.format_block([], T)


def test_polite_http_sends_public_user_agent(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(request, timeout):
        seen["ua"] = request.get_header("User-agent")
        return Resp()

    monkeypatch.setattr(dr.urllib.request, "urlopen", fake_urlopen)
    assert dr.PoliteHttp(gap=0)("https://example.invalid/x") == b"ok"
    assert seen["ua"] == dr.USER_AGENT
