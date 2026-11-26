"""Round 6 (benchmarks/PLAN-refine-round6-levels.md): offline checks of the build, exclusion and scoring rules."""

from datetime import date, timedelta

from backtest import round6 as r6


def daily(n=900, start=date(2024, 1, 1), f=lambda i: 100 + (i % 9) - 4):
    return [(start + timedelta(days=i), float(f(i))) for i in range(n)]


def test_family_frequency_and_policy_rule():
    assert r6.family("dbnomics", "ECB_EXR_D.USD.EUR.SP00.A", "x") == "ECB/EXR"
    assert r6.family("fred", "DGS10", "Market Yield on U.S. Treasury Securities at 10-Year") == "us-rates"
    assert r6.family("fred", "WM2NS", "M2") == "money"
    assert r6.family("fred", "X", "Something else") == "fred-other"
    for title in ("Liabilities and Capital: Liabilities: Earnings Remittances Due to the U.S. Treasury: Wednesday Level",
                  "Assets: Securities Held Outright: U.S. Treasury Securities: All: Wednesday Level",
                  "Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve",
                  "Liabilities and Capital: Liabilities: Deposits with F.R. Banks, Other Than Reserve Balances"):
        assert r6.family("fred", "X", title) == "h41-fed", title
    assert r6.frequency(daily()) == "daily"
    weekly = [(date(2024, 1, 3) + timedelta(weeks=i), 1.0 + i) for i in range(120)]
    assert r6.frequency(weekly) == "weekly"
    monthly = [(date(2020, 1, 1) + timedelta(days=30 * i), 1.0) for i in range(40)]
    assert r6.frequency(monthly) is None
    for name in ("30-Day Average SOFR", "Federal Funds Effective Rate", "4-Week Treasury Bill Secondary Market Rate",
                 "Euro Short-Term Rate: Volume-Weighted Trimmed Mean Rate", "Bank Prime Loan Rate",
                 "Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity"):
        assert r6.POLICY_RATE.search(name), name
    for name in ("Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity", "1-Year Treasury Bill",
                 "10-Year Treasury Constant Maturity Minus Federal Funds Rate", "M2"):
        assert not r6.POLICY_RATE.search(name), name


def test_threshold_is_drift_centred_and_shown_fixed():
    rows = daily(f=lambda i: 1000 + 2 * i + (i % 5))  # steady drift
    t0 = r6.threshold(rows, 30, 0.5)
    assert abs(t0["mean_change"] - 60) < 1  # 30 days of +2
    assert t0["threshold"] > t0["anchor"] + 50  # centred on the drift, not the anchor
    assert 0.0 <= t0["yes_rate"] <= 1.0 and t0["places"] == 1
    assert r6.show(6776089.333, 1) == "6776089.3"  # never scientific
    assert r6.threshold(rows[:15], 7, 1.0) is None


def test_exclusion_reasons():
    rows = daily()
    t = r6.threshold(rows, 7, 0.5)
    assert r6.exclusion("fred", "X", "Some index", "daily", t) is None
    assert r6.exclusion("fred", "X", "30-Day Average SOFR", "daily", t) == "policy-administered rate"
    assert r6.exclusion("fred", "X", "Some index", None, t) == "not daily or weekly"
    assert r6.exclusion("dbnomics", "ECB_CISS_D.BE.X", "x", "daily", t).startswith("dbnomics dataset")
    sticky = r6.threshold(daily(f=lambda i: 5.0 if i % 40 else 5.25), 7, 0.5)
    assert r6.exclusion("fred", "X", "Rate", "daily", sticky) == "sticky"
    sure = dict(t, yes_rate=0.95)
    assert r6.exclusion("fred", "X", "Some index", "daily", sure) == "historical Yes rate outside 10% to 90%"
    assert r6.exclusion("fred", "X", "Some index", "daily", t, date(2026, 6, 1)) is None
    old = dict(t, anchor_date="2026-07-06")
    assert r6.exclusion("fred", "X", "M2", "weekly", old, date(2026, 8, 5)) == "published with a lag over 21 days"
    neg = dict(t, threshold=-1.0, never_negative=True)
    assert r6.exclusion("fred", "X", "Some index", "daily", neg) == "negative threshold on a non-negative series"


def test_draw_balance_caps_and_backups():
    cands = []
    for i in range(200):
        cands.append({"key": f"s{i % 60}", "family": f"f{i % 12}", "multiple": r6.MULTIPLES[i % 4],
                      "ok": i % 7 != 0})
    prim, back = r6.draw(cands)
    assert len(prim) == 20 and len(back) == 3
    keys = [c["key"] for c in prim + back]
    assert len(keys) == len(set(keys))  # one per series
    fam = {}
    for c in prim + back:
        fam[c["family"]] = fam.get(c["family"], 0) + 1
    assert max(fam.values()) <= 3
    per_m = {}
    for c in prim:
        per_m[c["multiple"]] = per_m.get(c["multiple"], 0) + 1
    assert set(per_m.values()) == {5}
    assert all(c["ok"] for c in prim + back)


def test_scored_replaces_unresolvable_primaries_with_backups():
    def q(i, role, rank, outcome, cl="2026-08-05", h=7):
        return {"id": f"{cl}{h}{i}", "cluster": cl, "horizon_days": h, "role": role, "rank": rank, "outcome": outcome}
    qs = [q(i, "primary", i, None if i in (3, 7) else 1) for i in range(20)]
    qs += [q(100 + i, "backup", i, None if i == 0 else 0) for i in range(3)]
    keep, notes = r6.scored(qs)
    assert len(keep) == 20 and {x["id"] for x in keep} >= {"2026-08-057101", "2026-08-057102"}
    assert notes["2026-08-05 h7"]["unresolvable_primaries"] == 2 and notes["2026-08-05 h7"]["scored"]
    short = [q(i, "primary", i, None if i < 5 else 1, h=30) for i in range(20)] + [
        q(200 + i, "backup", i, 1, h=30) for i in range(3)]
    keep2, notes2 = r6.scored(short)
    assert keep2 == [] and not notes2["2026-08-05 h30"]["scored"]


def test_resolve_fails_closed_on_stale(monkeypatch):
    q = {"source": "fred", "series_id": "X", "frequency": "daily", "resolution_date": "2026-09-30", "threshold": 1.0}
    monkeypatch.setattr(r6.dr, "fred_vintage_rows", lambda *a, **k: [(date(2026, 9, 20), 5.0)])
    assert r6.resolve_one(q, None)["outcome"] is None
    monkeypatch.setattr(r6.dr, "fred_vintage_rows",
                        lambda *a, **k: [(date(2026, 9, 29), 0.5), (date(2026, 10, 1), 9.0)])
    r = r6.resolve_one(q, None)
    assert r["outcome"] == 0 and r["obs_date"] == "2026-09-29"  # later rows ignored; strictly above


def test_resolve_fetches_each_dbnomics_series_once(monkeypatch):
    calls = []
    monkeypatch.setattr(r6.dr, "dbnomics_doc", lambda *a: calls.append(a) or {})
    monkeypatch.setattr(r6.dr, "dbnomics_rows", lambda doc: [(date(2026, 9, 29), 2.0)])
    cache = {}
    for _ in range(3):
        q = {"source": "dbnomics", "series_id": "ECB_EXR_D.USD.EUR.SP00.A", "frequency": "daily",
             "resolution_date": "2026-09-30", "threshold": 1.0}
        assert r6.resolve_one(q, None, fresh=object(), dbn_cache=cache)["outcome"] == 1
    assert len(calls) == 1
