"""Exploratory splits for the Wikipedia research run (not pre-registered).

Reads saved runs only, no model calls; writes backtest/out/explore-wiki.json.
Run: uv run python -m backtest.explore_wiki
"""
import json, statistics, sys
from backtest import score as S
meta, base_qs, base = S.load(["pilot", "main"])
_, _, wiki = S.load(["wiki"])
_, _, ask = S.load(["research2"])
qs = {q["id"]: q for q in meta["questions"]}
groups_all = S.clusters(list(qs.values()))
y = {i: q["outcome"] for i, q in qs.items()}
crowd = {i: q["crowd"] for i, q in qs.items()}
med = lambda v: statistics.median(v[:3])
out = {}
# wiki vs asknews on shared questions (both median of 3)
shared = sorted(i for i in ask if len(ask[i]) >= 3 and i in wiki and len(wiki[i]) >= 3 and len(base.get(i, [])) >= 5)
groups = S.clusters([qs[i] for i in shared])  # built on the compared set, as in score.compare
d = {i: S.brier(med(wiki[i]), y[i]) - S.brier(med(ask[i]), y[i]) for i in shared}
m, lo, hi, p = S.group_bootstrap(d, groups)
out["wiki_vs_asknews_shared"] = {
    "n": len(shared), "clusters": len({groups[i] for i in shared}),
    "brier_baseline_sub3": statistics.fmean(S.subsample_brier(base[i], 3, y[i]) for i in shared),
    "brier_asknews": statistics.fmean(S.brier(med(ask[i]), y[i]) for i in shared),
    "brier_wiki": statistics.fmean(S.brier(med(wiki[i]), y[i]) for i in shared),
    "brier_crowd": statistics.fmean(S.brier(crowd[i], y[i]) for i in shared),
    "wiki_minus_asknews_brier": {"mean": m, "ci95": [lo, hi], "p": p},
}
# gain split by wiki leak flags
rs = {json.loads(l)["id"]: json.loads(l) for l in open("backtest/runs/wiki/research.jsonl")}
gain = {i: S.subsample_brier(base[i], 3, y[i]) - S.brier(med(wiki[i]), y[i]) for i in wiki}
def split(name, f):
    a = [gain[i] for i in gain if f(rs[i])]; b = [gain[i] for i in gain if not f(rs[i])]
    out[name] = {"flagged": {"n": len(a), "mean_gain": statistics.fmean(a) if a else None},
                 "not_flagged": {"n": len(b), "mean_gain": statistics.fmean(b) if b else None}}
split("by_later_date_flag", lambda r: bool(r["leak"]["later_dates"]))
split("by_market_flag", lambda r: bool(r["leak"]["markets"]))
split("by_no_articles", lambda r: "No articles were found" in r["text"])
# by source
src = {}
for i in gain: src.setdefault(qs[i]["source"], []).append(gain[i])
out["by_source"] = {k: {"n": len(v), "mean_gain": statistics.fmean(v)} for k, v in src.items()}
json.dump(out, open("backtest/out/explore-wiki.json", "w"), indent=1)
print(json.dumps(out, indent=1))
