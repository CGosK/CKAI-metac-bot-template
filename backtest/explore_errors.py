"""Error analysis for the Wikipedia research run (exploratory, not pre-registered).

Reads saved runs only, stdlib only, no model calls. Bot forecast per question = median of the
wiki arm's forecasts (k = 0..2). Writes backtest/out/errors-wiki.json.
Run from the repo root: python3 backtest/explore_errors.py
"""
import json, re, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
qs = {q["id"]: q for q in json.load(open(HERE / "data/questions.json"))["questions"]}
preds = {}
for line in open(HERE / "runs/wiki/predictions.jsonl"):
    r = json.loads(line)
    if r.get("prediction") is not None:
        preds.setdefault(r["id"], []).append((r["k"], float(r["prediction"])))

# Topic rule: first matching keyword list in this order, on the question title (case-insensitive).
TOPICS = [
    ("sports", r"us open|world series|championship series|f1 |wnba|nfl|worlds 2026|lol worlds|ballon|golden boot"
               r"|mlb|world cup|melbourne cup|season series"),
    ("elections", r"elect|nominee|nomination|house seat|senate|governor|mayor|primary|after (the )?(2026 )?midterms|majority|parliament|runoff"
                  r"|turnout|knesset|seats|first round|blue wave|blue tsunami|referendum|amendment|win .* race"
                  r"|county|premier"),
    ("geopolitics", r"iran|israel|russia|ukrain|china|taiwan|ceasefire|hormuz|nuclear|gaza|lebanon|military|putin"
                    r"|\bxi\b|coup|blockade|opec|cuba|mali|netanyahu|cyprus|haiti|albania|duterte|schengen"),
    ("prices_econ", r"bitcoin|crude|oil|\bfed\b|interest rate|inflation|xrp|solana|strc|gross|gas prices|treasury"
                    r"|fdv|dow jones|trillionaire|volume|tariff|attendance|ipo|s-1"),
    ("ai_tech", r"\bai\b|anthropic|openai|gemini|model|starship|starlink|metr|imo|linux|self improving"),
]
def topic(q):
    t = q["question"].lower()
    for name, pat in TOPICS:
        if re.search(pat, t):
            return name
    return "other"

def brier(p, y):
    return (p - y) ** 2

rows = []
for i, q in qs.items():
    if i not in preds:
        continue
    fs = [p for _, p in sorted(preds[i])][:3]
    p = statistics.median(fs)
    y = int(q["outcome"]); c = float(q["crowd"])
    rows.append({
        "id": i, "source": q["source"], "topic": topic(q), "question": q["question"],
        "freeze": q["freeze_datetime"], "resolution_date": q["resolution_date"], "outcome": y,
        "bot_median": p, "forecasts": fs, "spread": max(fs) - min(fs), "crowd": c,
        "bot_brier": brier(p, y), "crowd_brier": brier(c, y), "gap": brier(p, y) - brier(c, y),
    })

def summ(rs):
    n = len(rs)
    return {
        "n": n,
        "bot_brier": round(statistics.fmean(r["bot_brier"] for r in rs), 4),
        "crowd_brier": round(statistics.fmean(r["crowd_brier"] for r in rs), 4),
        "gap": round(statistics.fmean(r["gap"] for r in rs), 4),
        "yes_rate": round(statistics.fmean(r["outcome"] for r in rs), 3),
        "mean_bot": round(statistics.fmean(r["bot_median"] for r in rs), 3),
        "mean_crowd": round(statistics.fmean(r["crowd"] for r in rs), 3),
        # wrong side: strictly on the wrong side of 0.5 (0.5 itself counts as neither)
        "bot_wrong_crowd_right": sum((r["bot_median"] - .5) * (r["outcome"] - .5) < 0
                                     and (r["crowd"] - .5) * (r["outcome"] - .5) > 0 for r in rs),
        "crowd_wrong_bot_right": sum((r["crowd"] - .5) * (r["outcome"] - .5) < 0
                                     and (r["bot_median"] - .5) * (r["outcome"] - .5) > 0 for r in rs),
        "both_wrong": sum((r["crowd"] - .5) * (r["outcome"] - .5) < 0
                          and (r["bot_median"] - .5) * (r["outcome"] - .5) < 0 for r in rs),
    }

def by(key):
    g = {}
    for r in rows:
        g.setdefault(r[key], []).append(r)
    return {k: summ(v) for k, v in sorted(g.items())}

keep = ["id", "source", "topic", "question", "freeze", "resolution_date", "outcome", "bot_median",
        "forecasts", "crowd", "bot_brier", "crowd_brier", "gap"]
slim = lambda r: {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keep}
worst = sorted(rows, key=lambda r: -r["bot_brier"])[:25]
lost = sorted(rows, key=lambda r: -r["gap"])[:25]
union = {r["id"] for r in worst} | {r["id"] for r in lost}

# Brier share: how much of the bot's total Brier sits in the union; agreement among the 3 forecasts
total = sum(r["bot_brier"] for r in rows)
out = {
    "note": "Exploratory. Bot = median of wiki-arm forecasts (k=0..2). Topic = first keyword match, see TOPICS.",
    "overall": summ(rows),
    "by_source": by("source"),
    "by_topic": by("topic"),
    "forecast_spread": {
        "all_mean_max_minus_min": round(statistics.fmean(r["spread"] for r in rows), 4),
        "union_mean_max_minus_min": round(statistics.fmean(r["spread"] for r in rows if r["id"] in union), 4),
        "union_spread_le_0.10": sum(r["spread"] <= .10 for r in rows if r["id"] in union),
        "union_n": len(union),
    },
    "union_share_of_total_bot_brier": round(sum(r["bot_brier"] for r in rows if r["id"] in union) / total, 3),
    "worst_bot_brier": [slim(r) for r in worst],
    "largest_loss_to_crowd": [slim(r) for r in lost],
}
# Counted: when bot and crowd disagree by more than 0.2, who was right?
def diverge(f):
    rs = [r for r in rows if f(r)]
    return {"n": len(rs), "yes_rate": round(statistics.fmean(r["outcome"] for r in rs), 3) if rs else None,
            "bot_brier": round(statistics.fmean(r["bot_brier"] for r in rs), 4) if rs else None,
            "crowd_brier": round(statistics.fmean(r["crowd_brier"] for r in rs), 4) if rs else None}
out["divergence"] = {
    "bot_below_crowd_by_0.2": diverge(lambda r: r["bot_median"] < r["crowd"] - .2),
    "bot_above_crowd_by_0.2": diverge(lambda r: r["bot_median"] > r["crowd"] + .2),
    "within_0.2": diverge(lambda r: abs(r["bot_median"] - r["crowd"]) <= .2),
}
out["extremes"] = {
    "bot_below_0.1": diverge(lambda r: r["bot_median"] < .1), "crowd_below_0.1": diverge(lambda r: r["crowd"] < .1),
    "bot_above_0.9": diverge(lambda r: r["bot_median"] > .9), "crowd_above_0.9": diverge(lambda r: r["crowd"] > .9),
}
# Counted: US party questions (title names Democrats/Dems/Blue vs Republican and a win/seat/control word)
party = lambda pat: (lambda r: re.search(pat, r["question"]) and
                     re.search(r"win|seat|control|majority|wave|tsunami|hold", r["question"]))
out["us_party_questions"] = {
    "democrat_yes": {**diverge(party(r"Democrat|Dems|Blue")),
                     "mean_bot": round(statistics.fmean(r["bot_median"] for r in rows if party(r"Democrat|Dems|Blue")(r)), 3),
                     "mean_crowd": round(statistics.fmean(r["crowd"] for r in rows if party(r"Democrat|Dems|Blue")(r)), 3)},
    "republican_yes": {**diverge(party(r"Republican")),
                       "mean_bot": round(statistics.fmean(r["bot_median"] for r in rows if party(r"Republican")(r)), 3),
                       "mean_crowd": round(statistics.fmean(r["crowd"] for r in rows if party(r"Republican")(r)), 3)},
}

# JUDGED (not counted): one failure kind per case in the worst/lost union, assigned by reading the
# question, criteria, research text and reasoning. See out/errors-wiki.md for the per-case lines.
JUDGED = {
    "news_gap": ["cSAh00sC2d", "0x9c7b2500bf", "0x183c66ad40", "ppPhqSSddu", "OuQIQtgL0g", "PO5OBxubHj2G",
                 "6sR508pyy0"],
    "state_gap": ["0xd15f77a921", "44986", "0x1239389645", "0xba2998f828", "44799", "0x23969ec065",
                  "0xfb13ea7fb9"],
    "fed_status_quo": ["0x876506d8b2", "0x73d8a8208d", "0xa3b36b2d61", "0x1b56e46dba"],
    "midterm_lean_anchor": ["0x7d8ec7081e", "SyhgEZC68s", "0x096f26b2ab"],
    "judgment_vs_crowd_same_info": ["0x9e69d69cdb", "cd5hUN9EZO", "0x12dc2b6172", "0x294cb485e8"],
    "criteria_interpretation": ["Bn76fB9rYXMk", "38209"],
    "luck_upset_or_longshot": ["0x190ac165e7", "0x06a3407e26", "0x39b718f6e0", "0x6114306f72", "0x8ee27c276a",
                               "0x826627a999", "0x4ba91369e8", "0xa74ddc3ce6", "0xc3a509ea69", "spuyq56Py5"],
}
byid = {r["id"]: r for r in rows}
def find(prefix):
    m = [i for i in byid if i.startswith(prefix)]
    assert len(m) == 1, prefix
    return byid[m[0]]
kinds = {}
for k, ids in JUDGED.items():
    rs = [find(p) for p in ids]
    kinds[k] = {"n": len(rs), "bot_brier_sum": round(sum(r["bot_brier"] for r in rs), 3),
                "loss_to_crowd_sum": round(sum(r["gap"] for r in rs), 3), "ids": [r["id"] for r in rs]}
labelled = {r["id"] for k in JUDGED for r in map(find, JUDGED[k])}
assert labelled == union, (labelled ^ union)
out["judged_failure_kinds"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1]["loss_to_crowd_sum"]))

(HERE / "out").mkdir(exist_ok=True)
json.dump(out, open(HERE / "out/errors-wiki.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["overall", "by_source", "by_topic", "forecast_spread", "divergence", "extremes", "us_party_questions",
                                      "union_share_of_total_bot_brier"]}, indent=1))
for name in ["worst_bot_brier", "largest_loss_to_crowd"]:
    print("\n" + name)
    for r in out[name]:
        print(f"{r['id'][:12]:12} {r['source'][:4]} {r['topic'][:6]:6} y={r['outcome']} bot={r['bot_median']:.2f} "
              f"{r['forecasts']} crowd={r['crowd']:.2f} gap={r['gap']:+.3f} | {r['question'][:70]}")
