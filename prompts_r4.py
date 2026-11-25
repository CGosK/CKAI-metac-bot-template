"""
Round-4 prompt changes (benchmarks/PLAN-refine-round4.md). Kept apart from prompts_v2.py so the round-3 batch
that hashes that file can resume unchanged.

- add_evidence_step (G1, live setting PROMPT_EVIDENCE=on, off by default): two instructions added to whichever
  binary prompt is in use, just before its rationale line.
- reverse_wrap (X1, backtest only): the same prompt, asking for the probability that the question resolves NO.
- reconcile_prompt (X2, backtest only): related questions with their forecasts and reasoning, asking for
  mutually consistent probabilities.
"""

from __future__ import annotations

import json
import re

RATIONALE_PREFIX = "You write your rationale"

EVIDENCE_STEP = (
    "Then also write:\n"
    "(i) The single fact that would most decide this question, and whether the research contains it, with its "
    "date. If the research does not contain it, do not treat its absence from the research as evidence for No.\n"
    "(ii) If a reading of the resolution criteria implies the question has already resolved, that reading is "
    "wrong; say which reading you use.\n"
)

NOT_YET_SATISFIED = "These criteria have not yet been satisfied:"
REVERSE_HEADER = (
    "IMPORTANT: This task asks for the probability that the question below resolves NO, meaning the named "
    "event, candidate, party or outcome is not the result (including 'Other', or no result in time). Read the "
    "question and criteria as written. Assume the question does not resolve N/A or get annulled.\n\n"
)
REVERSE_FOOTER = (
    "\n\nRemember: your final answer is the probability that the question above resolves NO, "
    'written as "Probability: ZZ%", 0-100.'
)


def _rationale_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.startswith(RATIONALE_PREFIX):
            return i
    raise ValueError("binary prompt has no rationale line to insert the evidence step before")


def add_evidence_step(prompt: str) -> str:
    lines = prompt.split("\n")
    i = _rationale_index(lines)
    return "\n".join(lines[:i] + EVIDENCE_STEP.rstrip("\n").split("\n") + [""] + lines[i:])


def reverse_wrap(prompt: str) -> str:
    if NOT_YET_SATISFIED not in prompt:
        raise ValueError("binary prompt lacks the criteria line the reversal rewrites")
    body = prompt.replace(NOT_YET_SATISFIED, "This question has not yet resolved. Its criteria:")
    return REVERSE_HEADER + body + REVERSE_FOOTER


def combine_both_directions(p_yes: float, p_no: float) -> float:
    """Prophet Arena's combination (Appendix C.3): the mean of P(Yes) and 1 - P(No)."""
    return 0.5 * (p_yes + 1 - p_no)


def reconcile_prompt(questions: list[dict], today: str) -> str:
    """questions: id, question, criteria, forecast, reasoning."""
    parts = [
        "Below are related forecasting questions from the same market group. A forecaster has given each a "
        "probability with reasoning. Some may be mutually exclusive (at most one can resolve Yes), nested, or "
        "otherwise linked.\n",
        f"Today is {today}.\n",
        "Revise the probabilities so that they are consistent with each other and with the evidence in the "
        "reasoning. Change a probability only where consistency or the reasoning calls for it.\n",
    ]
    for q in questions:
        parts.append(
            f"--- Question {q['id']}\n{q['question']}\nCriteria: {q['criteria'][:1500]}\n"
            f"Forecast: {q['forecast']:.3f}\nReasoning:\n{q['reasoning']}\n"
        )
    ids = ", ".join(f'"{q["id"]}": <probability 0-1>' for q in questions)
    parts.append(f"Reply with JSON only, one entry per question: {{{ids}}}")
    return "\n".join(parts)


def parse_reconciled(reply: str, ids: list[str]) -> dict[str, float]:
    data = None
    for start in [m.start() for m in re.finditer(r"\{", reply)][::-1]:  # the last parseable JSON object
        try:
            data, _ = json.JSONDecoder().raw_decode(reply[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and all(i in data for i in ids):
            break
        data = None
    if data is None:
        raise ValueError(f"no JSON with every id in reconcile reply: {reply[:200]!r}")
    out = {}
    for i in ids:
        if i not in data:
            raise ValueError(f"reconcile reply lacks {i}")
        p = float(data[i])
        if p > 1:  # a percentage
            p /= 100
        if not 0 <= p <= 1:
            raise ValueError(f"reconcile reply gives {i} = {data[i]!r}")
        out[i] = min(0.99, max(0.01, p))  # the template's clamp
    return out
