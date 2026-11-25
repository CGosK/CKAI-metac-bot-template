"""
A revised binary forecasting prompt (round-3 plan, E2). Off by default: the bot uses it only when
PROMPT_VERSION=v2.

Changes from the template's binary prompt (main.py), each from the backtest's error analysis:
- a base-rate step (reference class and rate) first, and the rationale starts from it;
- a step that lists plausible readings of the resolution criteria and forecasts across them;
- the status-quo advice applies only when the research covers the last few days, and the status-quo
  step asks for the date of the most recent fact in the research, so the model checks how current it is.
Everything else (inputs, order, text processing, output format) is the template's: the prompt is built
the same way and passed through the same clean_indents.
"""

from __future__ import annotations

from forecasting_tools import clean_indents

PROMPT_VERSIONS = ("template", "v2")


def binary_prompt_v2(question_text: str, background: str, criteria: str, fine_print: str,
                     research: str, today: str, disclaimer: str) -> str:
    return clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question_text}

            Question background:
            {background}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {criteria}

            {fine_print}


            Your research assistant says:
            {research}

            Today is {today}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) A reference class of similar past situations and its base rate.
            (c) The plausible readings of the resolution criteria, if more than one, and how likely each is to be the one applied. Forecast across them rather than picking one.
            (d) The status quo outcome if nothing changed, and the date of the most recent fact in the research.
            (e) A brief description of a scenario that results in a No outcome.
            (f) A brief description of a scenario that results in a Yes outcome.

            You write your rationale starting from the base rate in (b) and adjusting for this question's specifics, remembering that the status quo deserves extra weight only when the research covers the last few days before today. When the research is older, events may have moved since it was written.
            {disclaimer}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
    )


# The lines that differ from the template's prompt, for the test that everything else is identical.
TEMPLATE_ONLY = (
    "(b) The status quo outcome if nothing changed.",
    "(c) A brief description of a scenario that results in a No outcome.",
    "(d) A brief description of a scenario that results in a Yes outcome.",
    "You write your rationale remembering that good forecasters put extra weight on the status quo outcome "
    "since the world changes slowly most of the time.",
)
V2_ONLY_PREFIXES = ("(b) A reference class", "(c) The plausible readings", "(d) The status quo outcome",
                    "(e) A brief", "(f) A brief", "You write your rationale starting from the base rate")
