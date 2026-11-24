"""
The bot workflows pass every setting the bot reads. A missing line here means
a GitHub secret or variable is set but never reaches the bot.

Run with:  uv run --with pytest python -m pytest tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
BOT_WORKFLOWS = ["test_bot.yaml", "run_bot_on_tournament.yaml", "run_bot_on_metaculus_cup.yaml"]

REQUIRED = {
    "METACULUS_TOKEN": "secrets",
    "OPENROUTER_API_KEY": "secrets",
    "ASKNEWS_API_KEY": "secrets",
    "ASKNEWS_CLIENT_ID": "secrets",
    "ASKNEWS_SECRET": "secrets",
    "FORECASTER_MODELS": "vars",
    "RESEARCHER_MODEL": "vars",
    "SUMMARIZER_MODEL": "vars",
    "PARSER_MODEL": "vars",
    "PREDICTIONS_PER_QUESTION": "vars",
    "REASONING_EFFORT": "vars",
}


@pytest.mark.parametrize("workflow", BOT_WORKFLOWS)
def test_workflow_passes_every_setting(workflow):
    text = (WORKFLOWS / workflow).read_text()
    for name, source in REQUIRED.items():
        line = f"{name}: ${{{{ {source}.{name} }}}}"
        assert line in text, f"{workflow} doesn't pass {name}"
