"""The backtest's Wikipedia research now lives in wiki_research.py (shared with
the live bot). This name stays so earlier launch records and imports work:
`from backtest import wiki` gives the same module object."""

import sys

import wiki_research

sys.modules[__name__] = wiki_research
