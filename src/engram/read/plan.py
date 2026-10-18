"""Question-shape routing for recall. Deterministic, no model call. The word patterns were fixed before they were
measured, on development data only."""

from __future__ import annotations

import re

# Counting / listing questions need breadth across whole conversations.
AGGREGATION = re.compile(r"\b(how many|in total|total|all|every|list|count)\b|\bhow much\b.*\btotal\b"
                         r"|\bwhich\b.*\b(have|own|do|did)\b", re.I)
# Time questions get the agent's dated facts first.
TIME_SHAPED = re.compile(r"\bwhen\b|\bwhat (date|day|month|year|time)\b|\bhow long\b"
                         r"|\bhow many (days|weeks|months|years)\b|\bago\b|\bbefore\b|\bafter\b", re.I)


def is_aggregation(question: str) -> bool:
    return bool(AGGREGATION.search(question))


def is_time_shaped(question: str) -> bool:
    return bool(TIME_SHAPED.search(question))
