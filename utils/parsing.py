"""Shared helper for extracting JSON from an LLM's text response.

Agents are prompted to respond with a single JSON object, but models
sometimes wrap it in prose or a markdown code fence — this strips that
off before parsing.
"""

import json
import logging
import re

logger = logging.getLogger("business_copilot.parsing")

_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from raw LLM output.

    Raises json.JSONDecodeError if no valid JSON object can be found.
    """
    fenced = _CODE_FENCE.search(text)
    candidate = fenced.group(1) if fenced else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT.search(candidate)
    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError("No JSON object found in LLM output", text, 0)


def safe_extract_json(text: str, default: dict | None = None) -> dict:
    """Like extract_json, but never raises.

    A malformed/truncated LLM response is a normal "ran, produced nothing
    usable" outcome, not a reason to crash the whole graph run — callers
    already handle a missing key via `.get(key, default)`, so returning
    `{}` here degrades the same way an empty-but-valid response would.
    """
    try:
        return extract_json(text)
    except json.JSONDecodeError:
        logger.warning("Failed to extract JSON from LLM output: %r", text[:500])
        return {} if default is None else default
