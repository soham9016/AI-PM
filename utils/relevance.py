"""Page-relevance matching, shared across any agent that filters fetched
pages against anchor terms (originally lived in agents/researcher.py;
moved here once agents/primary_research.py needed the identical logic).

A page is relevant if it matches EITHER the company anchor OR the sector
anchor — not sector alone (an earlier sector-only version rejected a
Swiggy earnings article because its text never spelled out "Quick
commerce" verbatim, while a company match ("Swiggy") would have caught
it immediately). Sector matching is fuzzy: case-insensitive, tolerant of
hyphen/space variants and the standard first-letter abbreviation style
("quick commerce" also matches "quick-commerce" and "q-commerce").
"""

import re


def sector_pattern(term: str) -> re.Pattern:
    """Case-insensitive pattern for a sector term that tolerates
    hyphen/space variation between words ("quick commerce" also matches
    "quick-commerce") and, for a multi-word term, the standard
    first-letter abbreviation style ("quick commerce" also matches
    "q-commerce" / "q commerce") — that's a real industry naming
    convention (q-commerce, e-commerce, m-commerce), not just a
    formatting quirk, and a plain substring match on the exact
    planner-generated phrase was too brittle to rely on."""
    words = term.strip().split()
    if not words:
        return re.compile(r"(?!)")  # matches nothing
    sep = r"[\s-]?"
    alternatives = [sep.join(re.escape(w) for w in words)]
    if len(words) > 1:
        alternatives.append(re.escape(words[0][0]) + sep + sep.join(re.escape(w) for w in words[1:]))
    return re.compile("|".join(alternatives), re.IGNORECASE)


def is_relevant(text: str, anchor_terms: dict) -> bool:
    """A page is relevant if it matches EITHER the company anchor OR the
    sector anchor. Fails OPEN only when neither term is present (e.g. the
    planner's own JSON parse failed) — this is a quality heuristic, not a
    safety gate, so a planner hiccup shouldn't turn every run into
    "insufficient data" by rejecting everything.
    """
    sector = (anchor_terms or {}).get("sector") or ""
    company = (anchor_terms or {}).get("company") or ""

    if not sector and not company:
        return True
    if company and company.lower() in text.lower():
        return True
    if sector and sector_pattern(sector).search(text):
        return True
    return False
