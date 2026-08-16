"""Shared token-overlap primitives, extracted from tools/evidence_extractor.py
so the same "is A actually locatable in B" logic isn't reimplemented
wherever it's needed — currently: grounding claims in page text
(evidence_extractor), branch-relevance for competitive-audit fan-out
(competitive_audit), and solution deduplication (solution_framing).

Longest-CONTIGUOUS-run overlap, not a bag-of-words Jaccard, on purpose:
the failure modes this catches (near-verbatim copying, restating the
same idea) show up as a long shared run of tokens in order, not just
shared vocabulary — two claims about the same company will naturally
share individual words without being duplicates of each other.
"""

import difflib
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def longest_run_ratio(a_tokens: list[str], b_tokens: list[str]) -> tuple[float, int]:
    """Longest contiguous token run shared between a and b, as a fraction
    of len(a_tokens), plus the raw run length."""
    if not a_tokens:
        return 0.0, 0
    matcher = difflib.SequenceMatcher(None, a_tokens, b_tokens, autojunk=False)
    match = matcher.find_longest_match(0, len(a_tokens), 0, len(b_tokens))
    return match.size / len(a_tokens), match.size
