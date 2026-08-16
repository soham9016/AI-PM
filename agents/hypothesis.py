"""Hypothesis agent — writes falsifiable hypotheses with kill conditions.

Four rules, enforced via prompt (see SYSTEM_PROMPT for the full text):
  RULE 1 — one claim, one direction. A hypothesis with two subjects or a
    vague verb ("affects"/"impacts"/"influences") can't be ruled either
    way — it's rejected, not written.
  RULE 2 — every hypothesis states `evidence_needed`: the specific
    document/figure/statement that would settle it. If that's
    unpublishable (internal-only data), the hypothesis must be reframed
    to a level that IS checkable, not generated as-is and left to rot as
    "inconclusive" forever.
  RULE 3 — every hypothesis states `evidence_type` (REGULATORY /
    FINANCIAL / PRODUCT / USER / MARKET / COMPETITIVE), since different
    types are settled by different kinds of source and get weighed
    differently downstream — see agents/researcher.py (query targeting)
    and agents/synthesizer.py (verdict weighting), both of which read
    this field directly off the hypothesis dict already passed to them.
  RULE 4 — scope to the named entity where the problem statement names
    one, industry-level otherwise (pre-existing rule, kept, now made to
    interact with RULE 2: a company-specific hypothesis whose evidence
    would have to be unpublishable gets reframed to sector level, not
    generated and abandoned).

Hypotheses must also spread across the issue tree's branches (one per
branch, in priority order, until MAX_HYPOTHESES is reached) rather than
clustering on a single branch — a run that generated four hypotheses,
all variations on one branch, while other branches got none, is a
coverage failure the critic's MECE-gap check would otherwise have to
catch after the fact.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import MAX_HYPOTHESES, get_llm
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry

AGENT_NAME = "hypothesis"

SYSTEM_PROMPT = f"""You are a hypothesis-driven analyst.

Given a business problem statement and a MECE issue tree, write up to
{MAX_HYPOTHESES} falsifiable hypotheses, following these rules in order.

RULE 1 — ONE CLAIM, ONE DIRECTION.
A hypothesis must have exactly one subject and one specific, directional
claim about it. Reject anything with two subjects or a vague verb like
"affects"/"impacts"/"influences" — those can never be ruled true or
false, only gestured at.
  BAD:  "Zomato's data usage affects retention and user experience"
        (two subjects — retention AND user experience — and a vague verb)
  GOOD: "Zomato's consent flow does not separate essential service data
        from optional marketing data" (one subject, one specific,
        checkable claim, a clear direction to confirm or kill)

RULE 2 — NAME THE EVIDENCE THAT WOULD SETTLE IT.
Every hypothesis carries "evidence_needed": the specific document,
figure, or statement that would confirm or kill it. If your honest
answer is "internal company data nobody publishes," the hypothesis as
written is unfalsifiable from public sources — do not write it that way.
Reframe it (see RULE 4) to a level where evidence_needed names something
that actually exists publicly, BEFORE finalizing it, not after it comes
back inconclusive.

RULE 3 — DECLARE THE EVIDENCE TYPE.
Every hypothesis carries "evidence_type", exactly one of: REGULATORY,
FINANCIAL, PRODUCT, USER, MARKET, COMPETITIVE. This matters because each
type is settled differently: a REGULATORY claim ("DPDP requires X") is
settled by the text of the law itself and should not be confirmed from a
LinkedIn post when the actual regulation is retrievable; a USER claim
("users want granular consent controls") needs research or survey data,
not a company's own marketing copy; FINANCIAL needs filings/financial
disclosures; PRODUCT/MARKET/COMPETITIVE need product documentation,
industry reports, or competitor material respectively. Pick the type
that determines what SOURCE would actually settle the claim.

RULE 4 — SCOPE TO WHAT'S CHECKABLE.
Does the problem statement name a SPECIFIC, identifiable company (a
proper name)? If it does NOT — if it describes a generic/anonymized
entity like "a mid-sized manufacturer" — every hypothesis MUST be
industry/sector-level, since there is no public way to check a claim
about a company's private internals:
  Write:  "sector input cost inflation outpaced price pass-through in FY23-25"
  NOT:    "this firm's input costs rose faster than it could pass through"
If the problem statement DOES name a specific company, company-specific
hypotheses are fine, but apply RULE 2 first: if evidence_needed for the
company-specific version would be unpublishable, reframe THAT hypothesis
to the sector level rather than keep the company-specific framing and
let it sit unconfirmed. Don't write a hypothesis you already know can't
be checked.

BRANCH COVERAGE — you will be given the issue tree's branches in
priority order. Write ONE hypothesis per branch, in that order, until
you reach {MAX_HYPOTHESES} hypotheses or run out of branches (skip a
branch only if it's purely descriptive/structural with no causal claim
to test). Do not write multiple hypotheses on one branch while other
branches get none — the point is coverage of the tree, not depth on a
single corner of it.

Respond with ONLY a JSON object of this exact shape:
{{
  "hypotheses": [
    {{
      "id": "H1",
      "branch": "<issue tree branch this maps to>",
      "statement": "<one subject, one specific, directional, checkable claim>",
      "evidence_needed": "<the specific document/figure/statement that would settle this>",
      "evidence_type": "REGULATORY|FINANCIAL|PRODUCT|USER|MARKET|COMPETITIVE",
      "kill_conditions": ["<evidence that would falsify this>", "..."],
      "supporting_conditions": ["<evidence that would confirm this>", "..."]
    }}
  ]
}}
"""


def _priority_branches(issue_tree: dict) -> list[str]:
    return [b.get("name") for b in issue_tree.get("branches", []) if b.get("name")]


@with_retry()
def _call_llm(problem_statement: str, issue_tree: dict, revision_note: str | None) -> str:
    llm = get_llm(AGENT_NAME)
    branches = _priority_branches(issue_tree)
    branch_list = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(branches)) or "(no branches given)"
    content = (
        f"Problem statement:\n{problem_statement}\n\n"
        f"Issue tree:\n{issue_tree}\n\n"
        f"Branches in priority order (write one hypothesis per branch, in this order, "
        f"until you reach {MAX_HYPOTHESES} or run out):\n{branch_list}"
    )
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior hypotheses — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _ensure_hypotheses(hypotheses: list[dict]) -> list[dict]:
    """Guarantee a non-empty list so `not hypotheses` reliably means "hasn't
    run" rather than "ran and found no testable claim" — those must not
    look the same to the ladder."""
    if hypotheses:
        return hypotheses[:MAX_HYPOTHESES]
    return [
        {
            "id": "H0",
            "branch": None,
            "statement": "No falsifiable hypothesis could be derived from the issue tree.",
            "evidence_needed": None,
            "evidence_type": None,
            "kill_conditions": [],
            "supporting_conditions": [],
            "unresolved": True,
        }
    ]


def hypothesis_node(state: dict) -> dict:
    """Write falsifiable hypotheses (with kill conditions) for the issue tree.

    Reads: state['problem_statement'], state['issue_tree'], state['revision_notes']['hypothesis']
    Writes: state['hypotheses'] (each with 'evidence_needed' and 'evidence_type'),
    state['messages'], state['run_path']
    """
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)
    raw = _call_llm(state["problem_statement"], state["issue_tree"], revision_note)
    parsed = safe_extract_json(raw)

    hypotheses = _ensure_hypotheses(parsed.get("hypotheses", []))

    available_branches = set(_priority_branches(state.get("issue_tree", {})))
    covered_branches = {h.get("branch") for h in hypotheses if h.get("branch")}
    branches_covered = len(available_branches & covered_branches)

    run_logger.record(
        AGENT_NAME,
        hypothesis_count=len(hypotheses),
        branches_available=len(available_branches),
        branches_covered=branches_covered,
    )

    return {
        "hypotheses": hypotheses,
        "messages": [AIMessage(content=raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
