"""Structurer agent — builds a MECE issue tree, classifies problem type,
classifies routing_mode, and derives anchor terms.

ROUTING MODE exists because "structurer -> hypothesis -> researcher ->
analyst -> synthesizer" is a machine for answering "why did this
happen?" — every step exists to find and verify a root cause. A PM
problem doesn't have that shape: the cause is either given as a premise
("orders dropped 3%") or irrelevant ("what should we build for DPDP
compliance?"). Running the diagnostic chain against a PM problem makes
it invent diagnostic questions nobody asked, fail to find public
evidence for them, and correctly report inconclusive — which is what
produced PRDs with `addresses_hypothesis: null` on every feature. The
fix is routing: classify what KIND of problem this is before deciding
which agents even run (see agents/engagement_manager.py's mode-specific
ladders).

ANCHOR TERMS (sector/geography/company) used to be derived independently
inside agents/researcher.py's query planner. They're computed here
instead, once, because the PM path (agents/primary_research.py,
agents/competitive_audit.py) needs the identical anchors and re-deriving
them per agent would mean inconsistent anchors and wasted LLM calls for
the same information. agents/researcher.py still derives its own (see
its module docstring) — moving it over to state['anchor_terms'] too is a
reasonable follow-up but out of scope for this change.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_llm
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry

AGENT_NAME = "structurer"
ALLOWED_PROBLEM_TYPES = {"product", "market", "operations", "strategy", "financial"}
DEFAULT_PROBLEM_TYPE = "strategy"

ALLOWED_ROUTING_MODES = {"diagnostic", "pm", "combined"}
DEFAULT_ROUTING_MODE = "diagnostic"

logger = logging.getLogger("business_copilot.structurer")

SYSTEM_PROMPT = """You are a management-consulting structuring specialist.

Given a business problem statement, produce:

1. A MECE (Mutually Exclusive, Collectively Exhaustive) issue tree that
   breaks the problem into 2-4 top-level branches, each with 2-4
   sub-branches. Branches must not overlap and must together cover the
   whole problem.

2. A classification of the problem type: one of "product", "market",
   "operations", "strategy", "financial". Use "product" whenever the
   problem centers on product features, user experience, or roadmap
   decisions — this triggers a downstream PM review.

3. A classification of routing_mode: one of "diagnostic", "pm",
   "combined".
   - "diagnostic": the problem asks WHY something happened and a root
     cause must be found. Signals: "why", "diagnose", "what's driving",
     "root cause".
   - "pm": the cause is already GIVEN as a premise — a stated fact the
     problem does NOT ask you to verify — or the ask is what to build,
     how to improve something, how to comply with something, or
     responds to a named regulation or competitor. Signals: "what
     should we build", "how do we improve", "recommend an approach",
     "how do we comply", a named regulation, a named competitor.
   - "combined": the problem clearly needs BOTH a root cause AND a set
     of recommended measures.

4. Anchor terms for downstream search/relevance filtering, in three
   categories:
   - sector: the industry/sector — REQUIRED, always provide a specific
     one. "Specialty chemicals" is usable; "chemicals" alone is too
     broad.
   - geography: the region/country, if named. If the problem statement
     doesn't literally name a country but references a sub-national
     locality tier or market descriptor (e.g. "Tier-2 cities", "metro
     vs. non-metro"), infer the country from other context if it's
     identifiable (a named company, currency, named cities) and include
     both, e.g. "India (Tier-2 cities)" — a locality-tier phrase without
     an inferred country is a missed signal, not genuinely absent
     information. Empty string only if truly nothing is inferable.
   - company: the specific company name, if one is named (empty string
     if the problem describes a generic/anonymized entity instead).

5. The single most binding operating constraint explicitly stated in the
   problem — a budget, timeline, headcount, or "we are NOT doing X"
   limitation that any recommendation must respect (e.g. "not adding
   dark stores in the next two quarters"). State it near-verbatim from
   the problem statement. Empty string if no such constraint is stated —
   do not invent one.

Respond with ONLY a JSON object of this exact shape:
{
  "problem_type": "product|market|operations|strategy|financial",
  "routing_mode": "diagnostic|pm|combined",
  "anchor_terms": {"sector": "<sector term>", "geography": "<geography term or empty string>", "company": "<company term or empty string>"},
  "constraint": "<the binding operating constraint, near-verbatim, or empty string>",
  "issue_tree": {
    "root": "<restated problem>",
    "branches": [
      {"name": "<branch name>", "sub_branches": ["<sub-branch>", "..."]}
    ]
  },
  "rationale": "<1-3 sentences on why this decomposition, problem_type, and routing_mode>"
}
"""


@with_retry()
def _call_llm(problem_statement: str, revision_note: str | None) -> str:
    llm = get_llm(AGENT_NAME)
    content = f"Business problem statement:\n{problem_statement}"
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior structuring — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _normalize_problem_type(raw_value) -> str:
    if isinstance(raw_value, str):
        candidate = raw_value.strip().lower()
        if candidate in ALLOWED_PROBLEM_TYPES:
            return candidate
    logger.warning("problem_type %r not in %s — falling back to %r", raw_value, ALLOWED_PROBLEM_TYPES, DEFAULT_PROBLEM_TYPE)
    return DEFAULT_PROBLEM_TYPE


def _normalize_routing_mode(raw_value) -> str:
    if isinstance(raw_value, str):
        candidate = raw_value.strip().lower()
        if candidate in ALLOWED_ROUTING_MODES:
            return candidate
    logger.warning("routing_mode %r not in %s — falling back to %r", raw_value, ALLOWED_ROUTING_MODES, DEFAULT_ROUTING_MODE)
    return DEFAULT_ROUTING_MODE


def _normalize_anchor_terms(raw_anchors) -> dict:
    if not isinstance(raw_anchors, dict):
        raw_anchors = {}
    return {
        "sector": str(raw_anchors.get("sector") or "").strip(),
        "geography": str(raw_anchors.get("geography") or "").strip(),
        "company": str(raw_anchors.get("company") or "").strip(),
    }


def _normalize_constraint(raw_value) -> str:
    if isinstance(raw_value, str):
        return raw_value.strip()
    return ""


def _ensure_issue_tree(issue_tree: dict, problem_statement: str) -> dict:
    """Guarantee a non-empty, at-least-one-branch tree so `not issue_tree`
    reliably means "structurer hasn't run" rather than "ran and produced
    nothing usable" — the two must not look the same to the ladder."""
    if isinstance(issue_tree, dict) and issue_tree.get("branches"):
        return issue_tree
    return {
        "root": problem_statement,
        "branches": [
            {
                "name": "unstructured",
                "sub_branches": [],
                "note": "structuring did not produce a real MECE decomposition; treating the whole problem as one branch",
            }
        ],
    }


def structurer_node(state: dict) -> dict:
    """Build a MECE issue tree; classify problem_type, routing_mode, and anchor_terms.

    Reads: state['problem_statement'], state['revision_notes']['structurer']
    Writes: state['issue_tree'], state['problem_type'], state['routing_mode'],
    state['anchor_terms'], state['constraint'], state['messages'], state['run_path']
    """
    problem_statement = state["problem_statement"]
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    raw = _call_llm(problem_statement, revision_note)
    parsed = safe_extract_json(raw)

    issue_tree = _ensure_issue_tree(parsed.get("issue_tree", {}), problem_statement)
    problem_type = _normalize_problem_type(parsed.get("problem_type"))
    routing_mode = _normalize_routing_mode(parsed.get("routing_mode"))
    anchor_terms = _normalize_anchor_terms(parsed.get("anchor_terms"))
    constraint = _normalize_constraint(parsed.get("constraint"))

    run_logger.record(
        AGENT_NAME, problem_type=problem_type, routing_mode=routing_mode,
        anchor_terms=anchor_terms, constraint=constraint,
    )

    return {
        "issue_tree": issue_tree,
        "problem_type": problem_type,
        "routing_mode": routing_mode,
        "anchor_terms": anchor_terms,
        "constraint": constraint,
        "messages": [AIMessage(content=raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
