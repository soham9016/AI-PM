"""EngagementState: the shared state schema threaded through every node
in the graph.

Two tracks after agents/structurer.py classifies routing_mode (see its
module docstring): the ORIGINAL diagnostic track (issue_tree ->
hypotheses -> research_findings -> analysis_results -> synthesis -> prd)
answers "why did this happen"; the PM track (issue_tree ->
primary_research_findings -> competitive_audit -> solutions -> prd)
answers "what should we do" without inventing a diagnosis nobody asked
for. "combined" mode runs both tracks before pm/critic. See
agents/engagement_manager.py for the mode-specific ladders that decide
which fields get filled in which order.

    run_id:                    str    — stamps evidence-base rows for this run
    problem_statement:         str    — the initial input
    problem_type:              str    — product|market|operations|strategy|financial
    routing_mode:               str    — diagnostic|pm|combined
    anchor_terms:                dict   — {sector, geography, company}, computed once
                                          by structurer, reused by researcher/
                                          primary_research/competitive_audit
    constraint:                  str    — the single most binding operating
                                          constraint stated in the problem (e.g.
                                          "not adding dark stores"), computed by
                                          structurer, empty string if none stated
    issue_tree:                  dict
    hypotheses:                  list[dict]   — diagnostic track
    research_findings:            list[dict]   — diagnostic track
    analysis_results:             list[dict]   — diagnostic track
    funnel:                       dict          — PM track, {metric, stages}: the PM-mode
                                                  analogue of a hypothesis -- decomposes the
                                                  stated metric into stages before solutions
                                                  are proposed, from agents/funnel_decomposition.py
    funnel_verdict:                str           — PM track, which funnel stage the gathered
                                                  evidence points to, or "undetermined";
                                                  judged by agents/primary_research.py
    primary_research_findings:    list[dict]   — PM track (authoritative-source findings)
    current_state_findings:       list[dict]   — PM track, what the company already
                                                  ships today, from primary_research
    competitive_audit:            dict          — PM track
    solutions:                    list[dict]   — PM track, kept/PRD-bound candidates
    dropped_solutions:            list[dict]   — PM track, excluded candidates (already
                                                  exists, violates the stated constraint, or
                                                  killed by agents/solution_review.py) with a
                                                  drop_reason each
    synthesis:                    dict          — diagnostic track, read by pm/critic
    prd:                          dict | None  — pm, either track
    solutions_reviewed:            bool          — True once agents/solution_review.py has run
                                                  (PM/combined only) -- the ladder gate between
                                                  pm and critic
    critique:                     list[dict]   — critic
    critic_passed:                bool          — True iff critique ends up empty
    revision_notes:               dict          — critic flags, keyed by target agent
    final_brief:                  dict          — terminal assembled output
    next_agent:                   str           — engagement_manager's routing decision
    iteration_count:              int           — loop-back cap counter
    messages:                     list          — Annotated with add_messages
    run_path:                     list[str]     — Annotated with operator.add
"""

from typing import Annotated, TypedDict
import operator
from langgraph.graph.message import add_messages
from utils.run_id import generate_run_id

class EngagementState(TypedDict):
    run_id: str
    problem_statement: str
    problem_type: str
    routing_mode: str
    anchor_terms: dict
    constraint: str
    issue_tree: dict
    hypotheses: list[dict]
    research_findings: list[dict]
    analysis_results: list[dict]
    funnel: dict
    funnel_verdict: str
    primary_research_findings: list[dict]
    current_state_findings: list[dict]
    competitive_audit: dict
    solutions: list[dict]
    dropped_solutions: list[dict]
    synthesis: dict
    prd: dict | None
    solutions_reviewed: bool
    critic_passed: bool
    critique: list[dict]
    revision_notes: dict
    final_brief: dict
    next_agent: str
    iteration_count: int
    messages: Annotated[list, add_messages]
    run_path: Annotated[list[str], operator.add]

def new_state(problem_statement: str) -> EngagementState:
    return {
        "run_id": generate_run_id(),
        "problem_statement": problem_statement,
        "problem_type": "",
        "routing_mode": "",
        "anchor_terms": {"sector": "", "geography": "", "company": ""},
        "constraint": "",
        "issue_tree": {},
        "hypotheses": [],
        "research_findings": [],
        "analysis_results": [],
        "funnel": {},
        "funnel_verdict": "",
        "primary_research_findings": [],
        "current_state_findings": [],
        "competitive_audit": {},
        "solutions": [],
        "dropped_solutions": [],
        "synthesis": {},
        "prd": None,
        "solutions_reviewed": False,
        "critic_passed": False,
        "critique": [],
        "revision_notes": {},
        "final_brief": {},
        "next_agent": "",
        "iteration_count": 0,
        "messages": [],
        "run_path": [],
    }