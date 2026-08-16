"""engagement_manager_node: the supervisor. Every worker agent returns
control here (flat supervisor pattern); this decides what runs next.

MODE-SPECIFIC LADDERS: routing_mode (set once by agents/structurer.py,
read here once per invocation) selects which forward ladder applies —
_DIAGNOSTIC_LADDER, _PM_LADDER, or _COMBINED_LADDER — rather than one
ladder with routing_mode checks threaded through every step. Each ladder
is just an ordered list of (state_key, next_agent, reason) triples:
walk it in order, route to the first agent whose state_key is still
falsy. "combined" mode's ladder is the diagnostic steps followed by the
PM steps, since it needs both a root cause and a set of measures.

Loop-backs (branch 2 below) still work exactly the same regardless of
mode — DOWNSTREAM's reset list and ORDER's earliest-agent-wins logic now
just also cover the five PM-path agents (funnel_decomposition,
primary_research, competitive_audit, solution_framing, solution_review).
"""

from state import EngagementState
from langchain_core.messages import AIMessage

MAX_ITER = 3
FINISH = "FINISH"

# Looping back to an agent invalidates everything downstream of it.
# Without this reset the forward ladder sees stale output, skips the
# rework, and the same critic flags fire forever.
#
# "critic_passed" is included in every list below — it wasn't in earlier
# versions of this dict, which was a live bug: critic_passed is only
# ever set True/False by critic_node, but nothing reset it on a
# loop-back, so a stale True from a PREVIOUS critic pass would satisfy
# branch 3 ("critic ran and passed") immediately after a loop-back reset
# — before the just-invalidated downstream agents ever got to re-run.
# Concretely: flags fire -> route back to researcher -> critique cleared
# -> researcher re-runs -> back to EM -> critique is empty (branch 2
# skipped) -> critic_passed is STILL True from before -> branch 3 fires
# -> FINISH, without analyst/synthesizer/critic ever re-running. Every
# loop-back was silently a no-op. Fixed by resetting critic_passed
# alongside critique on every target.
DOWNSTREAM = {
    "structurer":  ["hypotheses", "research_findings", "analysis_results",
                    "funnel", "funnel_verdict",
                    "primary_research_findings", "current_state_findings",
                    "competitive_audit", "solutions", "dropped_solutions",
                    "synthesis", "prd", "solutions_reviewed", "critique", "critic_passed"],
    "hypothesis":  ["research_findings", "analysis_results",
                    "synthesis", "prd", "solutions_reviewed", "critique", "critic_passed"],
    "researcher":  ["analysis_results", "synthesis", "prd", "solutions_reviewed", "critique", "critic_passed"],
    "analyst":     ["synthesis", "prd", "solutions_reviewed", "critique", "critic_passed"],
    "synthesizer": ["prd", "solutions_reviewed", "critique", "critic_passed"],
    # funnel_verdict is primary_research's supplementary output (like
    # current_state_findings) -- resetting funnel_decomposition itself
    # invalidates it (below), but a loop-back TO primary_research never
    # resets primary_research's own output, funnel_verdict included.
    "funnel_decomposition": ["primary_research_findings", "current_state_findings", "funnel_verdict",
                              "competitive_audit", "solutions", "dropped_solutions",
                              "prd", "solutions_reviewed", "critique", "critic_passed"],
    "primary_research": ["competitive_audit", "solutions", "dropped_solutions",
                          "prd", "solutions_reviewed", "critique", "critic_passed"],
    "competitive_audit": ["solutions", "dropped_solutions", "prd", "solutions_reviewed", "critique", "critic_passed"],
    "solution_framing":  ["prd", "solutions_reviewed", "critique", "critic_passed"],
    "pm":          ["solutions_reviewed", "critique", "critic_passed"],
    "solution_review": ["critique", "critic_passed"],
}

# Earliest-first, so we can fix the root cause rather than the symptom.
# Covers both tracks — a flag's suggested_action will only ever name an
# agent from the track that actually ran, but a single shared list keeps
# _target_for simple regardless of mode.
ORDER = ["structurer", "hypothesis", "researcher", "analyst", "synthesizer",
         "funnel_decomposition", "primary_research", "competitive_audit",
         "solution_framing", "pm", "solution_review"]

EMPTY = {
    "hypotheses": [], "research_findings": [], "analysis_results": [],
    "funnel": {}, "funnel_verdict": "",
    "primary_research_findings": [], "current_state_findings": [],
    "competitive_audit": {}, "solutions": [], "dropped_solutions": [],
    "critic_passed": False,
    "synthesis": {}, "prd": None, "solutions_reviewed": False, "critique": [],
}

# Each ladder is an ordered list of (state_key, next_agent, reason).
# Walked top to bottom; the first entry whose state_key is still falsy
# wins. "structurer" isn't in these lists — it's handled as a shared
# first step in engagement_manager_node before the mode-specific ladder
# even applies, since routing_mode itself doesn't exist until structurer
# has run once.
_DIAGNOSTIC_STEPS = [
    ("hypotheses", "hypothesis", "Issue tree ready; generating hypotheses."),
    ("research_findings", "researcher", "Hypotheses ready; gathering evidence."),
    ("analysis_results", "analyst", "Evidence gathered; running quantitative analysis."),
    ("synthesis", "synthesizer", "Analysis done; synthesizing verdicts."),
]
_PM_STEPS = [
    ("funnel", "funnel_decomposition", "PM problem; decomposing the funnel before jumping to solutions."),
    ("primary_research_findings", "primary_research", "Targeting authoritative sources for the ask."),
    ("competitive_audit", "competitive_audit", "Checking what named competitors already ship."),
    ("solutions", "solution_framing", "Framing candidate solutions from research + audit."),
]

_LADDERS = {
    "diagnostic": _DIAGNOSTIC_STEPS,
    "pm": _PM_STEPS,
    "combined": _DIAGNOSTIC_STEPS + _PM_STEPS,
}


def _next_in_ladder(state: EngagementState, steps: list[tuple[str, str, str]]) -> tuple[str, str] | None:
    for state_key, agent, reason in steps:
        if not state[state_key]:
            return agent, reason
    return None


def engagement_manager_node(state: EngagementState) -> dict:
    flags = state["critique"]

    # 1. Iteration cap reached with flags outstanding — ship, disclosed.
    if flags and state["iteration_count"] >= MAX_ITER:
        return _decide(state, FINISH,
                       f"Iteration cap ({MAX_ITER}) reached with "
                       f"{len(flags)} unresolved flag(s); finishing with "
                       f"limitations disclosed.")

    # 2. Material flags — send work back to the earliest responsible agent.
    if flags:
        target = _target_for(flags)
        update = _decide(state, target,
                         f"Critic raised {len(flags)} flag(s); routing back "
                         f"to {target} (attempt {state['iteration_count'] + 1}).")
        update["iteration_count"] = state["iteration_count"] + 1
        update["revision_notes"] = _notes_for(flags)
        for field in DOWNSTREAM.get(target, []):
            update[field] = EMPTY[field]
        return update

    # 3. Critic ran and passed.
    if state["critic_passed"]:
        return _decide(state, FINISH, "Critic passed with no material flags.")

    # 4. No issue tree yet — this step is shared by every mode, since
    # routing_mode itself is only known once structurer has run.
    if not state["issue_tree"]:
        return _decide(state, "structurer", "No issue tree yet; structuring the problem.")

    # 5. Mode-specific forward ladder.
    steps = _LADDERS.get(state["routing_mode"], _DIAGNOSTIC_STEPS)
    ladder_result = _next_in_ladder(state, steps)
    if ladder_result is not None:
        nxt, why = ladder_result
        return _decide(state, nxt, why)

    # 6. Ladder fully walked — PM gate, solution review, then critic.
    if state["problem_type"] == "product" and state["prd"] is None:
        return _decide(state, "pm", "Product-type problem; drafting PRD.")
    if state["routing_mode"] in ("pm", "combined") and state["prd"] is None:
        return _decide(state, "pm", "PM-mode problem; drafting PRD.")
    if state["routing_mode"] in ("pm", "combined") and state["prd"] is not None and not state["solutions_reviewed"]:
        return _decide(state, "solution_review",
                        "PRD drafted; reviewing each solution for evidence/mechanism/effort "
                        "credibility before it ships.")

    return _decide(state, "critic", "Deliverables complete; sending for review.")


def _decide(state: EngagementState, target: str, rationale: str) -> dict:
    update = {
        "next_agent": target,
        "run_path": ["engagement_manager"],
        "messages": [AIMessage(content=f"[EM → {target}] {rationale}",
                               name="engagement_manager")],
    }
    if target == FINISH:
        update["final_brief"] = _assemble(state)
    return update


def _target_for(flags: list[dict]) -> str:
    """Earliest agent named by any flag — fix the root, not the symptom."""
    named = {f.get("suggested_action") for f in flags}
    for agent in ORDER:
        if agent in named:
            return agent
    return "synthesizer"


def _notes_for(flags: list[dict]) -> dict:
    notes: dict[str, list[str]] = {}
    for f in flags:
        notes.setdefault(f.get("suggested_action", "synthesizer"), []) \
             .append(f.get("issue", ""))
    return notes


def _assemble(state: EngagementState) -> dict:
    brief = {
        "problem_statement": state["problem_statement"],
        "problem_type": state["problem_type"],
        "routing_mode": state["routing_mode"],
        "issue_tree": state["issue_tree"],
        "prd": state["prd"],
        "limitations": state["critique"],
        "run_path": state["run_path"],
    }
    if state["routing_mode"] in ("diagnostic", "combined"):
        brief["verdicts"] = state["synthesis"].get("verdicts", [])
        brief["recommendation"] = state["synthesis"].get("answer", "")
        brief["caveats"] = state["synthesis"].get("caveats", "")
    if state["routing_mode"] in ("pm", "combined"):
        brief["solutions"] = state["solutions"]
        brief["competitive_audit"] = state["competitive_audit"]
    return brief
