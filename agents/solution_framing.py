"""Solution-framing agent — PM path, stage 3 of 4.

Turns primary research + the competitive audit into candidate solutions.
Each solution states: the problem it addresses, which finding_ids
support that the problem is real, required vs optional (regulation-
mandated is not optional), whether it works within the problem's stated
constraint, whether Zepto already ships it, and what makes it different
from competitors. agents/pm.py then RICE-scores these the same way it
RICE-scores hypothesis-backed features in diagnostic mode.

EVIDENCE DISCIPLINE: finding_ids are validated against the evidence-base
DB in code after parsing — evidence_exists() is the same check the
critic uses for traceability. A model can claim a finding_id that
doesn't exist; code can't be talked into believing it does. This is the
direct answer to "don't let PM mode become a way to skip evidence": a
solution with a hallucinated finding_id ends up with an empty
finding_ids list after validation, which agents/pm.py then treats
exactly like zero evidence — because it is.

THE STATED CONSTRAINT IS ENFORCED FOR PRESENCE, NOT TRUTH: state
['constraint'] (set by agents/structurer.py) is passed into the prompt,
and every solution must explain HOW it operates within that constraint.
Code can only verify the field is non-empty — whether the explanation
actually respects the constraint is a semantic judgment the prompt is
responsible for. A solution with no explanation at all is dropped rather
than silently discarded (see DROPPED, NOT DELETED below).

CURRENT-STATE AWARENESS: state['current_state_findings'] (from
agents/primary_research.py) tells the model what Zepto already ships.
Every solution states status: "new" | "partial" | "exists", citing which
current-state evidence established that when it isn't "new". A solution
marked "exists" does not reach the PRD — recommending something the
company already has is a wasted recommendation.

DROPPED, NOT DELETED: solutions excluded for status=="exists" or a
missing works_within_constraint are moved to state['dropped_solutions']
with a drop_reason, not filtered into nothing. A PM reviewing the final
brief should see what the tool checked and ruled out, not just what
survived — see tools/brief.py's "Considered and dropped" section.

DEDUPLICATION MERGES, NOT DROPS: two solutions covering the same branch
with overlapping finding_ids or near-duplicate wording are the same idea
proposed twice, not two ideas — merged in code (see _dedupe_solutions)
using utils/text_similarity.py's token-overlap primitive, the same one
tools/evidence_extractor.py uses for grounding.

differentiation IS PROSE, NOT A LABEL: observed failure — the model
filled differentiation with the competitive-audit classification word
("differentiator") instead of writing the sentence the prompt asked for.
_validate_differentiation clears (and logs) any differentiation value
that exactly matches one of agents/competitive_audit.py's own
classification labels, since that's a sure sign of the confusion, not a
legitimate answer.

FUNNEL STAGE PER SOLUTION: when agents/funnel_decomposition.py has run,
every solution must state addresses_funnel_stage — a real stage name, or
"undetermined". When state['funnel_verdict'] is itself "undetermined",
the prompt explicitly legitimizes proposing instrumentation ("measure X
before building") as a real solution, not a fallback for having nothing
to say — this is the fix for PM mode producing scattered, untargeted
solutions when nothing establishes where a metric is actually breaking.

INSTRUMENTATION IS CODE-GUARANTEED, NOT JUST PROMPTED FOR: observed
failure — funnel_verdict came back "undetermined" (correctly — the
evidence gap was real) but the prompt's encouragement wasn't enough; no
solution addressed it, and tools/brief.py's funnel callout promised an
instrumentation recommendation that didn't exist. _ensure_instrumentation_solution
now synthesizes one directly from the funnel's own internal-locatability
stages (not invented — built from state the funnel already established)
whenever funnel_verdict=="undetermined" and the LLM didn't propose one
itself, tagged is_instrumentation=True so agents/pm.py and
agents/solution_review.py can each give it their own guarantee in turn
(a guaranteed feature, and immunity from the kill review — see their
docstrings).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import invoke_llm
from tools.db import evidence_exists, get_facts_for_run, get_findings_for_run
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry
from utils.text_similarity import longest_run_ratio, tokenize

AGENT_NAME = "solution_framing"
MAX_SOLUTIONS = 6
ALLOWED_STATUS = {"new", "partial", "exists"}
# differentiation must be prose, not agents/competitive_audit.py's own
# branch-level classification vocabulary -- a value that exactly matches
# one of these is a sure sign the model filled in the label it was just
# shown instead of writing the sentence the prompt actually asked for.
_CLASSIFICATION_LABELS = {"differentiator", "parity_gap", "table_stakes", "not_comparable"}
# Looser than the grounding check's 0.7/6 -- these are short paraphrases
# of the same idea (solution names/problem statements), not page-verbatim
# quotes, so a lower bar catches real duplicates without needing near-
# identical wording.
DEDUP_TEXT_OVERLAP_THRESHOLD = 0.5
DEDUP_MIN_RUN_TOKENS = 4

SYSTEM_PROMPT = """You are a solution-framing lead. Given a business
problem statement, its issue-tree branches, primary-research findings
(what authoritative sources say), current-state findings (what the
company already ships today, from its own product/help-centre/policy
pages), a competitive audit (what named competitors already ship,
classified per branch as table_stakes / differentiator / parity_gap),
a funnel decomposition of the problem's metric with a verdict on where
the evidence points (if this problem has one), and the problem's stated
operating constraint (if any), propose up to 6 candidate solutions.

FUNNEL DISCIPLINE: if a funnel and funnel_verdict are given below, each
solution should target the identified stage specifically — a scattered
solution per stage with nothing establishing WHERE the drop is happening
is the exact failure this exists to prevent. If funnel_verdict is
"undetermined", that means the evidence does not distinguish between
stages — in that case, YOU MUST include at least one solution with
is_instrumentation=true that says what to measure, at which stage(s),
and what result would identify the drop being at that stage. This is a
legitimate, valuable recommendation, not a fallback for having nothing
else to say — do not treat "undetermined" as a reason to fall back to
generic, untargeted build solutions instead.

Each solution MUST state:
  - branch: which issue-tree branch / capability area it addresses
  - problem_addressed: the specific problem this solves, in one sentence
  - finding_ids: the evidence_ids ("finding:<id>" or "fact:<id>") that
    support that this problem is REAL — from the evidence given to you.
    Never invent one. If you have no supporting evidence, leave this
    empty rather than citing something not given to you — an empty list
    is honest; a made-up id is not.
  - required: true if this is regulation-mandated (i.e. the branch's
    primary-research evidence traces to an actual legal/regulatory
    requirement), false if it's optional/discretionary.
  - status: "new" if the current-state findings show no sign the company
    already does this; "partial" if it exists but is incomplete; "exists"
    if the current-state findings show the company already ships this.
    Judge this from the current-state findings given to you, not a guess.
  - current_state_evidence_id: REQUIRED (an evidence_id from the
    current-state findings) whenever status is "partial" or "exists" —
    which specific current-state finding established that. null when
    status is "new".
  - works_within_constraint: if a constraint is given below, this is
    REQUIRED and must specifically explain how the solution operates
    given that constraint — a solution that can't work within a stated
    constraint should not be proposed at all. Empty string only if no
    constraint was given.
  - differentiation: REQUIRED — a SENTENCE explaining what makes this
    solution different from what named competitors ship, or literally
    the string "parity" if it only brings the company to what
    competitors already have. This is prose, NOT a classification label
    — do NOT put "differentiator", "parity_gap", "table_stakes", or
    "not_comparable" here; those are the competitive audit's own branch-
    level classifications, computed and shown separately. Writing one of
    those words here instead of an actual sentence is treated as invalid
    and discarded. PMs build to beat competitors, not just match them —
    say plainly when a solution is only parity.
  - addresses_funnel_stage: REQUIRED whenever a funnel is given below —
    the exact stage name this solution targets, or "undetermined" if it
    doesn't target a specific stage (e.g. an instrumentation solution
    proposed because funnel_verdict itself is "undetermined"). Empty
    string only if no funnel was given at all.
  - is_instrumentation: true if this solution's job is to MEASURE/find
    out where the drop is (not to build a fix), false otherwise. When
    true, also fill in instrumentation_plan with:
      - stage: which stage(s) this measures
      - measure: exactly what to instrument/measure
      - identifying_result: what result would tell you the drop is at
        that stage (vs. an earlier or later one)
    finding_ids may be empty for an is_instrumentation solution — the
    evidence gap itself (funnel_verdict=="undetermined") is the
    justification, not a weakness.

Respond with ONLY a JSON object of this exact shape:
{
  "solutions": [
    {
      "id": "S1",
      "branch": "<issue tree branch>",
      "name": "<short solution name>",
      "problem_addressed": "<the problem this solves>",
      "finding_ids": ["finding:12", "fact:5"],
      "required": true,
      "status": "new|partial|exists",
      "current_state_evidence_id": "finding:8 (or null if status is new)",
      "works_within_constraint": "<how this respects the stated constraint, or empty string>",
      "differentiation": "<what's different from competitors, or the literal string \\"parity\\">",
      "addresses_funnel_stage": "<stage name, or \\"undetermined\\", or empty string if no funnel>",
      "is_instrumentation": false,
      "instrumentation_plan": {"stage": "", "measure": "", "identifying_result": ""}
    }
  ]
}
"""


@with_retry()
def _call_llm(
    problem_statement: str, issue_tree: dict, primary_findings: list[dict], facts: list[dict],
    current_state_findings: list[dict], competitive_audit: dict, constraint: str,
    funnel: dict, funnel_verdict: str, revision_note: str | None,
) -> str:
    evidence = {
        "findings": [{"evidence_id": f"finding:{f['id']}", "claim": f["claim"], "stance": f["stance"]} for f in primary_findings],
        "facts": [{"evidence_id": f"fact:{f['id']}", "metric": f["metric"], "value": f["value"], "unit": f.get("unit")} for f in facts],
    }
    content = (
        f"Problem statement:\n{problem_statement}\n\n"
        f"Constraint:\n{constraint or '(none stated)'}\n\n"
        f"Issue tree:\n{issue_tree}\n\n"
        f"Funnel (may be empty):\n{funnel}\n\n"
        f"Funnel verdict (which stage the evidence points to, or 'undetermined', or empty if no funnel):\n{funnel_verdict or '(none)'}\n\n"
        f"Evidence (authoritative sources):\n{evidence}\n\n"
        f"Current-state findings (what the company already ships today):\n{current_state_findings}\n\n"
        f"Competitive audit:\n{competitive_audit}"
    )
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior solutions — address it:\n{revision_note}"
    response = invoke_llm(AGENT_NAME, [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)], json_mode=True)
    return response.content


def _validate_finding_ids(solution: dict) -> dict:
    """Drop any finding_id that doesn't actually exist in the evidence-base
    DB — a hallucinated id must not be able to make an exploratory
    solution look evidence-backed. See module docstring."""
    raw_ids = solution.get("finding_ids") or []
    valid_ids = [fid for fid in raw_ids if isinstance(fid, str) and evidence_exists(fid)]
    dropped = len(raw_ids) - len(valid_ids)
    if dropped:
        run_logger.record(AGENT_NAME, dropped_finding_ids=dropped, solution_id=solution.get("id"))
    solution["finding_ids"] = valid_ids
    return solution


def _normalize_status(solution: dict) -> dict:
    status = solution.get("status")
    if status not in ALLOWED_STATUS:
        if status is not None:
            run_logger.record(AGENT_NAME, invalid_status=status, solution_id=solution.get("id"))
        status = "new"
    solution["status"] = status
    return solution


def _validate_current_state_citation(solution: dict) -> dict:
    """A "partial"/"exists" status must point at a real current-state
    evidence row — invalid or missing citations are nulled, not treated
    as grounds to drop the solution (the status judgment itself may
    still be right even if the citation was malformed)."""
    if solution.get("status") not in ("partial", "exists"):
        solution["current_state_evidence_id"] = None
        return solution
    eid = solution.get("current_state_evidence_id")
    if not isinstance(eid, str) or not evidence_exists(eid):
        if eid:
            run_logger.record(AGENT_NAME, invalid_current_state_evidence_id=eid, solution_id=solution.get("id"))
        solution["current_state_evidence_id"] = None
    return solution


def _validate_differentiation(solution: dict) -> dict:
    """Reject (clear + log) a differentiation value that's actually a
    competitive-audit classification label rather than the prose sentence
    the prompt asks for — see module SYSTEM_PROMPT. Cleared, not dropped:
    the solution itself may still be sound even if this one field needs
    a human to fill in, same treatment as an invalid current-state citation."""
    value = (solution.get("differentiation") or "").strip()
    if value.lower() in _CLASSIFICATION_LABELS:
        run_logger.record(AGENT_NAME, invalid_differentiation=value, solution_id=solution.get("id"))
        solution["differentiation"] = ""
    return solution


def _normalize_funnel_stage(solution: dict, funnel: dict) -> dict:
    """Never trust an invented stage name -- validated against the
    funnel's real stage names + "undetermined". No funnel at all
    normalizes to "" (the field genuinely doesn't apply)."""
    if not funnel:
        solution["addresses_funnel_stage"] = ""
        return solution
    stage_names = {s.get("name") for s in funnel.get("stages", [])}
    candidate = solution.get("addresses_funnel_stage")
    if candidate not in stage_names and candidate != "undetermined":
        if candidate:
            run_logger.record(AGENT_NAME, invalid_funnel_stage=candidate, solution_id=solution.get("id"))
        candidate = "undetermined"
    solution["addresses_funnel_stage"] = candidate
    return solution


def _normalize_instrumentation_plan(solution: dict, funnel: dict, funnel_verdict: str) -> dict:
    """An "undetermined" verdict means the whole point of this solution
    is finding WHICH stage is broken -- observed failure: the LLM wrote
    instrumentation_plan.stage as a single stage name (e.g. "Checkout
    Initiation") while addresses_funnel_stage correctly said
    "undetermined", directly contradicting itself (instrument one stage
    when you don't yet know which one is broken defeats the purpose).
    Code-guaranteed here, not just prompted for: when funnel_verdict is
    "undetermined", stage is forced to cover every stage in the funnel,
    full stop. When the verdict names a specific stage, the LLM's own
    scoping (that stage and its neighbours, or narrower) is left alone --
    that case isn't a contradiction."""
    if not solution.get("is_instrumentation") or not funnel:
        return solution
    if funnel_verdict != "undetermined":
        return solution
    stage_names = [s.get("name") for s in funnel.get("stages", []) if s.get("name")]
    if not stage_names:
        return solution
    plan = dict(solution.get("instrumentation_plan") or {})
    plan["stage"] = ", ".join(stage_names)
    solution["instrumentation_plan"] = plan
    return solution


def _ensure_instrumentation_solution(solutions: list[dict], funnel: dict, funnel_verdict: str) -> list[dict]:
    """Code-guaranteed, not just prompted for (see module docstring) --
    when funnel_verdict=="undetermined" and the LLM didn't propose an
    is_instrumentation solution itself, synthesize one directly from the
    funnel's own stages. Built from state already established by
    agents/funnel_decomposition.py, not invented.

    Targets EVERY stage, not just internal-locatability ones: this path
    only ever runs when funnel_verdict=="undetermined", i.e. the drop
    could be at ANY stage -- narrowing to internal-only stages here would
    be the exact same contradiction _normalize_instrumentation_plan
    exists to prevent for LLM-authored solutions."""
    if funnel_verdict != "undetermined" or not funnel:
        return solutions
    if any(s.get("is_instrumentation") for s in solutions):
        return solutions

    stages = funnel.get("stages", [])
    stage_names = ", ".join(s.get("name", "") for s in stages) or "the funnel"
    measures = "; ".join(
        f"{s.get('name', '')}: {s.get('evidence_needed') or 'instrument this stage directly'}"
        for s in stages
    )

    fallback_id = f"S{len(solutions) + 1}"
    # Inserted FIRST, not appended -- _ensure_solutions below caps the
    # list at MAX_SOLUTIONS by keeping the first N, and this guarantee
    # must survive that cap regardless of how many other solutions the
    # LLM proposed.
    solutions.insert(0, {
        "id": fallback_id,
        "branch": None,
        "name": f"Instrument the funnel ({stage_names})",
        "problem_addressed": (
            f"The gathered evidence does not distinguish which stage of {funnel.get('metric', 'the metric')!r} "
            f"the drop is at — building before knowing where would be a guess, not a fix."
        ),
        "finding_ids": [],
        "required": False,
        "status": "new",
        "current_state_evidence_id": None,
        "works_within_constraint": "",
        "differentiation": "",
        "addresses_funnel_stage": "undetermined",
        "is_instrumentation": True,
        "instrumentation_plan": {
            "stage": stage_names,
            "measure": measures,
            "identifying_result": "A clear drop-off concentrated at one stage, isolating it from the others.",
        },
    })
    run_logger.record(AGENT_NAME, instrumentation_solution_synthesized=fallback_id)
    return solutions


def _split_dropped(solutions: list[dict], constraint: str) -> tuple[list[dict], list[dict]]:
    """Solutions that don't reach the PRD are DROPPED, not deleted — kept
    with a drop_reason so tools/brief.py can show a PM what was checked
    and ruled out, not just what's missing."""
    kept: list[dict] = []
    dropped: list[dict] = []
    for solution in solutions:
        if solution.get("status") == "exists":
            solution["drop_reason"] = "already exists"
            dropped.append(solution)
            continue
        # An instrumentation solution measures; it doesn't build anything,
        # so a build-time constraint (e.g. "cannot increase incentive
        # spend") doesn't apply to it -- exempt from this check rather
        # than dropping the one recommendation the "undetermined" verdict
        # guarantees exists.
        if constraint and not solution.get("is_instrumentation") and not (solution.get("works_within_constraint") or "").strip():
            solution["drop_reason"] = f"does not state how it works within the stated constraint: {constraint}"
            dropped.append(solution)
            continue
        kept.append(solution)
    return kept, dropped


def _similar_text(a: str, b: str) -> bool:
    ratio, run_len = longest_run_ratio(tokenize(a), tokenize(b))
    return ratio >= DEDUP_TEXT_OVERLAP_THRESHOLD and run_len >= DEDUP_MIN_RUN_TOKENS


def _dedupe_solutions(solutions: list[dict]) -> list[dict]:
    """Two solutions on the same branch that share evidence or near-
    duplicate wording are the same idea proposed twice — merge (union
    finding_ids, keep the first id) rather than silently dropping one,
    which would look like the idea was never considered at all."""
    merged: list[dict] = []
    for solution in solutions:
        match = None
        for existing in merged:
            if existing.get("branch") != solution.get("branch"):
                continue
            shared_evidence = bool(set(existing.get("finding_ids", [])) & set(solution.get("finding_ids", [])))
            text_overlap = (
                _similar_text(existing.get("problem_addressed", ""), solution.get("problem_addressed", ""))
                or _similar_text(existing.get("name", ""), solution.get("name", ""))
            )
            if shared_evidence or text_overlap:
                match = existing
                break
        if match is None:
            merged.append(solution)
            continue
        match["finding_ids"] = sorted(set(match.get("finding_ids", [])) | set(solution.get("finding_ids", [])))
        match.setdefault("merged_from", []).append(solution.get("id"))
        run_logger.record(AGENT_NAME, duplicate_solution_merged=solution.get("id"), merged_into=match.get("id"))
    return merged


def _ensure_solutions(solutions: list[dict]) -> list[dict]:
    """Same "ran but empty" guarantee used throughout — `not solutions`
    must reliably mean "hasn't run", not "ran and proposed nothing"."""
    if solutions:
        return solutions[:MAX_SOLUTIONS]
    return [
        {
            "id": "S0",
            "branch": None,
            "name": "No candidate solution could be framed",
            "problem_addressed": "No evidence supported a concrete solution for any branch.",
            "finding_ids": [],
            "required": False,
            "status": "new",
            "current_state_evidence_id": None,
            "works_within_constraint": "",
            "differentiation": "",
            "addresses_funnel_stage": "",
            "is_instrumentation": False,
            "instrumentation_plan": {},
            "unresolved": True,
        }
    ]


def solution_framing_node(state: dict) -> dict:
    """Turn primary research + current state + competitive audit into
    candidate solutions that respect the stated constraint and aren't
    something the company already ships.

    Reads: state['problem_statement'], state['issue_tree'], state['run_id'],
    state['constraint'], state['current_state_findings'], state['competitive_audit'],
    state['funnel'], state['funnel_verdict'], state['revision_notes']['solution_framing']
    Writes: state['solutions'] (kept, deduped, PRD-bound; each with
    addresses_funnel_stage, is_instrumentation, instrumentation_plan — an
    instrumentation solution is code-guaranteed present whenever
    funnel_verdict=="undetermined"), state['dropped_solutions'] (excluded,
    each with a drop_reason), state['messages'], state['run_path']
    """
    problem_statement = state.get("problem_statement", "")
    issue_tree = state.get("issue_tree", {})
    run_id = state.get("run_id")
    constraint = state.get("constraint", "")
    current_state_findings = state.get("current_state_findings", [])
    competitive_audit = state.get("competitive_audit", {})
    funnel = state.get("funnel", {})
    funnel_verdict = state.get("funnel_verdict", "")
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    findings = get_findings_for_run(run_id) if run_id else []
    facts = get_facts_for_run(run_id) if run_id else []

    raw = _call_llm(
        problem_statement, issue_tree, findings, facts, current_state_findings,
        competitive_audit, constraint, funnel, funnel_verdict, revision_note,
    )
    parsed = safe_extract_json(raw)

    solutions = parsed.get("solutions", [])
    solutions = [_validate_finding_ids(s) for s in solutions]
    solutions = [_normalize_status(s) for s in solutions]
    solutions = [_validate_current_state_citation(s) for s in solutions]
    solutions = [_validate_differentiation(s) for s in solutions]
    solutions = [_normalize_funnel_stage(s, funnel) for s in solutions]
    solutions = [_normalize_instrumentation_plan(s, funnel, funnel_verdict) for s in solutions]

    kept, dropped = _split_dropped(solutions, constraint)
    kept = _dedupe_solutions(kept)
    kept = _ensure_instrumentation_solution(kept, funnel, funnel_verdict)
    kept = _ensure_solutions(kept)

    run_logger.record(
        AGENT_NAME,
        solution_count=len(kept),
        dropped_count=len(dropped),
        required_count=sum(1 for s in kept if s.get("required")),
        unevidenced_count=sum(1 for s in kept if not s.get("finding_ids")),
    )

    return {
        "solutions": kept,
        "dropped_solutions": dropped,
        "messages": [AIMessage(content=raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
