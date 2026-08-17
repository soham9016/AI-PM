"""Solution-review agent — kill authority. Runs after pm, before critic.

WHY AFTER pm, NOT BEFORE (despite "kill a solution before it reaches the
PRD" reading that way at first): criterion (c) below needs the feature's
actual reach/impact/effort/rice_score, which don't exist until pm_node
has authored features from solutions — a solution alone carries no RICE
inputs. Criterion (b) needs a real target metric, best served by
prd['north_star_metric'] rather than approximating one before it exists.
"Before it reaches the PRD" is honored from the READER's side: a killed
solution is stripped out of prd['features'] and moved to
state['dropped_solutions'] before the brief ever renders anything — pm_node
scoring it first internally is invisible to the delivered output.

THREE TESTS, ONE VERDICT PER SOLUTION (promote / keep / kill):
  a) EVIDENCE INTEGRITY — kill ONLY if the evidence actively CONTRADICTS
     the solution, or the solution has ZERO supporting evidence of any
     kind. This is deliberately NOT "does the evidence prove this exact
     problem exists" — the public sources this system can reach (app-
     store reviews, forum posts, industry commentary) give SIGNALS, not
     statistical proof, and demanding proof from a source that only ever
     offers signal is rigid, not rigorous. Observed failure (the fix this
     replaces): a criterion asking "does this evidence show THIS PROBLEM
     EXISTS for this company" killed 4 of 5 candidate solutions on a real
     run, with reasons like "the source only lists available payment
     methods; it does not prove users abandon because preferred methods
     are missing" — that source was never going to meet a proof standard,
     and a PM reading a payment-method complaint doesn't discard it, they
     turn it into a testable hypothesis. A surviving solution instead gets
     a signal_strength label (direct / adjacent / inferred — see
     SYSTEM_PROMPT) so the reader can see how strong the evidentiary basis
     actually is, without the solution being deleted over it.
  b) MECHANISM — does the feature's mechanism plausibly move the target
     metric FOR THE POPULATION IT ACTUALLY REACHES? "Better streaming
     quality -> more first-time adoption" is a weak chain: streaming
     quality affects people who already use the feature, not people
     deciding whether to try it for the first time.
  c) EFFORT CREDIBILITY — is the effort estimate plausible for the scope
     described? effort=1 for infrastructure/quality work is what put a
     badly-supported feature at the top of a RICE-sorted list.

kill: evidence CONTRADICTS the solution, or there is ZERO supporting
evidence, or (b) fails badly, or the effort estimate isn't credible
enough to trust the RICE score at all. Killed solutions are DROPPED, NOT
DELETED — same treatment as "already exists" and constraint violations
(see agents/solution_framing.py): moved to state['dropped_solutions']
with drop_reason="solution review: <reason>", rendered in tools/brief.py's
"Considered and dropped" -> "Killed on review" section, not silently
discarded.

MOSCOW AND THE HEADER ARE RECOMPUTED AFTER KILLS, NOT INHERITED:
agents/pm.py's _assign_moscow/_build_header are both relative to a
feature SET (the RICE floor/median are fractions of the top score in
play; the header's counts are fractions of how many features exist) —
inheriting pm_node's pre-kill values would describe a set that no longer
exists. Observed failure: two real surviving features were mislabeled
"Won't" against a RICE floor derived from a feature this node had just
killed. Both are re-run here against the post-kill feature list before
this node returns.

INSTRUMENTATION SOLUTIONS ARE NEVER SENT TO THE KILL REVIEW: an
is_instrumentation solution (see agents/solution_framing.py) is
evidence-appropriate BY CONSTRUCTION — the whole reason it exists is that
the evidence gap (funnel_verdict=="undetermined") couldn't be resolved,
so judging it by "does the evidence show this problem is real" would be
backwards. _build_candidates excludes these from the LLM candidate list
entirely and auto-promotes them in code — not a prompt instruction the
model could get wrong, a structural guarantee it never reaches the model
at all.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_llm
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry

_AUTO_PROMOTE_REASON = (
    "Instrumentation solution — the evidence gap that produced funnel_verdict='undetermined' "
    "is itself the justification; not subject to kill review by construction."
)

AGENT_NAME = "solution_review"
ALLOWED_VERDICTS = {"promote", "keep", "kill"}
ALLOWED_SIGNAL_STRENGTHS = {"direct", "adjacent", "inferred"}
DEFAULT_SIGNAL_STRENGTH = "inferred"  # safe/conservative default when the LLM omits or misclassifies

SYSTEM_PROMPT = """You are a skeptical-but-fair PM doing a final review
before a PRD ships. For each candidate below, decide: promote / keep /
kill.

Apply three tests:
a) EVIDENCE INTEGRITY: kill ONLY if the cited evidence actively
   CONTRADICTS the solution's premise, or the solution has ZERO
   supporting evidence of any kind. Do NOT kill just because the
   evidence falls short of proving the problem outright — the sources
   available here (app-store reviews, forum posts, industry commentary)
   give SIGNALS, not statistical proof, and were never going to meet a
   proof standard. Turning a real signal into a testable hypothesis is
   the job; rejecting it for not already being the answer is not rigor,
   it's rigid thinking.

   If the solution survives this test, label its signal_strength:
     - "direct": the evidence names this specific problem/friction point
       for this company.
     - "adjacent": the evidence shows a real, related friction point that
       plausibly bears on the stated problem, even though it doesn't name
       it directly. Example: evidence says "users complain heavily about
       return and exchange policy" — a solution proposing return-policy
       reassurance at the Cart stage (to reduce commitment anxiety and
       improve Proceed-to-Checkout rate) is ADJACENT, not a kill, even
       though the evidence never says "checkout."
     - "inferred": the evidence is general to the industry/category, not
       specific to this company, but is still a reasonable basis for a
       hypothesis.
   Each candidate's evidence_rationale (written by the solution-framing
   step) already explains its author's reasoning for how the cited
   evidence bears on the problem — read it before judging; it often
   already states the inferential step that makes a signal "adjacent"
   rather than "direct."
b) MECHANISM: does the feature's mechanism plausibly move the target
   metric, for the population it actually reaches? A feature that only
   affects existing/active users cannot plausibly move a first-time-
   adoption metric, for example — trace the causal chain explicitly
   rather than assuming a plausible-sounding connection is a real one.
c) EFFORT CREDIBILITY: is the effort estimate (in person-months) credible
   for the scope described? Infrastructure/quality work claiming a very
   low effort is not credible and should be treated as understated, not
   taken at face value — that alone is grounds to distrust the RICE score.

kill: evidence CONTRADICTS the solution, or there is ZERO supporting
evidence, or (b) fails badly, or the effort estimate isn't credible
enough to trust the RICE score at all.
keep: survives (a) with signal_strength "adjacent" or "inferred", or (b)/
(c) are only weakly credible — ships, but weaker than "promote".
promote: signal_strength is "direct", the mechanism is a clear, traceable
path to the target metric, and the effort estimate is credible.

Respond with ONLY a JSON object of this exact shape:
{
  "verdicts": [
    {"solution_id": "S1", "verdict": "promote|keep|kill", "signal_strength": "direct|adjacent|inferred", "reason": "<1-3 sentences>"}
  ]
}
signal_strength is required for "promote"/"keep" verdicts; omit it (or
leave empty) for "kill".
"""


def _normalize_signal_strength(raw_value) -> str:
    """Never trust an invented label -- validated against the fixed
    allowed set, falling back to the most conservative claim (DEFAULT_SIGNAL_STRENGTH)
    rather than the strongest one when the LLM omits or misclassifies it."""
    if isinstance(raw_value, str) and raw_value.strip().lower() in ALLOWED_SIGNAL_STRENGTHS:
        return raw_value.strip().lower()
    return DEFAULT_SIGNAL_STRENGTH


@with_retry()
def _call_llm(north_star_metric: str, candidates: list[dict]) -> str:
    llm = get_llm(AGENT_NAME, json_mode=True)
    content = f"Target metric (north star):\n{north_star_metric or '(not stated)'}\n\nCandidates:\n{candidates}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _resolve_evidence(finding_ids: list[str], db_path=None) -> list[dict]:
    from tools.db import get_evidence_by_id  # local import: same cycle-avoidance pattern used elsewhere

    resolved = []
    for eid in finding_ids or []:
        row = get_evidence_by_id(eid, db_path=db_path)
        if row is None:
            continue
        claim = row.get("claim")
        if claim is None:
            unit = row.get("unit") or ""
            claim = f"{row.get('metric', 'metric')}: {row.get('value')} {unit}".strip()
        resolved.append({"claim": claim, "source_url": row.get("source_url")})
    return resolved


def _build_candidates(solutions: list[dict], features: list[dict]) -> tuple[list[dict], dict, list[str]]:
    """Only solutions with a linked feature are this node's concern —
    exploratory/hypothesis-linked features have no solution to kill.
    is_instrumentation solutions are excluded from the LLM candidate list
    (see module docstring) and returned separately for auto-promotion.
    Returns (LLM candidates, features_by_solution_id, auto-promoted solution ids)."""
    features_by_solution = {f.get("addresses_solution"): f for f in features if f.get("addresses_solution")}
    candidates = []
    auto_promoted_ids = []
    for solution in solutions:
        feature = features_by_solution.get(solution.get("id"))
        if feature is None:
            continue
        if solution.get("is_instrumentation"):
            auto_promoted_ids.append(solution.get("id"))
            continue
        candidates.append({
            "solution_id": solution.get("id"),
            "name": solution.get("name"),
            "problem_addressed": solution.get("problem_addressed"),
            "evidence_rationale": solution.get("evidence_rationale", ""),
            "evidence": _resolve_evidence(solution.get("finding_ids", [])),
            "feature": {
                "name": feature.get("name"),
                "description": feature.get("description"),
                "reach": feature.get("reach"),
                "impact": feature.get("impact"),
                "confidence": feature.get("confidence"),
                "effort": feature.get("effort"),
                "rice_score": feature.get("rice_score"),
            },
        })
    return candidates, features_by_solution, auto_promoted_ids


def solution_review_node(state: dict) -> dict:
    """Review every solution with a linked PRD feature against three
    credibility tests; kill authority removes a solution from the PRD
    before the reader ever sees it. is_instrumentation solutions are
    auto-promoted without ever reaching the LLM (see module docstring).

    Reads: state['solutions'], state['prd'], state['dropped_solutions']
    Writes: state['solutions'] (kills removed; survivors gain
    signal_strength: direct/adjacent/inferred), state['dropped_solutions']
    (kills appended, drop_reason="solution review: <reason>"), state['prd']
    (killed features removed from prd['features']; survivors gain the same
    signal_strength; moscow and header both recomputed against the
    surviving features, not inherited from pm_node's pre-kill pass),
    state['solutions_reviewed']=True, state['messages'], state['run_path']
    """
    solutions = state.get("solutions", [])
    prd = state.get("prd") or {}
    features = prd.get("features", [])
    north_star = prd.get("north_star_metric", "")
    dropped = list(state.get("dropped_solutions", []))

    candidates, features_by_solution, auto_promoted_ids = _build_candidates(solutions, features)

    # Auto-promoted (is_instrumentation) solutions never touch the LLM --
    # a structural guarantee, not a prompt instruction it could ignore.
    verdicts_by_id = {sid: {"verdict": "promote", "reason": _AUTO_PROMOTE_REASON} for sid in auto_promoted_ids}

    if not candidates and not auto_promoted_ids:
        run_logger.record(AGENT_NAME, candidate_count=0, killed_count=0)
        return {
            "solutions_reviewed": True,
            "messages": [AIMessage(content='{"verdicts": []}', name=AGENT_NAME)],
            "run_path": [AGENT_NAME],
        }

    if candidates:
        raw = _call_llm(north_star, candidates)
        parsed = safe_extract_json(raw)
        for v in parsed.get("verdicts", []):
            if v.get("verdict") in ALLOWED_VERDICTS:
                verdicts_by_id[v.get("solution_id")] = v
    else:
        raw = '{"verdicts": []}'

    kept_solutions = []
    killed_solution_ids: set[str] = set()
    for solution in solutions:
        verdict_entry = verdicts_by_id.get(solution.get("id"))
        if verdict_entry is None:
            kept_solutions.append(solution)
            continue
        if verdict_entry["verdict"] == "kill":
            solution = dict(solution)
            solution["drop_reason"] = f"solution review: {verdict_entry.get('reason', '')}"
            dropped.append(solution)
            killed_solution_ids.add(solution.get("id"))
            continue
        solution = dict(solution)
        solution["review_verdict"] = verdict_entry["verdict"]
        solution["review_reason"] = verdict_entry.get("reason", "")
        # Auto-promoted (is_instrumentation) solutions have no evidence by
        # construction -- signal_strength doesn't apply to them, left unset.
        if solution.get("id") not in auto_promoted_ids:
            solution["signal_strength"] = _normalize_signal_strength(verdict_entry.get("signal_strength"))
        kept_solutions.append(solution)

    # A kill changes the feature SET, and moscow/header are both computed
    # RELATIVE to that set (moscow's RICE floor/median off the top score
    # in play, header's counts off how many features there are) -- both
    # must be recomputed against the survivors, not left over from
    # pm_node's pre-kill pass. Observed failure: two real features were
    # mislabeled "Won't" against a floor derived from a feature that had
    # just been killed. Local import: same cycle-avoidance pattern used
    # elsewhere in this file.
    from agents.pm import _assign_moscow, _build_header

    signal_strength_by_solution_id = {
        s.get("id"): s["signal_strength"] for s in kept_solutions if s.get("signal_strength")
    }
    surviving_features = [f for f in features if f.get("addresses_solution") not in killed_solution_ids]
    for feature in surviving_features:
        strength = signal_strength_by_solution_id.get(feature.get("addresses_solution"))
        if strength:
            feature["signal_strength"] = strength
    _assign_moscow(surviving_features)

    updated_prd = dict(prd)
    updated_prd["features"] = surviving_features
    updated_prd["header"] = _build_header(surviving_features)
    if killed_solution_ids:
        updated_prd["header"] = (
            f"{updated_prd['header']} {len(killed_solution_ids)} candidate feature(s) were "
            f"killed on review (see 'Considered and dropped')."
        ).strip()

    run_logger.record(
        AGENT_NAME,
        candidate_count=len(candidates),
        auto_promoted_count=len(auto_promoted_ids),
        killed_count=len(killed_solution_ids),
        promoted_count=sum(1 for v in verdicts_by_id.values() if v["verdict"] == "promote"),
        kept_count=sum(1 for v in verdicts_by_id.values() if v["verdict"] == "keep"),
        direct_count=sum(1 for s in kept_solutions if s.get("signal_strength") == "direct"),
        adjacent_count=sum(1 for s in kept_solutions if s.get("signal_strength") == "adjacent"),
        inferred_count=sum(1 for s in kept_solutions if s.get("signal_strength") == "inferred"),
    )

    return {
        "solutions": kept_solutions,
        "dropped_solutions": dropped,
        "prd": updated_prd,
        "solutions_reviewed": True,
        "messages": [AIMessage(content=raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
