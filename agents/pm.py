"""PM agent — produces a PRD-lite with RICE-scored features.

Runs when state['problem_type'] == "product" (diagnostic/combined mode)
OR whenever routing_mode is "pm"/"combined" (gated by conditional edges
in routing.py/agents/engagement_manager.py).

EVIDENCE AWARENESS: a PRD that recommends four features against four
hypotheses the synthesizer marked "inconclusive" reads as a confident
spec built on a diagnosis the system itself couldn't confirm — the same
failure PM mode risks in a sneakier form, since PM mode has no verdicts
to come back inconclusive at all. The PM still gets to recommend —
refusing to isn't the fix — but every feature is labeled by what
actually backs it, set here in CODE, never LLM-self-assessed:
  evidence_backed    — addresses a 'supported' hypothesis, OR a solution
                        with real (code-validated) finding_ids
  regulation_backed  — hypothesis is evidence_type REGULATORY, OR
                        solution is marked required (regulation-mandated)
  hypothesis_driven  — addresses a hypothesis that isn't 'supported'
  exploratory         — addresses a solution with zero valid finding_ids,
                        or no hypothesis/solution link at all
A feature addressing a solution with zero supporting finding_ids is
"exploratory" — same as a hypothesis-mode feature with no hypothesis
link. PM mode does not get a free pass on evidence; see
agents/solution_framing.py for where hallucinated finding_ids get
stripped before they ever reach here.

The PRD also carries a one-line `header` (unconfirmed-feature count, and
a warning if no feature declares a guardrail_metric), a `north_star_metric`
for the whole PRD, and every feature states `success_metric`,
`guardrail_metric` (a metric that must not degrade), and `reach_basis`
(a fact_id, or literally "assumption") so an invented reach number can't
pass as a sourced estimate.

COMPETITIVE IMPACT CAPPING: when a feature addresses a solution,
agents/competitive_audit.py's per-branch classification caps `impact` in
code (never LLM-trusted) — building to table_stakes parity isn't
"massive impact" by definition, no matter how the model scores it.

MOSCOW, CODE-COMPUTED: every feature gets a `moscow` label
(Must/Should/Could/Won't) via _assign_moscow, derived purely from
already-established fields (evidence_strength, rice_score) — never asked
of the LLM. Must = regulation_backed (always ships regardless of RICE).
Won't = rice_score below 25% of this run's top score, unless already
Must. Should = evidence_backed and at/above the median rice_score among
this run's features. Could = everything else. The floor/median are
relative to THIS run's features, not a fixed number, since reach
magnitudes vary wildly across problems.

VALIDATION PLAN: every feature gets a `validation_plan` — cohort,
duration, and success_threshold are genuinely new judgment calls (no
existing data to derive them from) so they're LLM-authored; but
`hypothesis_tested` is filled in from CODE (the linked solution's
problem_addressed or hypothesis's statement, already resolved elsewhere
in this file) rather than asked of the LLM, since restating already-known
text risks drifting from the real wording.

INSTRUMENTATION FEATURE IS CODE-GUARANTEED: agents/solution_framing.py
guarantees an is_instrumentation solution exists whenever
funnel_verdict=="undetermined". That guarantee is worthless if the PRD
never turns it into a feature — observed failure: the solution existed
in an earlier version of this fix but the LLM never wrote a feature for
it, so it never reached the brief at all.
_ensure_instrumentation_feature closes that gap the same way: if no
feature addresses the instrumentation solution, one is synthesized
directly from it (small effort, no invented reach) and inserted first,
before the LLM's own features are processed — so it goes through the
same rice_score/evidence_strength/moscow pipeline as everything else,
labeled "instrumentation" (not "exploratory": a deliberate measurement
step isn't a shot in the dark) and forced to moscow="Must" (measuring
where the problem is has to come before betting on a fix for it).
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_llm
from tools.rag_retriever import retrieve_framework_notes
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry
from utils.text_similarity import longest_run_ratio, tokenize

AGENT_NAME = "pm"

logger = logging.getLogger("business_copilot.pm")

# Won't if rice_score falls below this fraction of the run's top score
# (unless the feature is already Must) -- relative to this run's own
# features, not a fixed absolute number that wouldn't generalize across
# problems with very different reach magnitudes.
RICE_FLOOR_RATIO = 0.25

# table_stakes: matching competitors isn't a differentiator, cap impact
# at "medium". parity_gap: partial competitive coverage, cap at "high".
# differentiator (or unclassified): no cap.
IMPACT_CAP_BY_CLASSIFICATION = {"table_stakes": 1, "parity_gap": 2}

SYSTEM_PROMPT = """You are a product manager writing a lightweight PRD.

Given the problem statement, issue tree, and EITHER a synthesis
(governing thought + hypothesis verdicts, diagnostic mode) OR a set of
candidate solutions with a competitive audit (PM mode) — possibly both —
propose 3-6 candidate features/initiatives and RICE-score each:

RICE score = (Reach x Impact x Confidence) / Effort
- Reach: estimated users/customers touched per quarter (a number)
- Impact: 3=massive, 2=high, 1=medium, 0.5=low, 0.25=minimal
- Confidence: 1.0=high, 0.8=medium, 0.5=low
- Effort: estimated person-months (a number > 0)

Each feature addresses EITHER a hypothesis (if you were given verdicts —
set "addresses_hypothesis" to its id) OR a solution (if you were given
solution candidates — set "addresses_solution" to its id). Leave the one
you didn't use as null. If a feature doesn't trace to either, leave both
null — it's exploratory, and that's fine to propose, just be honest
about it.

reach_basis is REQUIRED for every feature: either the fact_id
("fact:<id>") the reach number was derived from, or literally the string
"assumption" if you invented the number. Do not let an invented number
look like a sourced estimate.

Every feature needs success_metric (the metric that shows it worked, and
it should plausibly DRIVE the PRD's north_star_metric, not be an
unrelated number) and guardrail_metric (a metric that must NOT degrade
because of this feature — e.g. checkout conversion while adding a
consent flow; empty string only if genuinely nothing could plausibly be
put at risk). The PRD as a whole needs ONE north_star_metric it's
ultimately trying to move.

Every feature also needs a validation_plan — this is how you'd actually
de-risk the bet before committing fully:
  - cohort: who and what percentage — e.g. "10% of Tier-2 users"
  - duration: how long to run it before deciding — e.g. "4 weeks"
  - success_threshold: the specific, checkable threshold that would
    count as validated — not a vague "if it goes well"
If reach_basis is "assumption" (you invented the reach number), the
validation plan's cohort/success_threshold MUST describe how that
assumed number would actually get measured — this is the honest answer
to an unsourced reach estimate, not a separate concern. Do NOT include a
hypothesis_tested field in validation_plan — that's filled in separately
from what you already told us in addresses_hypothesis/addresses_solution.

Do NOT self-assess or state how well-evidenced a feature is, and do not
include any field for that — evidence strength is computed separately,
from the actual hypothesis verdicts or solution evidence, not from your
own judgment of your own recommendation. Similarly, do not self-assess
competitive impact — that's applied separately from the competitive
audit if one was given to you.

Do NOT compute or include rice_score — it is derived automatically from
reach, impact, confidence, and effort. Do not include this field at all;
anything you put there is discarded and recomputed from the four inputs.

Respond with ONLY a JSON object of this exact shape:
{
  "problem_summary": "<1-2 sentences>",
  "north_star_metric": "<the one metric this whole PRD should move>",
  "features": [
    {
      "name": "<feature name>",
      "description": "<1-2 sentences>",
      "addresses_hypothesis": "<hypothesis id, or null>",
      "addresses_solution": "<solution id, or null>",
      "reach": 1000,
      "reach_basis": "fact:12 (or the literal string \\"assumption\\")",
      "impact": 2,
      "confidence": 0.8,
      "effort": 2,
      "success_metric": "<the metric that shows this feature worked>",
      "guardrail_metric": "<a metric that must not degrade, or empty string>",
      "validation_plan": {
        "cohort": "<who and what percentage>",
        "duration": "<how long to run it before deciding>",
        "success_threshold": "<the specific threshold that counts as validated>"
      }
    }
  ]
}
"""


@with_retry()
def _call_llm(
    problem_statement, issue_tree, synthesis, solutions, competitive_audit, framework_notes,
    revision_note: str | None,
) -> str:
    llm = get_llm(AGENT_NAME)
    content = (
        f"Problem statement:\n{problem_statement}\n\n"
        f"Issue tree:\n{issue_tree}\n\n"
        f"Synthesis (diagnostic mode; may be empty):\n{synthesis}\n\n"
        f"Candidate solutions (PM mode; may be empty):\n{solutions}\n\n"
        f"Competitive audit (PM mode; may be empty):\n{competitive_audit}\n\n"
        f"Framework guidance:\n{framework_notes}"
    )
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior PRD — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _feature_evidence_label(feature: dict, hypotheses_by_id: dict, verdicts_by_id: dict, solutions_by_id: dict) -> str:
    """Code-determined, not LLM-self-assessed. See module docstring for
    the four labels. Checks addresses_solution first (PM mode), then
    addresses_hypothesis (diagnostic mode) — a feature only ever sets one
    of the two, per the prompt."""
    solution_id = feature.get("addresses_solution")
    if solution_id:
        solution = solutions_by_id.get(solution_id)
        if not solution:
            return "exploratory"
        if solution.get("is_instrumentation"):
            return "instrumentation"
        if solution.get("required"):
            return "regulation_backed"
        if solution.get("finding_ids"):
            return "evidence_backed"
        return "exploratory"

    hypothesis_id = feature.get("addresses_hypothesis")
    hypothesis = hypotheses_by_id.get(hypothesis_id) if hypothesis_id else None
    if not hypothesis:
        return "exploratory"
    if (hypothesis.get("evidence_type") or "").upper() == "REGULATORY":
        return "regulation_backed"
    verdict = verdicts_by_id.get(hypothesis_id, {}).get("verdict")
    if verdict == "supported":
        return "evidence_backed"
    return "hypothesis_driven"


def _compute_rice_score(feature: dict) -> float | None:
    """The single formula for RICE in this codebase — reach x impact x
    confidence / effort. Called unconditionally, never trusting whatever
    the LLM may have put in a rice_score field (the prompt asks it not
    to supply one at all, but PM's call isn't JSON-mode-constrained, so
    this is the actual enforcement, not the prompt instruction)."""
    try:
        reach = float(feature["reach"])
        impact = float(feature["impact"])
        confidence = float(feature["confidence"])
        effort = float(feature["effort"])
    except (TypeError, ValueError, KeyError):
        return None
    if effort <= 0:
        return None
    return round((reach * impact * confidence) / effort, 2)


def _classification_by_branch(competitive_audit: dict) -> dict:
    return {
        area.get("branch"): area.get("classification")
        for area in (competitive_audit or {}).get("areas", [])
        if area.get("branch")
    }


def _apply_competitive_cap(feature: dict, solutions_by_id: dict, classification_by_branch: dict) -> None:
    """Cap `impact` in code when the addressed solution's branch was
    classified table_stakes/parity_gap — never LLM-trusted. Mutates
    `feature` in place."""
    solution_id = feature.get("addresses_solution")
    if not solution_id:
        return
    solution = solutions_by_id.get(solution_id)
    if not solution:
        return
    classification = classification_by_branch.get(solution.get("branch"))
    cap = IMPACT_CAP_BY_CLASSIFICATION.get(classification)
    impact = feature.get("impact")
    if cap is None or not isinstance(impact, (int, float)) or impact <= cap:
        return

    original = impact
    feature["impact"] = cap
    feature["impact_capped_reason"] = (
        f"competitive_audit classified branch {solution.get('branch')!r} as {classification!r} "
        f"— impact capped from {original} to {cap}"
    )
    feature["rice_score"] = _compute_rice_score(feature)


def _has_guardrail_coverage(features: list[dict]) -> bool:
    return any((f.get("guardrail_metric") or "").strip() for f in features)


def _hypothesis_tested_text(feature: dict, hypotheses_by_id: dict, solutions_by_id: dict) -> str:
    """Code-derived, not LLM-restated (see module docstring) — pulled
    directly from the already-resolved link, so it can't drift from the
    real wording the way asking the model to restate it could."""
    solution_id = feature.get("addresses_solution")
    if solution_id and solution_id in solutions_by_id:
        return solutions_by_id[solution_id].get("problem_addressed", "")
    hypothesis_id = feature.get("addresses_hypothesis")
    if hypothesis_id and hypothesis_id in hypotheses_by_id:
        return hypotheses_by_id[hypothesis_id].get("statement", "")
    return "No specific hypothesis/solution link — exploratory bet."


def _copy_solution_fields(feature: dict, solutions_by_id: dict) -> None:
    """Copy works_within_constraint/differentiation/addresses_funnel_stage/
    is_instrumentation/instrumentation_plan from the addressed solution
    onto the feature, so tools/brief.py can render them without
    cross-referencing solutions_by_id itself."""
    solution_id = feature.get("addresses_solution")
    solution = solutions_by_id.get(solution_id) if solution_id else None
    feature["works_within_constraint"] = (solution or {}).get("works_within_constraint", "")
    feature["differentiation"] = (solution or {}).get("differentiation", "")
    feature["addresses_funnel_stage"] = (solution or {}).get("addresses_funnel_stage", "")
    feature["is_instrumentation"] = (solution or {}).get("is_instrumentation", False)
    feature["instrumentation_plan"] = (solution or {}).get("instrumentation_plan", {})


def _ensure_instrumentation_feature(features: list[dict], solutions_by_id: dict) -> list[dict]:
    """Code-guaranteed, not just hoped for: agents/solution_framing.py
    guarantees an is_instrumentation SOLUTION exists whenever
    funnel_verdict=="undetermined", but that's worthless if the PRD never
    turns it into a FEATURE. If the LLM didn't address it, synthesize one
    directly from the solution (no invented reach, minimal effort — this
    is measurement work, not a product bet) and insert it first."""
    instrumentation_solutions = [s for s in solutions_by_id.values() if s.get("is_instrumentation")]
    if not instrumentation_solutions:
        return features
    addressed_solution_ids = {f.get("addresses_solution") for f in features}
    for solution in instrumentation_solutions:
        if solution.get("id") in addressed_solution_ids:
            continue
        plan = solution.get("instrumentation_plan") or {}
        features.insert(0, {
            "name": solution.get("name") or "Instrument the funnel",
            "description": plan.get("measure") or solution.get("problem_addressed", ""),
            "addresses_hypothesis": None,
            "addresses_solution": solution.get("id"),
            "reach": 0,
            "reach_basis": "assumption",
            "impact": 0.25,
            "confidence": 1.0,
            "effort": 0.5,
            "success_metric": plan.get("identifying_result") or "Funnel drop-off stage identified",
            "guardrail_metric": "not applicable",
        })
        run_logger.record(AGENT_NAME, instrumentation_feature_synthesized=solution.get("id"))
    return features


def _maybe_unrelated_to_north_star(feature: dict, north_star_metric: str) -> None:
    """Soft visibility only — whether a metric genuinely 'drives' the
    north star is a business judgment no keyword check can settle. Logs
    when there's zero token overlap at all, which is at least worth a
    second look."""
    success_metric = feature.get("success_metric") or ""
    if not north_star_metric or not success_metric:
        return
    _, run_len = longest_run_ratio(tokenize(success_metric), tokenize(north_star_metric))
    if run_len == 0:
        logger.warning(
            "success_metric_may_not_drive_north_star: feature %r success_metric %r shares no "
            "words with north_star_metric %r",
            feature.get("name"), success_metric, north_star_metric,
        )


def _assign_moscow(features: list[dict]) -> None:
    """Must/Should/Could/Won't, entirely code-computed from fields already
    established elsewhere in this file. See module docstring for the
    exact precedence (Must overrides Won't) and why the floor/median are
    relative to this run's own features rather than a fixed constant.

    IDEMPOTENT BY DESIGN: agents/solution_review.py calls this a second
    time on the same feature dicts after removing killed features, so a
    feature's moscow can legitimately change between calls (e.g. Won't ->
    Should once a higher-scoring feature that inflated the floor is gone).
    moscow_reason is only meaningful for Won't -- explicitly cleared on
    every other branch so a stale reason from an earlier call can't survive
    a re-classification (observed failure: exactly this, immediately after
    fixing the floor recomputation itself)."""
    scored = sorted(f["rice_score"] for f in features if f.get("rice_score") is not None)
    max_score = scored[-1] if scored else None
    floor = max_score * RICE_FLOOR_RATIO if max_score is not None else None
    if scored:
        mid = len(scored) // 2
        median_score = (scored[mid - 1] + scored[mid]) / 2 if len(scored) % 2 == 0 else scored[mid]
    else:
        median_score = None

    for feature in features:
        rice_score = feature.get("rice_score")
        evidence_strength = feature.get("evidence_strength")
        feature.pop("moscow_reason", None)

        if evidence_strength in ("regulation_backed", "instrumentation"):
            # Measuring where the problem is has to come before betting
            # on a fix for it -- RICE math (often tiny for pure
            # measurement work) doesn't get a vote here, same as
            # regulation-mandated work doesn't.
            feature["moscow"] = "Must"
            continue

        if floor is not None and rice_score is not None and rice_score < floor:
            feature["moscow"] = "Won't"
            feature["moscow_reason"] = (
                f"RICE score {rice_score} is below the floor of {floor:.2f} "
                f"(25% of this run's top score {max_score})"
            )
            continue

        if evidence_strength == "evidence_backed" and median_score is not None and rice_score is not None and rice_score >= median_score:
            feature["moscow"] = "Should"
            continue

        feature["moscow"] = "Could"


def _build_header(features: list[dict]) -> str:
    """The unconfirmed-count sentence + guardrail-coverage warning — a
    pure function of whatever feature list it's given. Deliberately
    reusable: agents/solution_review.py calls this again after removing
    killed features, so the header describes what actually ships, not
    what pm_node originally proposed before the kill review ran."""
    if not features:
        return "No features were proposed."
    unconfirmed_count = sum(1 for f in features if f.get("evidence_strength") in ("hypothesis_driven", "exploratory"))
    parts = [
        f"{unconfirmed_count} of {len(features)} recommended feature(s) rest on "
        f"hypotheses/problems that have not been confirmed by evidence."
    ]
    if not _has_guardrail_coverage(features):
        parts.append("WARNING: no feature declares a guardrail_metric.")
    return " ".join(parts)


def pm_node(state: dict) -> dict:
    """Produce a PRD-lite with RICE-scored features, evidence-aware in
    both diagnostic mode (hypotheses/verdicts) and PM mode (solutions/
    competitive audit).

    Reads: state['problem_statement'], state['issue_tree'], state['hypotheses'],
    state['synthesis'], state['solutions'], state['competitive_audit'],
    state['revision_notes']['pm']
    Writes: state['prd'] (each feature gains code-set 'evidence_strength',
    'rice_score', 'moscow' (+'moscow_reason' for Won't), 'validation_plan'
    (with code-derived 'hypothesis_tested'), 'works_within_constraint',
    'differentiation', and, when applicable, a competitive impact cap;
    PRD gains 'header' and 'north_star_metric'), state['messages'], state['run_path']
    """
    notes = retrieve_framework_notes("RICE scoring reach impact confidence effort north star guardrail metrics")
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    hypotheses = state.get("hypotheses", [])
    synthesis = state.get("synthesis", {})
    solutions = state.get("solutions", [])
    competitive_audit = state.get("competitive_audit", {})

    raw = _call_llm(
        state["problem_statement"], state.get("issue_tree", {}), synthesis, solutions,
        competitive_audit, notes, revision_note,
    )
    parsed = safe_extract_json(raw)

    hypotheses_by_id = {h.get("id"): h for h in hypotheses if h.get("id")}
    verdicts_by_id = {v.get("hypothesis_id"): v for v in synthesis.get("verdicts", []) if v.get("hypothesis_id")}
    solutions_by_id = {s.get("id"): s for s in solutions if s.get("id")}
    classification_by_branch = _classification_by_branch(competitive_audit)
    north_star_metric = parsed.get("north_star_metric") or ""

    features = _ensure_instrumentation_feature(parsed.get("features", []), solutions_by_id)
    unconfirmed_count = 0
    for feature in features:
        # Always code-computed, never read from the LLM (see module
        # docstring/_compute_rice_score) -- set first so every later step
        # (evidence label doesn't depend on it, but the competitive cap
        # below recomputes from this same baseline) works from a real number.
        feature["rice_score"] = _compute_rice_score(feature)
        if feature["rice_score"] is None:
            logger.warning("Could not compute rice_score for feature %r — non-numeric or missing reach/impact/confidence/effort", feature.get("name"))

        label = _feature_evidence_label(feature, hypotheses_by_id, verdicts_by_id, solutions_by_id)
        feature["evidence_strength"] = label
        if label == "instrumentation":
            # Measurement work has no guardrail metric by definition --
            # nothing is shipping that could degrade something else.
            # Forced here, not left to the LLM/renderer: observed failure
            # was the LLM writing the literal string "None" into this
            # field, which `guardrail_metric or "(none)"` doesn't catch
            # (a non-empty string is truthy).
            feature["guardrail_metric"] = "not applicable"
        if not feature.get("reach_basis"):
            feature["reach_basis"] = "assumption"
        _apply_competitive_cap(feature, solutions_by_id, classification_by_branch)
        _copy_solution_fields(feature, solutions_by_id)

        validation_plan = feature.get("validation_plan")
        if not isinstance(validation_plan, dict):
            validation_plan = {}
        validation_plan["hypothesis_tested"] = _hypothesis_tested_text(feature, hypotheses_by_id, solutions_by_id)
        feature["validation_plan"] = validation_plan

        _maybe_unrelated_to_north_star(feature, north_star_metric)

        if label in ("hypothesis_driven", "exploratory"):
            unconfirmed_count += 1

    _assign_moscow(features)

    parsed["features"] = features
    parsed["header"] = _build_header(features)
    parsed.setdefault("north_star_metric", "")

    run_logger.record(
        AGENT_NAME,
        feature_count=len(features),
        unconfirmed_count=unconfirmed_count,
        evidence_backed=sum(1 for f in features if f["evidence_strength"] == "evidence_backed"),
        regulation_backed=sum(1 for f in features if f["evidence_strength"] == "regulation_backed"),
        instrumentation_count=sum(1 for f in features if f["evidence_strength"] == "instrumentation"),
        competitive_capped=sum(1 for f in features if "impact_capped_reason" in f),
        guardrail_covered=_has_guardrail_coverage(features),
        must_count=sum(1 for f in features if f.get("moscow") == "Must"),
        should_count=sum(1 for f in features if f.get("moscow") == "Should"),
        could_count=sum(1 for f in features if f.get("moscow") == "Could"),
        wont_count=sum(1 for f in features if f.get("moscow") == "Won't"),
    )

    return {
        "prd": parsed,
        "messages": [AIMessage(content=raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
