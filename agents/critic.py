"""Critic agent — checks are mode-aware (see agents/structurer.py for
routing_mode). Diagnostic/combined modes get the original four:

  1. traceability      — every cited evidence_id is verified against the
                          evidence-base DB (not just internally consistent
                          within the synthesis object).
  2. verdict-evidence   — a verdict of "supported"/"killed" backed by zero
     consistency           evidence_ids is a contradiction.
  3. overreach          — a hypothesis ruled "inconclusive" or
                          "insufficient_data" that isn't listed in the
                          write-up's `unknowns` is being treated as settled.
  4. MECE gaps          — an issue-tree branch with no hypothesis testing
                          it, and not present in the synthesizer's own
                          `unexamined_branches` disclosure.

PM/combined modes ALSO get two more — MECE/verdict checks don't apply to
solutions (there's no verdict to be inconsistent, and issue-tree coverage
is competitive_audit's/solution_framing's concern, not a hypothesis gap):

  5. solution traceability — every solution's finding_ids must exist in
                          the evidence-base DB, and a solution with ZERO
                          finding_ids is flagged exploratory (PM mode
                          doesn't get to skip evidence just because there
                          are no verdicts left to come back inconclusive
                          — see agents/solution_framing.py).
  6. regulation ranking — a regulation-mandated (required) feature RICE-
                          ranked below an optional one is backwards:
                          compliance isn't optional just because the RICE
                          math came out lower.

All checks are deterministic (computed in code against the DB/state, not
guessed by an LLM) so they can't be talked out of existing. The model's
job is materiality: given the full candidate list, decide which would
actually change the recommendation or mislead a reader — a critic that
surfaces every technically-true nitpick is as useless as one that
surfaces nothing. The model may drop candidates; it may not invent new
ones (any flag it returns that doesn't match a real candidate is
discarded before output).

FAIL CLOSED: the materiality filter is an optional refinement on top of
the deterministic candidates, which are the actual safety property. If
the LLM's filter response can't be parsed as JSON, that is NOT treated
as "the model said drop everything" — every candidate is kept instead.
A parse failure silently turning real flags into a pass is exactly
backwards for a guardrail.
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_llm
from tools.db import evidence_exists
from utils.logger import run_logger
from utils.parsing import extract_json
from utils.retry import with_retry

AGENT_NAME = "critic"

logger = logging.getLogger("business_copilot.critic")

MATERIALITY_SYSTEM_PROMPT = """You are a skeptical reviewer applying a
materiality filter to a list of automatically-detected candidate issues
in a business brief.

For each candidate, decide whether it is MATERIAL: would leaving it
unaddressed change the recommendation or mislead a reader? Drop
candidates that are technically true but inconsequential. Do NOT invent
issues beyond the candidates given to you — every flag you return must
correspond to one of the candidates (same type + target).

Respond with ONLY a JSON object of this exact shape:
{
  "flags": [
    {"type": "<candidate type>", "target": "<candidate target>", "issue": "<why this matters>", "suggested_action": "<which agent should fix it>"}
  ]
}
"""


@with_retry()
def _filter_by_materiality(candidates: list[dict], synthesis: dict) -> str:
    llm = get_llm(AGENT_NAME, json_mode=True)
    content = f"Candidate issues:\n{candidates}\n\nSynthesis under review:\n{synthesis}"
    response = llm.invoke([SystemMessage(content=MATERIALITY_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _check_traceability(synthesis: dict) -> list[dict]:
    """Every evidence_id cited by a verdict or a key argument must exist
    in the DB — a hallucinated id here means the claim isn't real evidence."""
    cited_ids: set[str] = set()
    for verdict in synthesis.get("verdicts", []):
        cited_ids.update(verdict.get("evidence_ids") or [])
    for arg in synthesis.get("key_arguments", []):
        cited_ids.update(arg.get("evidence_ids") or [])

    flags = []
    for evidence_id in sorted(cited_ids):
        if not evidence_exists(evidence_id):
            flags.append(
                {
                    "type": "traceability",
                    "target": evidence_id,
                    "issue": f"evidence_id {evidence_id!r} is cited but does not exist in the evidence-base DB.",
                    "suggested_action": "synthesizer",
                }
            )
    return flags


def _check_verdict_consistency(synthesis: dict) -> list[dict]:
    """A 'supported' or 'killed' verdict with zero backing evidence_ids
    is a contradiction (mirrors the synthesizer's own sanity check —
    checked again here independently, in case that check was bypassed)."""
    flags = []
    for verdict in synthesis.get("verdicts", []):
        if verdict.get("verdict") in ("supported", "killed") and not verdict.get("evidence_ids"):
            flags.append(
                {
                    "type": "verdict_evidence_consistency",
                    "target": verdict.get("hypothesis_id"),
                    "issue": f"verdict is {verdict.get('verdict')!r} but cites zero evidence_ids.",
                    "suggested_action": "synthesizer",
                }
            )
    return flags


def _check_overreach(synthesis: dict) -> list[dict]:
    """A hypothesis ruled inconclusive/insufficient_data must be listed
    in `unknowns`, or the write-up is implicitly treating it as settled."""
    unknowns = set(synthesis.get("unknowns") or [])
    flags = []
    for verdict in synthesis.get("verdicts", []):
        if verdict.get("verdict") in ("inconclusive", "insufficient_data"):
            hypothesis_id = verdict.get("hypothesis_id")
            if hypothesis_id not in unknowns:
                flags.append(
                    {
                        "type": "overreach",
                        "target": hypothesis_id,
                        "issue": (
                            f"hypothesis {hypothesis_id} is {verdict.get('verdict')!r} but is not listed "
                            f"in the write-up's `unknowns` — it may be presented as more settled than it is."
                        ),
                        "suggested_action": "synthesizer",
                    }
                )
    return flags


def _check_mece_gaps(issue_tree: dict, hypotheses: list[dict], synthesis: dict) -> list[dict]:
    """An issue-tree branch with no hypothesis testing it is a silent gap
    unless the synthesizer's `unexamined_branches` list (computed
    deterministically in synthesizer_node, not guessed from prose) says so.

    With a hard cap on hypothesis count, a tree with more branches than
    the cap makes full coverage structurally impossible — regenerating
    hypotheses can't fix that, only disclosing the gap can. So the fix
    for this flag is the synthesizer (write the disclosure), not the
    hypothesis agent (which would just hit the same cap again and
    produce the identical flag on the next pass)."""
    branch_names = {b.get("name") for b in issue_tree.get("branches", []) if b.get("name")}
    covered_branches = {h.get("branch") for h in hypotheses if h.get("branch")}
    uncovered = branch_names - covered_branches

    disclosed_branches = {b.lower() for b in (synthesis.get("unexamined_branches") or []) if isinstance(b, str)}

    flags = []
    for branch in sorted(uncovered):
        if branch.lower() not in disclosed_branches:
            flags.append(
                {
                    "type": "mece_gap",
                    "target": branch,
                    "issue": (
                        f"issue-tree branch {branch!r} has no hypothesis testing it and isn't listed in "
                        f"the synthesis's unexamined_branches — disclose it there rather than leaving it silent."
                    ),
                    "suggested_action": "synthesizer",
                }
            )
    return flags


def _check_solution_traceability(solutions: list[dict]) -> list[dict]:
    """Mirrors _check_traceability/_check_verdict_consistency but for PM
    mode's solutions: every finding_id must be real, and a solution with
    zero finding_ids is exploratory — flagged explicitly rather than
    silently treated as if it were evidence-backed. This is the check
    that stops PM mode from becoming a way to skip evidence."""
    flags = []
    for solution in solutions:
        solution_id = solution.get("id")
        finding_ids = solution.get("finding_ids") or []
        if not finding_ids:
            flags.append(
                {
                    "type": "solution_traceability",
                    "target": solution_id,
                    "issue": (
                        f"solution {solution_id!r} ({solution.get('name')!r}) has zero supporting "
                        f"finding_ids — it is exploratory, not evidence-backed."
                    ),
                    "suggested_action": "solution_framing",
                }
            )
            continue
        for fid in finding_ids:
            if not evidence_exists(fid):
                flags.append(
                    {
                        "type": "solution_traceability",
                        "target": f"{solution_id}:{fid}",
                        "issue": f"solution {solution_id!r} cites evidence_id {fid!r} which does not exist in the evidence-base DB.",
                        "suggested_action": "solution_framing",
                    }
                )
    return flags


def _check_regulation_ranking(prd: dict) -> list[dict]:
    """A regulation-mandated (required, i.e. evidence_strength
    'regulation_backed') feature ranked below an optional one by RICE
    score is backwards — compliance isn't optional just because its RICE
    math comes out lower than a nice-to-have's."""
    features = prd.get("features") or []
    if len(features) < 2:
        return []
    ranked = sorted(features, key=lambda f: f.get("rice_score") or 0, reverse=True)
    required_ranks = [i for i, f in enumerate(ranked) if f.get("evidence_strength") == "regulation_backed"]
    optional_ranks = [i for i, f in enumerate(ranked) if f.get("evidence_strength") != "regulation_backed"]
    if required_ranks and optional_ranks and max(required_ranks) > min(optional_ranks):
        return [
            {
                "type": "regulation_ranking",
                "target": "prd.features",
                "issue": (
                    "at least one regulation-mandated feature is ranked below an optional feature by "
                    "RICE score — compliance requirements shouldn't be deprioritized by a scoring formula."
                ),
                "suggested_action": "pm",
            }
        ]
    return []


def critic_node(state: dict) -> dict:
    """Run the mode-appropriate checks against the synthesis and/or
    solutions, then materiality-filter the candidates with the LLM.

    Reads: state['routing_mode'], state['synthesis'], state['hypotheses'],
    state['issue_tree'], state['solutions'], state['prd']
    Writes: state['critique'], state['critic_passed'] (True iff critique
    ends up empty), state['messages'], state['run_path']
    """
    routing_mode = state.get("routing_mode", "diagnostic")
    synthesis = state.get("synthesis", {})
    hypotheses = state.get("hypotheses", [])
    issue_tree = state.get("issue_tree", {})
    solutions = state.get("solutions", [])
    prd = state.get("prd") or {}

    candidates = []
    if routing_mode in ("diagnostic", "combined") and synthesis:
        candidates += (
            _check_traceability(synthesis)
            + _check_verdict_consistency(synthesis)
            + _check_overreach(synthesis)
            + _check_mece_gaps(issue_tree, hypotheses, synthesis)
        )
    if routing_mode in ("pm", "combined") and solutions:
        candidates += _check_solution_traceability(solutions) + _check_regulation_ranking(prd)

    if not candidates:
        run_logger.record(AGENT_NAME, candidate_count=0, flag_count=0)
        return {
            "critique": [],
            "critic_passed": True,
            "messages": [AIMessage(content='{"flags": [], "passed": true}', name=AGENT_NAME)],
            "run_path": [AGENT_NAME],
        }

    raw = _filter_by_materiality(candidates, synthesis)

    # Materiality filtering only decides INCLUSION (type, target) — the kept
    # flag's content always comes from the original candidate, never from
    # the LLM's restated copy, so a drifted suggested_action/issue can't
    # slip through and silently fall through to engagement_manager's
    # default loop-back target.
    candidates_by_key = {(c["type"], c["target"]): c for c in candidates}

    try:
        parsed = extract_json(raw)
    except json.JSONDecodeError:
        # FAIL CLOSED: an unparseable filter response is not a materiality
        # decision — keep every candidate rather than silently passing.
        logger.warning(
            "Materiality filter returned unparseable JSON; failing closed and keeping all %d candidate(s): %r",
            len(candidates), raw[:500],
        )
        final_flags = list(candidates_by_key.values())
    else:
        kept_keys = {
            (flag.get("type"), flag.get("target"))
            for flag in parsed.get("flags", [])
            if (flag.get("type"), flag.get("target")) in candidates_by_key
        }
        final_flags = [candidates_by_key[key] for key in candidates_by_key if key in kept_keys]

    run_logger.record(AGENT_NAME, candidate_count=len(candidates), flag_count=len(final_flags))

    return {
        "critique": final_flags,
        "critic_passed": not final_flags,
        "messages": [AIMessage(content=raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
