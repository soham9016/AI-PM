"""Funnel-decomposition agent — PM path, runs immediately after
structurer, before primary_research.

THE PM-MODE ANALOGUE OF agents/hypothesis.py: PM mode has no hypothesis/
verdict machinery (routing_mode exists specifically so it doesn't invent
diagnostic questions nobody asked — see agents/structurer.py). But
removing the CONSULTING diagnostic also removed the PM diagnostic, which
is a different thing: funnel decomposition. A PM handed "only 11% of
gym-booking users have ever used live classes" asks first WHERE in the
funnel the other 89% are dropping — awareness, trial, or repeat use are
different problems implying different builds. Without this step, PM mode
went straight to solutions and produced one scattered feature per funnel
stage, none of them targeted at anything (the Cult.fit run that exposed
this).

NOT A FIXED TEMPLATE: the LLM proposes 3-5 stages appropriate to the
metric actually being discussed — an adoption metric roughly follows
Aware -> Considered -> Tried -> Repeated, a retention metric roughly
follows Activated -> Habituated -> Retained, but neither is hardcoded;
these are examples in the prompt, not an enum.

PUBLIC VS. INTERNAL EVIDENCE: each stage states whether the evidence that
would confirm/rule it out is publicly findable or requires the company's
own instrumentation. agents/primary_research.py targets the public ones;
when nothing distinguishes between stages, the honest output is
"undetermined" plus a recommendation to instrument before building —
that is a legitimate PM output, not a failure (see
agents/solution_framing.py and tools/brief.py, which render it
prominently rather than treating it as an empty result).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import invoke_llm
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry

AGENT_NAME = "funnel_decomposition"
MIN_STAGES = 3
MAX_STAGES = 5
ALLOWED_LOCATABILITY = {"public", "internal"}

SYSTEM_PROMPT = """You are decomposing a business metric into the funnel
stages a user must pass through, so a PM can figure out WHERE a drop-off
is happening before jumping to solutions.

Given the problem statement and issue tree, identify the metric being
discussed (e.g. "% of gym-booking users who have ever used live
classes") and propose 3 to 5 ordered funnel stages appropriate to THIS
metric — do not force a fixed template. An adoption metric roughly
follows Aware -> Considered -> Tried -> Repeated; a retention metric
roughly follows Activated -> Habituated -> Retained; these are examples
of the SHAPE, not a checklist — use whatever stages actually fit the
metric described.

For EACH stage, state:
  - name: short stage name
  - definition: what this stage means for this specific product/metric
  - drop_signal: what would have to be true for the drop-off to be
    happening AT this stage specifically, not an earlier or later one
  - evidence_needed: what specific evidence would confirm or rule out
    the drop being at this stage
  - evidence_locatability: "public" if PROXY evidence for this stage
    could plausibly be found via public research, "internal" ONLY if you
    genuinely cannot think of any such proxy and it truly requires the
    company's own product instrumentation/analytics.

    Mark "internal" too readily is a common mistake — do not default to
    it just because the exact conversion percentage isn't public. Most
    late-funnel stages (checkout, purchase completion, returns,
    cancellation, post-purchase satisfaction) have real public PROXY
    signal even though the precise rate doesn't: app store reviews,
    support-forum threads, and published UX teardowns routinely describe
    exactly where users get stuck, in their own words. That proxy is
    "public" evidence for the stage, even without an exact number —
    evidence_needed should describe what to look for in it (e.g. "app
    store reviews mentioning problems with returns/exchanges at
    checkout"), not demand a metric nobody publishes.
    Worked example: a checkout-abandonment funnel's "Checkout friction"
    stage is "public", not "internal" — app store reviews complaining
    about a confusing or broken checkout/returns flow are real, findable
    evidence of friction at that stage, even though the exact
    drop-off percentage is internal.

Respond with ONLY a JSON object of this exact shape:
{
  "metric": "<the metric being decomposed>",
  "stages": [
    {
      "name": "<stage name>",
      "definition": "<what this stage means here>",
      "drop_signal": "<what would have to be true for the drop to be at this stage>",
      "evidence_needed": "<what evidence would confirm or rule this out>",
      "evidence_locatability": "public|internal"
    }
  ]
}
"""


@with_retry()
def _call_llm(problem_statement: str, issue_tree: dict, revision_note: str | None) -> str:
    content = f"Problem statement:\n{problem_statement}\n\nIssue tree:\n{issue_tree}"
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior funnel — address it:\n{revision_note}"
    response = invoke_llm(AGENT_NAME, [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _validate_stages(raw_stages) -> list[dict]:
    """Keep only well-formed stages (non-empty name + definition), cap
    at MAX_STAGES. Doesn't enforce MIN_STAGES here -- that's
    _ensure_funnel's job, since "too few valid stages" and "zero stages"
    both mean the same thing to the caller: decomposition failed."""
    if not isinstance(raw_stages, list):
        return []
    valid = []
    for s in raw_stages:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").strip()
        definition = (s.get("definition") or "").strip()
        if not name or not definition:
            continue
        locatability = s.get("evidence_locatability")
        if locatability not in ALLOWED_LOCATABILITY:
            locatability = "internal"  # the more conservative default -- don't assume public-findable
        valid.append({
            "name": name,
            "definition": definition,
            "drop_signal": (s.get("drop_signal") or "").strip(),
            "evidence_needed": (s.get("evidence_needed") or "").strip(),
            "evidence_locatability": locatability,
        })
        if len(valid) >= MAX_STAGES:
            break
    return valid


def _ensure_funnel(parsed: dict, problem_statement: str) -> dict:
    """Same "ran but empty" guarantee used throughout this codebase --
    `not state['funnel']` must reliably mean "hasn't run", not "ran and
    couldn't produce a usable decomposition". Fewer than MIN_STAGES
    valid stages is treated as decomposition failure, same as zero."""
    stages = _validate_stages(parsed.get("stages"))
    if len(stages) >= MIN_STAGES:
        return {"metric": parsed.get("metric") or problem_statement, "stages": stages}
    return {
        "metric": parsed.get("metric") or problem_statement,
        "stages": [
            {
                "name": "undetermined",
                "definition": "Funnel decomposition did not produce a usable stage breakdown.",
                "drop_signal": "",
                "evidence_needed": "",
                "evidence_locatability": "internal",
            }
        ],
        "unresolved": True,
    }


def funnel_decomposition_node(state: dict) -> dict:
    """Decompose the problem's stated metric into funnel stages before
    any solution is proposed — the PM-mode analogue of hypothesis
    generation.

    Reads: state['problem_statement'], state['issue_tree'], state['revision_notes']['funnel_decomposition']
    Writes: state['funnel'] ({metric, stages}), state['funnel_verdict']
    (reset to "" here — judged later by agents/primary_research.py once
    evidence exists), state['messages'], state['run_path']
    """
    problem_statement = state.get("problem_statement", "")
    issue_tree = state.get("issue_tree", {})
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    raw = _call_llm(problem_statement, issue_tree, revision_note)
    parsed = safe_extract_json(raw)
    funnel = _ensure_funnel(parsed, problem_statement)

    run_logger.record(
        AGENT_NAME,
        metric=funnel.get("metric"),
        stage_count=len(funnel["stages"]),
        stage_names=[s["name"] for s in funnel["stages"]],
        unresolved=funnel.get("unresolved", False),
    )

    return {
        "funnel": funnel,
        "funnel_verdict": "",
        "messages": [AIMessage(content=raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
