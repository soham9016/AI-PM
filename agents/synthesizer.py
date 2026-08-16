"""Synthesizer agent — two separate moves, not one call:

  (a) Verdict per hypothesis: pulls findings + facts by hypothesis_id
      straight from the evidence-base DB (the source of truth — not the
      ephemeral state['research_findings'] scratch summary), and rules
      supported / killed / inconclusive / insufficient_data. A code
      sanity check catches contradictions an LLM might produce (e.g.
      "supported" with zero supporting evidence_ids) before they reach
      the write-up.
  (b) Answer-first write-up built ONLY from the verdicts, never from raw
      evidence directly — the write-up cannot assert anything that
      doesn't trace back to an evidence_id already attached to a verdict.

With a hard cap on hypothesis count (config.MAX_HYPOTHESES), full
issue-tree coverage is structurally impossible once a tree has more
branches than the cap — that's not a bug to route back to hypothesis
generation (regenerating a capped set can't cover an uncapped tree
either). The correct response is disclosure, the same move a real
consultant makes under time pressure: say plainly which branches
weren't examined. `unexamined_branches` is computed here in code (a
deterministic set-difference against the issue tree, not left to the
write-up LLM's prose to remember) so the critic can check it exactly,
rather than guessing from free text.

Disclosure and the headline answer are kept in SEPARATE output fields
(`answer` vs `caveats`) — an earlier version told the write-up LLM to
fold the disclosure into `answer`, which produced recommendations like
"...but this analysis does not examine cost structure, market and
competition, operational efficiency, or pricing" — four gaps listed in
the one sentence meant to be the actionable conclusion, leaving nothing
a reader could act on. `answer` must now be actionable on its own;
`caveats` is where limitations go.

SOURCE QUALITY: provenance (a real source_url, DB constraints) is not
the same thing as source QUALITY — a run can have every claim traced to
a real URL and still have its "supported" verdict resting entirely on
one person's LinkedIn opinion post. `source_tier(url)` (utils/source_tier.py
— moved there once agents/primary_research.py needed the same
classification) classifies each piece of evidence by domain pattern
(filing > analyst > news > blog > social) computed on read from the
already-stored source_url — not a stored column, so no schema
change/migration and no backfill question for rows inserted before this
existed. It's fed into the verdict LLM's prompt so it can weight
accordingly, AND enforced by `_sanity_check` as a deterministic backstop:
a "supported"/"killed" verdict resting only on social-tier evidence is
downgraded to "inconclusive" regardless of what the LLM says, same
defense-in-depth pattern as the critic's deterministic candidates
backing its LLM materiality filter.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_llm
from tools.db import get_facts, get_findings
from tools.rag_retriever import retrieve_framework_notes
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry
from utils.source_tier import TIER_RANK, source_tier

AGENT_NAME = "synthesizer"

VERDICT_SYSTEM_PROMPT = """You are a synthesis lead ruling on one
hypothesis. You are given the findings (qualitative claims with a
stance) and facts (numeric, grouped) recorded against it in the
evidence base — each carries an evidence_id ("finding:<id>" or
"fact:<id>") and a source_tier: "filing" (company's own regulatory/IR
materials) > "analyst" (research firms) > "news" (established outlets)
> "blog" > "social" (LinkedIn/X/Reddit/etc — one person's opinion, not
verified reporting). Rule strictly from this evidence; do not use
outside knowledge.

Weight by source_tier: a single social-tier post is weak evidence on its
own — prefer "inconclusive" over "supported"/"killed" if social-tier
evidence is genuinely all you have, even if it clearly leans one way.
This will also be enforced in code as a backstop, but rule as if it
weren't — don't rely on the backstop to do your job.

The hypothesis also carries an "evidence_type" — weigh accordingly:
  REGULATORY  — only rule "supported" if evidence traces to the actual
                regulation/official regulator source, not commentary
                about it, even if that commentary is news-tier or higher.
                A LinkedIn post or news article summarizing "what a law
                requires" is not the same as the law's own text.
  FINANCIAL   — filings/investor-relations sources carry the most weight;
                third-party estimates of a company's own financials are
                weaker even at news-tier.
  USER        — needs actual research/survey data; marketing copy or a
                company's own claims about what users want don't settle it.
  PRODUCT / MARKET / COMPETITIVE — weigh normally by source_tier.

Verdict must be one of:
  "supported"          — findings/facts clearly back the hypothesis, contradicting evidence is weak/absent
  "killed"              — findings/facts clearly contradict the hypothesis (a kill condition was met)
  "inconclusive"        — evidence exists but is mixed, too weak, or too low-tier to rule either way
  "insufficient_data"   — little or no relevant evidence was gathered

Every evidence_id you cite MUST be one of the ids given to you — never
invent one.

Respond with ONLY a JSON object of this exact shape:
{
  "hypothesis_id": "<id>",
  "verdict": "supported|killed|inconclusive|insufficient_data",
  "reasoning": "<1-2 sentences citing the evidence>",
  "evidence_ids": ["finding:3", "fact:7"],
  "quantified": true
}
"""

WRITEUP_SYSTEM_PROMPT = """You are a synthesis lead writing an answer-
first business brief. You are given the per-hypothesis verdicts (not
raw evidence — every verdict already carries its own evidence_ids) and
a list of issue-tree branches that were never examined at all (no
hypothesis was even written for them, e.g. because of a cap on how many
hypotheses this run could pursue). Write:
  - answer: the recommendation/conclusion in ONE sentence, built ONLY
    from what WAS examined. Lead with the finding — a reader must be
    able to act on this sentence alone. Do NOT mention unexamined
    branches, gaps, or limitations here; that belongs in `caveats`, not
    the headline. A recommendation that spends its one sentence listing
    what it didn't look at isn't a recommendation.
  - key_arguments: 2-4 grouped supporting arguments, each listing the
    evidence_ids (pulled from the verdicts you were given) backing it
  - ruled_out: hypothesis_ids with verdict "killed"
  - unknowns: hypothesis_ids with verdict "inconclusive" or "insufficient_data"
  - caveats: a separate closing note (1-2 sentences) naming what this
    analysis did NOT examine — list the unexamined branches you were
    given, by name — plus any other material limitation. This is where
    disclosure belongs. Omit or leave empty only if there is truly
    nothing to disclose (no unexamined branches and no other gap).

Disclosure is expected, not a failure — a bounded piece of research that
says what it didn't cover is more honest than one that's silent about
it. But it goes in `caveats`, never folded into `answer`.

You may not assert anything that isn't backed by an evidence_id present
in the verdicts you were given. If the evidence doesn't support a
confident answer, say so in the answer rather than overstating it — that
is different from disclosing unexamined branches, and still belongs in
`answer` since it's about the evidence you DID look at.

Respond with ONLY a JSON object of this exact shape:
{
  "answer": "<one sentence, actionable on its own, no caveats folded in>",
  "key_arguments": [
    {"argument": "<supporting argument>", "evidence_ids": ["finding:3", "fact:7"]}
  ],
  "ruled_out": ["H2"],
  "unknowns": ["H3"],
  "caveats": "<what wasn't examined, named explicitly, and any other limitation>"
}
"""


@with_retry()
def _call_verdict_llm(hypothesis: dict, findings: list[dict], facts: list[dict]) -> str:
    llm = get_llm(AGENT_NAME)
    evidence = {
        "findings": [
            {
                "evidence_id": f"finding:{f['id']}",
                "claim": f["claim"],
                "stance": f["stance"],
                "source_tier": source_tier(f.get("source_url")),
            }
            for f in findings
        ],
        "facts": [
            {
                "evidence_id": f"fact:{f['id']}",
                "entity": f.get("entity"),
                "metric": f["metric"],
                "value": f["value"],
                "unit": f.get("unit"),
                "source_tier": source_tier(f.get("source_url")),
            }
            for f in facts
        ],
    }
    content = f"Hypothesis:\n{hypothesis}\n\nEvidence:\n{evidence}"
    response = llm.invoke([SystemMessage(content=VERDICT_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


@with_retry()
def _call_writeup_llm(
    verdicts: list[dict], framework_notes, revision_note: str | None, unexamined_branches: list[str]
) -> str:
    llm = get_llm(AGENT_NAME)
    content = (
        f"Verdicts:\n{verdicts}\n\n"
        f"Unexamined branches (no hypothesis was written for these):\n{unexamined_branches}\n\n"
        f"Framework guidance:\n{framework_notes}"
    )
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior write-up — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=WRITEUP_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _sanity_check(verdict: dict, tier_by_evidence_id: dict[str, str]) -> dict:
    """Code-level contradiction check — catches e.g. "supported" with zero
    supporting evidence_ids, or with evidence that's all social-tier.
    Downgrades the verdict and states the reason in `reasoning` (so it's
    visible in the brief, not a silent drop) rather than trusting the
    LLM's self-report."""
    verdict_value = verdict.get("verdict")
    evidence_ids = verdict.get("evidence_ids") or []

    if not evidence_ids and verdict_value in ("supported", "killed"):
        verdict = dict(verdict)
        verdict["reasoning"] = (
            f"[sanity check override: verdict was '{verdict_value}' with zero evidence_ids — "
            f"downgraded to inconclusive] {verdict.get('reasoning', '')}"
        ).strip()
        verdict["verdict"] = "inconclusive"
        return verdict

    if verdict_value in ("supported", "killed"):
        tiers = [tier_by_evidence_id.get(eid, "social") for eid in evidence_ids]
        if all(TIER_RANK.get(t, 0) <= TIER_RANK["social"] for t in tiers):
            verdict = dict(verdict)
            verdict["reasoning"] = (
                f"[sanity check override: verdict was '{verdict_value}' backed only by social-tier "
                f"evidence {evidence_ids} — a single social post isn't strong enough on its own, "
                f"downgraded to inconclusive] {verdict.get('reasoning', '')}"
            ).strip()
            verdict["verdict"] = "inconclusive"

    return verdict


def synthesizer_node(state: dict) -> dict:
    """Rule a verdict per hypothesis from DB evidence, then write up from
    the verdicts only.

    Reads: state['hypotheses'], state['issue_tree'], state['run_id'], state['revision_notes']['synthesizer']
    Writes: state['synthesis'] (includes 'unexamined_branches' and 'caveats' — disclosure
    lives in 'caveats', 'answer' is meant to be actionable on its own), state['messages'],
    state['run_path']
    """
    hypotheses = state.get("hypotheses", [])
    issue_tree = state.get("issue_tree", {})
    run_id = state.get("run_id")
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    verdicts = []
    for hypothesis in hypotheses:
        hypothesis_id = hypothesis.get("id")

        if not run_id or not hypothesis_id:
            verdicts.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "verdict": "insufficient_data",
                    "reasoning": "No run_id available yet — no evidence could be looked up.",
                    "evidence_ids": [],
                    "quantified": False,
                }
            )
            continue

        findings = get_findings(run_id, hypothesis_id)
        facts = get_facts(run_id, hypothesis_id)

        if not findings and not facts:
            verdicts.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "verdict": "insufficient_data",
                    "reasoning": "No findings or facts recorded for this hypothesis in this run.",
                    "evidence_ids": [],
                    "quantified": False,
                }
            )
            continue

        tier_by_evidence_id = {f"finding:{f['id']}": source_tier(f.get("source_url")) for f in findings}
        tier_by_evidence_id.update({f"fact:{f['id']}": source_tier(f.get("source_url")) for f in facts})

        raw = _call_verdict_llm(hypothesis, findings, facts)
        parsed = safe_extract_json(raw)
        parsed.setdefault("hypothesis_id", hypothesis_id)
        parsed.setdefault("verdict", "inconclusive")
        parsed.setdefault("evidence_ids", [])
        parsed.setdefault("reasoning", "LLM response could not be parsed; defaulting to inconclusive.")
        verdicts.append(_sanity_check(parsed, tier_by_evidence_id))

    # Deterministic, not LLM-guessed: which issue-tree branches have zero
    # hypotheses testing them at all (distinct from a hypothesis that WAS
    # examined and came back inconclusive — that's `unknowns`, not this).
    branch_names = {b.get("name") for b in issue_tree.get("branches", []) if b.get("name")}
    covered_branches = {h.get("branch") for h in hypotheses if h.get("branch")}
    unexamined_branches = sorted(branch_names - covered_branches)

    notes = retrieve_framework_notes("Pyramid Principle governing thought key line answer first")
    writeup_raw = _call_writeup_llm(verdicts, notes, revision_note, unexamined_branches)
    writeup = safe_extract_json(writeup_raw)

    synthesis = {
        "verdicts": verdicts,
        "answer": writeup.get("answer", ""),
        "key_arguments": writeup.get("key_arguments", []),
        "ruled_out": writeup.get("ruled_out", []),
        "unknowns": writeup.get("unknowns", []),
        "caveats": writeup.get("caveats", ""),
        "unexamined_branches": unexamined_branches,
    }

    run_logger.record(
        AGENT_NAME,
        verdict_count=len(verdicts),
        supported=sum(1 for v in verdicts if v["verdict"] == "supported"),
        killed=sum(1 for v in verdicts if v["verdict"] == "killed"),
    )

    return {
        "synthesis": synthesis,
        "messages": [AIMessage(content=writeup_raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
