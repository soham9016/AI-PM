"""Competitive-audit agent — PM path, stage 2 of 4.

Given company/sector anchors and the issue tree's branches (treated as
capability areas), checks what named competitors already ship for each
area, then classifies each area as:
  table_stakes   — most/all named competitors already have it
  differentiator — few or none of the named competitors have it
  parity_gap     — some competitors have it, partially or inconsistently
This classification feeds agents/pm.py's RICE impact scoring directly —
a feature that only brings a product to table_stakes parity shouldn't
score "massive impact" the same way a genuine differentiator can.

Two LLM calls, same shape as agents/researcher.py: plan (identify
competitors + per-branch queries) then classify (given the gathered
per-competitor findings, rule table_stakes/differentiator/parity_gap per
branch). Per-competitor findings are persisted to the evidence-base DB
via the same tools/evidence_extractor.py pipeline as every other agent —
source_url NOT NULL, Python-attached — tagged with a topic_id of
"competitor:<slug>" in the hypothesis_id column (see tools/db.py).

FAN-OUT IS BRANCH-RELEVANCE FILTERED, NOT BLIND: no finding row carries a
per-branch tag (extract_evidence doesn't know branches), so an earlier
version handed the classifier EVERY competitor finding for EVERY branch
and let it "match claim to branch" itself. Observed failure: a finding
about regional holiday promos got cited as evidence for a delivery-
logistics branch, because it was technically present in that branch's
evidence set purely from the blind duplication — the fan-out was already
wrong before the classifier had a chance to be. _claim_relevant_to_branch
now keyword-filters the fan-out (branch name + issue-tree sub-branches),
and _validate_competitor_evidence rejects any cited evidence_id that
isn't in that branch's (now-scoped) evidence set — the same
don't-trust-the-citation backstop pattern as
tools/evidence_extractor.py's _grounded_in_page.

EXACTLY ONE CLASSIFICATION PER BRANCH: the prompt asks for this, but
duplicate branch entries were observed anyway (same branch classified
differentiator and parity_gap in separate list entries). _dedupe_areas
collapses duplicates in code by merging their competitor lists — see
CLASSIFICATION IS DERIVED, NEVER ASSERTED below for why it no longer
needs to pick between the LLM's two conflicting labels.

ONE PIECE OF EVIDENCE CAN'T BACK TWO CLASSIFICATIONS: the branch-
relevance fan-out reduces but doesn't eliminate a claim being genuinely
keyword-relevant to more than one branch (e.g. an "AI-driven dynamic
pricing" claim can plausibly relate to both a data/analytics branch and
an operational-efficiency branch). Observed failure: the SAME
(competitor, evidence_id) pair backed a "differentiator" call in one
branch and a "parity_gap" call in another — one fact can't honestly
support two different competitive conclusions for the same competitor.
_dedupe_cross_branch_evidence keeps the citation only in the first
branch it appears in (iteration order), dropping it from every later one.

CLASSIFICATION IS DERIVED, NEVER ASSERTED: observed failure — a branch
was labeled "table_stakes" ("all competitors have this") with evidence
directly underneath showing 2 of 3 named competitors at "none". The LLM's
classification field was never trustworthy in the first place; it was
only ever picked between (see the old conservatism-based _dedupe_areas
logic this replaced). _derive_classification computes the real label in
code from the does_it counts, after dedup — majority "full" ->
table_stakes, mostly "none" with at most one "full" -> differentiator,
otherwise parity_gap. Zero comparable evidence -> no classification at
all (see below), not a guess.

A CLASSIFICATION WITH ZERO EVIDENCE IS NOT A CLASSIFICATION: two branches
were previously observed classified table_stakes with an empty
competitors list, and that unevidenced classification silently capped a
feature's RICE impact downstream (agents/pm.py trusts whatever
classification exists for a branch, since it has no way to know the
classification was invented from nothing). _derive_classification
returning None for a branch with no comparable evidence *is* this case —
that branch's name goes to `unclassified_branches` instead of `areas`,
so no classification means no impact cap, not a guessed one.

COMPETITOR NAMES ARE STRIPPED OF A LEAKED "Competitor: " PREFIX: both
prompts' JSON-shape examples used to use "Competitor A"/"Competitor B" as
placeholder names, which the model sometimes echoed literally (e.g.
"Competitor: Nike Training Club"). Fixed at the source in two layers: the
examples now use realistic brand-style names, and _strip_competitor_prefix
is applied as a backstop to every competitor name parsed from either LLM
call, regardless of whether the prompt fix alone was enough.
"""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import MAX_EVIDENCE_EXTRACTIONS, get_llm, invoke_llm
from tools.evidence_extractor import extract_evidence
from tools.fetch import fetch_page
from tools.search import web_search
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.relevance import is_relevant
from utils.retry import with_retry
from utils.text_similarity import longest_run_ratio, tokenize

AGENT_NAME = "competitive_audit"
MAX_COMPETITORS = 3
MAX_BRANCHES = 4
MAX_QUERIES = 6
MAX_FETCH_PER_QUERY = 1
ALLOWED_CLASSIFICATIONS = {"table_stakes", "differentiator", "parity_gap"}
# "not_comparable" -- distinct from "none": "none" means the comparison
# was made and the competitor doesn't do it; "not_comparable" means the
# three-part comparability test (same problem/mechanism/constraint) failed,
# so no judgment should be made at all. See CLASSIFY_SYSTEM_PROMPT.
ALLOWED_DOES_IT = {"full", "partial", "none", "not_comparable"}


# Stopwords excluded when building a branch's keyword set for the
# relevance-fan-out check below -- short function words carry no
# topical signal and would make almost any claim "relevant" to almost
# any branch.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is",
    "are", "with", "by", "at", "as", "from", "that", "this", "it", "be",
}

# _classify_areas' prompt size must stay well under its provider's
# per-request TPM ceiling REGARDLESS of how much evidence upstream
# extraction produced -- every finding gets duplicated across every
# branch (no per-branch tag exists on a finding row), so this cap bounds
# MAX_BRANCHES * MAX_EVIDENCE_PER_BRANCH_FOR_CLASSIFY items, not the raw
# finding count. This call is routed to Cerebras (see config.py's
# "competitive_audit_classify" entry, ~30k TPM vs. Groq's 6k) specifically
# because it's this agent's large-payload call; the cap stays regardless
# as defense in depth, not a replacement for the provider fix.
MAX_EVIDENCE_PER_BRANCH_FOR_CLASSIFY = 6
MAX_CLAIM_CHARS_FOR_CLASSIFY = 200

logger = logging.getLogger("business_copilot.competitive_audit")

PLAN_SYSTEM_PROMPT = """You are a competitive-intelligence research
planner. Given a business problem statement, its issue-tree branches
(treated as capability areas), and sector/geography/company anchors:

1. Identify up to 3 named competitors — use ones explicitly named in the
   problem statement if any; otherwise infer well-known competitors in
   the same sector/geography.
2. Propose up to 6 search queries that would reveal what each competitor
   already ships for these capability areas. Tag each query with which
   competitor and which branch it targets. Every query MUST include the
   sector term.

Respond with ONLY a JSON object of this exact shape — use the ACTUAL
competitor names you identified, not placeholder text:
{
  "competitors": ["Nike", "Peloton"],
  "queries": [
    {"query": "<search query>", "competitor": "Nike", "branch": "<issue tree branch>"}
  ]
}
"""

CLASSIFY_SYSTEM_PROMPT = """You are a competitive-intelligence analyst.
Given per-branch, per-competitor findings about what each named
competitor ships, classify EACH branch EXACTLY ONCE as one of:
  "table_stakes"   — most/all named competitors already have this
  "differentiator" — few or none of the named competitors have this
  "parity_gap"     — some competitors have it, partially or inconsistently
Do not list the same branch more than once — one classification per branch.

Before judging a competitor "full" or "partial" on a branch, apply this
three-part comparability test: does the competitor's feature (a) address
the SAME user problem, (b) use the SAME mechanism, and (c) operate under
the SAME constraint as what's being evaluated? If any of the three
doesn't hold, the pairing isn't comparable — use "not_comparable" instead
of guessing "none" or "partial". "none" means you compared and they
genuinely don't have it; "not_comparable" means the comparison itself
doesn't apply (e.g. their feature solves a related but different
problem, or works under a different constraint).

For each competitor+branch pair, state whether they do it "full",
"partial", "none", or "not_comparable", with the evidence_id backing that
judgment. Every evidence_id you cite MUST be one of the ids given to you,
AND must actually be evidence about THAT competitor for THAT branch —
never invent one, and never borrow evidence from an unrelated branch just
because it mentions the same competitor. If you have no findings for a
branch at all, omit it rather than guessing a classification.

Respond with ONLY a JSON object of this exact shape — use the ACTUAL
competitor names given to you, never the word "Competitor" as part of a
name:
{
  "areas": [
    {
      "branch": "<issue tree branch>",
      "classification": "table_stakes|differentiator|parity_gap",
      "rationale": "<1-2 sentences>",
      "competitors": [
        {"name": "Nike", "does_it": "full|partial|none|not_comparable", "evidence_id": "finding:12"}
      ]
    }
  ]
}
"""


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "unknown"


_COMPETITOR_PREFIX_RE = re.compile(r"^\s*competitor\s*[:\-]?\s*", re.IGNORECASE)


def _strip_competitor_prefix(name: str) -> str:
    """Backstop for a leaked "Competitor: " prefix (see module docstring)
    — applied at parse time to every competitor name from either LLM
    call, regardless of whether the prompt's example-name fix alone was
    enough to stop it."""
    return _COMPETITOR_PREFIX_RE.sub("", name or "").strip()


@with_retry()
def _plan_audit(problem_statement: str, issue_tree: dict, anchor_terms: dict, revision_note: str | None) -> str:
    llm = get_llm(AGENT_NAME)
    content = (
        f"Problem statement:\n{problem_statement}\n\n"
        f"Issue tree:\n{issue_tree}\n\n"
        f"Anchor terms:\n{anchor_terms}"
    )
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior audit — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


@with_retry()
def _classify_areas(branches: list[str], evidence_by_branch: dict, revision_note: str | None) -> str:
    # Distinct lookup key from AGENT_NAME -- see config.py's MODELS: this
    # is the large-payload call routed to Cerebras; _plan_audit above
    # stays on Groq under the plain "competitive_audit" key.
    content = f"Branches:\n{branches}\n\nPer-branch, per-competitor findings:\n{evidence_by_branch}"
    if revision_note:
        content += f"\n\nA previous review flagged this issue — address it:\n{revision_note}"
    response = invoke_llm(f"{AGENT_NAME}_classify", [SystemMessage(content=CLASSIFY_SYSTEM_PROMPT), HumanMessage(content=content)], json_mode=True)
    return response.content


def _priority_branches(issue_tree: dict) -> list[str]:
    return [b.get("name") for b in issue_tree.get("branches", []) if b.get("name")]


def _keywords(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def _branch_keywords(branch_name: str, issue_tree: dict) -> set[str]:
    keywords = _keywords(branch_name)
    for b in issue_tree.get("branches", []):
        if b.get("name") == branch_name:
            for sub in b.get("sub_branches") or []:
                keywords |= _keywords(sub)
            break
    return keywords


def _claim_relevant_to_branch(claim: str, branch_keywords: set[str]) -> bool:
    """Fails open (True) only when we have no keyword signal to filter
    on at all -- otherwise a claim must share at least one topical
    keyword with the branch it's being fanned into. See module docstring
    for the failure this replaces (blind fan-out into every branch)."""
    if not branch_keywords:
        return True
    return bool(_keywords(claim) & branch_keywords)


def _dedupe_areas(areas: list[dict]) -> list[dict]:
    """Exactly one entry per branch. Duplicate branch entries (observed:
    the same branch appearing twice with different LLM-asserted
    classifications) are merged by unioning their competitor lists
    (deduped by name) — classification itself is no longer picked
    between two untrusted LLM labels here; it's computed once, after
    merging, by _derive_classification from the real evidence."""
    by_branch: dict[str, dict] = {}
    for area in areas:
        branch = area.get("branch")
        existing = by_branch.get(branch)
        if existing is None:
            by_branch[branch] = area
            continue

        seen_names = {c.get("name") for c in existing.get("competitors", [])}
        for c in area.get("competitors", []):
            if c.get("name") not in seen_names:
                existing.setdefault("competitors", []).append(c)
                seen_names.add(c.get("name"))

    return list(by_branch.values())


def _derive_classification(competitors: list[dict]) -> str | None:
    """table_stakes/differentiator/parity_gap computed from the actual
    does_it values, never asserted by the LLM (see module docstring for
    the reported case this fixes: a branch labeled table_stakes with
    2-of-3 named competitors at "none" underneath it). not_comparable
    entries don't count toward the comparison -- they explicitly mean
    "this pairing doesn't apply", not "they don't have it"."""
    comparable = [c for c in competitors if c.get("does_it") in ("full", "partial", "none")]
    if not comparable:
        return None
    total = len(comparable)
    full_count = sum(1 for c in comparable if c["does_it"] == "full")
    none_count = sum(1 for c in comparable if c["does_it"] == "none")
    if full_count > total / 2:
        return "table_stakes"
    if none_count > total / 2 and full_count <= 1:
        return "differentiator"
    return "parity_gap"


def _dedupe_cross_branch_evidence(areas: list[dict]) -> list[dict]:
    """The same (competitor, evidence_id) pair backing a classification
    judgment in two different branches is suspicious -- one piece of
    evidence shouldn't simultaneously support two different competitive
    conclusions about the same competitor. First branch (iteration order)
    keeps the citation; it's dropped from every later branch that also
    cited it."""
    seen: set[tuple[str, str]] = set()
    for area in areas:
        branch = area.get("branch")
        kept = []
        for c in area.get("competitors", []):
            key = (c.get("name"), c.get("evidence_id"))
            if key in seen:
                logger.warning(
                    "competitive_evidence_reused_across_branches: competitor %r evidence_id %r already "
                    "backs a classification in another branch — dropping this citation from %r",
                    c.get("name"), c.get("evidence_id"), branch,
                )
                continue
            seen.add(key)
            kept.append(c)
        area["competitors"] = kept
    return areas


def _validate_competitor_evidence(area: dict, valid_ids_by_branch: dict[str, set[str]]) -> dict:
    """Drop any competitor entry whose cited evidence_id doesn't belong
    to THIS branch's (relevance-scoped) evidence set — the same
    don't-trust-the-citation backstop as tools/evidence_extractor.py's
    _grounded_in_page, applied to competitive classification instead of
    extraction."""
    branch = area.get("branch")
    valid_ids = valid_ids_by_branch.get(branch, set())
    kept = []
    for c in area.get("competitors", []):
        eid = c.get("evidence_id")
        if eid is not None and eid not in valid_ids:
            logger.warning(
                "competitive_evidence_mismatch: branch %r cited evidence_id %r, which isn't in "
                "that branch's evidence set — dropping this competitor judgment",
                branch, eid,
            )
            continue
        kept.append(c)
    area["competitors"] = kept
    return area


def competitive_audit_node(state: dict) -> dict:
    """Check what named competitors ship per issue-tree branch, persist
    findings to the evidence base, and classify each branch's competitive
    landscape (table_stakes / differentiator / parity_gap).

    Reads: state['problem_statement'], state['issue_tree'], state['anchor_terms'],
    state['run_id'], state['revision_notes']['competitive_audit']
    Writes: state['competitive_audit'] (each area de-duplicated to exactly one
    per branch, with evidence-validated competitor citations and no
    evidence_id reused across branches, plus top-level
    'competitors_with_no_evidence' and 'unclassified_branches' lists),
    state['messages'], state['run_path']
    """
    problem_statement = state.get("problem_statement", "")
    issue_tree = state.get("issue_tree", {})
    anchor_terms = state.get("anchor_terms", {})
    run_id = state.get("run_id")
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    branches = _priority_branches(issue_tree)[:MAX_BRANCHES]

    plan_raw = _plan_audit(problem_statement, issue_tree, anchor_terms, revision_note)
    plan = safe_extract_json(plan_raw)
    competitors = [
        _strip_competitor_prefix(c) for c in plan.get("competitors", []) if isinstance(c, str) and c.strip()
    ]
    competitors = [c for c in competitors if c][:MAX_COMPETITORS]
    queries = plan.get("queries", [])[:MAX_QUERIES]

    fetched_pages: list[tuple[dict, str]] = []  # (page, competitor)
    pages_skipped = 0
    search_error: str | None = None
    queries_by_competitor: dict[str, int] = {c: 0 for c in competitors}

    for q in queries:
        query_text = q.get("query", "")
        competitor = _strip_competitor_prefix(q.get("competitor", ""))
        if not query_text or competitor not in competitors:
            continue
        queries_by_competitor[competitor] += 1

        try:
            results = web_search.invoke({"query": query_text, "max_results": 5})
        except Exception as exc:  # noqa: BLE001 — any search failure must not crash the node
            search_error = str(exc)
            logger.warning("web_search failed for query %r: %s", query_text, exc)
            continue

        for result in results[:MAX_FETCH_PER_QUERY]:
            page = fetch_page.invoke({"url": result["url"]})
            unusable = page.get("error") or page.get("skipped") or not is_relevant(page.get("text", ""), anchor_terms)
            if unusable:
                pages_skipped += 1
                continue
            fetched_pages.append((page, competitor))

    # A named competitor whose plan yielded zero matching queries produces
    # zero evidence for the exact same reason a real "no coverage found"
    # result would — nothing here distinguishes "the plan silently missed
    # this competitor" from "we looked and found nothing". Log it so that
    # ambiguity isn't hidden the way it was for Blinkit in an earlier run.
    unqueried_competitors = [c for c, count in queries_by_competitor.items() if count == 0]
    for competitor in unqueried_competitors:
        logger.warning(
            "competitor %r was named but the plan proposed no matching queries for it "
            "(named %d competitors, %d total queries planned)",
            competitor, len(competitors), len(queries),
        )

    inserted_findings = 0
    rejected = 0
    extraction_errors = 0
    for page, competitor in fetched_pages[:MAX_EVIDENCE_EXTRACTIONS]:
        try:
            result = extract_evidence(
                page_text=page["text"],
                source_url=page["url"],
                source_name=page.get("title") or page["url"],
                run_id=run_id,
                hypothesis_id=f"competitor:{_slugify(competitor)}",
                hypothesis_statement=f"What does {competitor} ship for: {problem_statement}",
                problem_statement=problem_statement,
            )
        except Exception as exc:  # noqa: BLE001 — one page's extraction failing (e.g. a timeout that outlasted with_retry) must not crash the whole run
            extraction_errors += 1
            logger.warning("extract_evidence failed for %r: %s", page["url"], exc)
            continue
        inserted_findings += len(result["inserted_finding_ids"])
        rejected += result["rejected_count"]

    # Pull back what actually landed in the DB (not just what the LLM
    # claimed it found) so the classification call only ever sees real,
    # already-validated evidence_ids.
    from tools.db import get_findings_for_run  # local import: avoids a cycle with tools/evidence_extractor

    all_findings = get_findings_for_run(run_id) if run_id else []
    competitor_topic_ids = {f"competitor:{_slugify(c)}" for c in competitors}
    branch_keywords = {branch: _branch_keywords(branch, issue_tree) for branch in branches}
    evidence_by_branch: dict[str, list[dict]] = {branch: [] for branch in branches}
    for finding in all_findings:
        if finding.get("hypothesis_id") not in competitor_topic_ids:
            continue
        # Only fan a finding into a branch it's actually relevant to (see
        # module docstring for the "wrong branch" failure this replaces).
        # Capped and truncated per branch (see MAX_EVIDENCE_PER_BRANCH_FOR_CLASSIFY
        # above) so this fan-out can't blow the TPM ceiling regardless of
        # how many findings extraction produced upstream.
        claim_text = finding["claim"]
        if len(claim_text) > MAX_CLAIM_CHARS_FOR_CLASSIFY:
            claim_text = claim_text[:MAX_CLAIM_CHARS_FOR_CLASSIFY] + "..."
        for branch in branches:
            if len(evidence_by_branch[branch]) >= MAX_EVIDENCE_PER_BRANCH_FOR_CLASSIFY:
                continue
            if not _claim_relevant_to_branch(finding["claim"], branch_keywords[branch]):
                continue
            evidence_by_branch[branch].append(
                {"evidence_id": f"finding:{finding['id']}", "competitor": finding["hypothesis_id"], "claim": claim_text}
            )

    if branches and any(evidence_by_branch.values()):
        classify_raw = _classify_areas(branches, evidence_by_branch, revision_note)
        classify_parsed = safe_extract_json(classify_raw)
    else:
        classify_raw = '{"areas": []}'
        classify_parsed = {"areas": []}

    valid_ids_by_branch = {
        branch: {item["evidence_id"] for item in items} for branch, items in evidence_by_branch.items()
    }

    # Validate does_it against ALLOWED_* (classification itself is no
    # longer trusted from the LLM at all -- kept only as "llm_classification"
    # for the mismatch comparison/log below); strip any leaked "Competitor: "
    # prefix from names; reject citations that don't belong to that
    # branch's evidence set; then collapse duplicate branch entries.
    areas = []
    for area in classify_parsed.get("areas", []):
        competitors_out = [
            {**c, "name": _strip_competitor_prefix(c.get("name", ""))}
            for c in area.get("competitors", []) if c.get("does_it") in ALLOWED_DOES_IT
        ]
        validated_area = _validate_competitor_evidence(
            {
                "branch": area.get("branch"),
                "llm_classification": area.get("classification"),
                "rationale": area.get("rationale", ""),
                "competitors": competitors_out,
            },
            valid_ids_by_branch,
        )
        areas.append(validated_area)
    areas = _dedupe_areas(areas)
    areas = _dedupe_cross_branch_evidence(areas)

    # Classification is computed ONCE, after merging/validation, from the
    # real evidence -- never asserted by the LLM. A branch with zero
    # comparable evidence (from the start, or emptied out by validation/
    # dedup above) derives to None, which IS the unclassified case: no
    # classification means no impact cap downstream, not a guessed one.
    areas_with_evidence = []
    unclassified_branches = []
    for area in areas:
        computed = _derive_classification(area.get("competitors", []))
        if computed is None:
            unclassified_branches.append(area.get("branch"))
            continue
        llm_said = area.pop("llm_classification", None)
        if llm_said not in ALLOWED_CLASSIFICATIONS:
            llm_said = None
        area["classification"] = computed
        if llm_said is not None and llm_said != computed:
            logger.warning(
                "competitive_classification_corrected: branch %r -- LLM said %r, evidence-derived "
                "classification is %r",
                area.get("branch"), llm_said, computed,
            )
            area["rationale"] = (area.get("rationale") or "") + f" [classification corrected from {llm_said!r} to {computed!r} based on competitor evidence]"
        areas_with_evidence.append(area)
    areas = areas_with_evidence

    # A named competitor with ZERO evidence anywhere (not just zero for
    # one branch) is a different, more serious signal than
    # unqueried_competitors above (zero queries planned) -- surface it
    # explicitly rather than letting the competitor just never appear in
    # any area's competitors list.
    evidenced_topic_ids = {f.get("hypothesis_id") for f in all_findings if f.get("hypothesis_id") in competitor_topic_ids}
    competitors_with_no_evidence = [
        c for c in competitors if f"competitor:{_slugify(c)}" not in evidenced_topic_ids
    ]

    audit = {
        "competitors": competitors,
        "areas": areas,
        "competitors_with_no_evidence": competitors_with_no_evidence,
        "unclassified_branches": unclassified_branches,
    }
    if not areas:
        audit["unresolved"] = True
        audit["reason"] = search_error or "no competitor evidence was gathered to classify"

    run_logger.record(
        AGENT_NAME,
        competitors=competitors,
        branches=len(branches),
        pages_fetched=len(fetched_pages),
        pages_skipped=pages_skipped,
        queries_by_competitor=queries_by_competitor,
        unqueried_competitors=unqueried_competitors,
        competitors_with_no_evidence=competitors_with_no_evidence,
        unclassified_branches=unclassified_branches,
        evidence_findings_inserted=inserted_findings,
        evidence_rejected=rejected,
        extraction_errors=extraction_errors,
        areas_classified=len(areas),
    )

    return {
        "competitive_audit": audit,
        "messages": [AIMessage(content=classify_raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
