"""Researcher agent — plans web searches against hypothesis kill conditions,
runs them, fetches top pages, and compiles findings.

Two stages:
  (a) search + fetch, then write a qualitative summary to state — this is
      the existing behaviour, unchanged, and is the goal of this agent;
      numeric evidence is a bonus, not the point.
  (b) for each successfully fetched page (capped at
      config.MAX_EVIDENCE_EXTRACTIONS, and only for pages whose target
      hypothesis actually exists in state['hypotheses']), run it through
      tools/evidence_extractor.py and persist the resulting findings/facts
      rows to the evidence-base DB, tagged with run_id + hypothesis_id.

NOTE on run_id: this node reads state.get("run_id") — state.py now
guarantees it's pre-populated by new_state(), so this node does NOT
return run_id in its update dict (returning it back would be a redundant
write path against a field that already has a stable value).

Pages that fetch_page() marks unusable (binary extension, non-text
Content-Type, or post-extraction garbage — see tools/fetch.py) are
counted and logged (pages_skipped, skip_reasons) rather than silently
dropped, so a run where everything was filtered out is visibly
distinguishable from a run where search itself found nothing.

PRIMARY SOURCES: when the problem statement names a specific company,
the planner is instructed to include at least one query aimed at that
company's own primary materials (investor relations, quarterly/annual
results) rather than only secondary commentary about it — prompt-only,
no guarantee the search index actually surfaces them. This does not by
itself fix evidence *quality* once pages are fetched (see
agents/synthesizer.py's source_tier weighting for that); it only
improves the odds a primary source is in the result set to begin with.

RELEVANCE GATE: the query planner also derives anchor terms from the
problem statement, split by category (sector / geography / company). A
page is usable as evidence if its fetched text matches EITHER the
company anchor OR the sector anchor — not sector alone. An earlier
version required sector-only matching, which rejected a Swiggy earnings
article (the single most relevant source in that run's results) because
its text never spelled out "Quick commerce" verbatim, while a company
match ("Swiggy") would have caught it immediately; a thin academic paper
and a LinkedIn carousel passed instead, purely because they happened to
contain the sector phrase. A named company from the problem statement is
at least as strong a relevance signal as the sector. Sector matching is
also fuzzy now — case-insensitive, and tolerant of hyphen/space variants
and the standard first-letter abbreviation style ("quick commerce" also
matches "quick-commerce" and "q-commerce"), since a plain substring
match on the exact planner-generated phrase was too brittle. Fails open
(page is treated as relevant) only when the planner produced neither a
sector nor a company term — this is a quality heuristic, not a safety
gate, so a planner hiccup shouldn't turn every run into "insufficient
data" by rejecting everything. Traceability checks (source_url present,
DB constraints) only verify a claim came from *somewhere real* — they
say nothing about whether that source is on topic, which is what this
gate is for.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import MAX_EVIDENCE_EXTRACTIONS, get_llm
from tools.evidence_extractor import extract_evidence
from tools.fetch import fetch_page
from tools.search import web_search
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.relevance import is_relevant
from utils.retry import with_retry

AGENT_NAME = "researcher"
MAX_QUERIES = 4
MAX_FETCH_PER_QUERY = 2

logger = logging.getLogger("business_copilot.researcher")

SYSTEM_PROMPT = """You are a research planner.

Given a business problem statement and a list of hypotheses (with kill
conditions), do two things:

1. Identify anchor terms from the problem statement, in three separate
   categories:
   - sector: the industry/sector — REQUIRED, always provide a specific
     one. "Specialty chemicals" is usable; "chemicals" alone is too
     broad (matches tires, beverages, anything vaguely chemical).
   - geography: the region/country, if the problem statement names one.
     If it doesn't literally name a country but references a
     sub-national locality tier or market descriptor (e.g. "Tier-2
     cities", "metro vs. non-metro"), infer the country from other
     context if identifiable (a named company, currency, named cities)
     and include both, e.g. "India (Tier-2 cities)". Empty string only
     if truly nothing is inferable.
   - company: the specific company name, if one is named (empty string
     if the problem statement describes a generic/anonymized entity
     instead, e.g. "a mid-sized manufacturer").
   A page is only treated as relevant evidence later if it mentions the
   SECTOR term — geography alone is not enough, since a country-overview
   page mentions the geography but has nothing to do with the sector.

2. Propose up to 4 web search queries that would surface evidence
   bearing on the hypotheses' kill conditions — prioritize queries that
   could disprove a hypothesis, not just confirm it. Every query MUST
   include the sector term, plus the geography/company term when one
   exists — a query with no sector anchor will happily match an
   unrelated industry. Every query's "targets_hypothesis" MUST be one of
   the exact hypothesis ids given to you — never invent one.

   If a company IS named (the company anchor is non-empty), at least ONE
   of your queries MUST specifically target that company's own primary
   sources — its investor relations page, quarterly/annual results, or
   official filings — rather than only secondary commentary about it,
   e.g. "{company} investor relations quarterly results" or "{company}
   annual report". Secondary sources (news, analyst commentary, social
   posts) are still valuable and your other queries can surface them,
   but without at least one query aimed at the company's own materials,
   the result set skews toward commentary ABOUT the company instead of
   what the company itself has actually reported.

   Each hypothesis carries an "evidence_type" — match your query strategy
   to it, since different claim types are settled by different kinds of
   source:
     REGULATORY  — target the actual law/regulation text or an official
                   regulator's site, not commentary about it (e.g. "site
                   specific regulator or official gazette" style queries)
     FINANCIAL   — target filings, investor relations, financial results
     PRODUCT     — target product documentation, changelogs, app store listings
     USER        — target survey data, research reports, user studies
     MARKET      — target industry/market-research reports
     COMPETITIVE — target competitor announcements, comparison coverage

Respond with ONLY a JSON object of this exact shape:
{
  "anchor_terms": {"sector": "<sector term>", "geography": "<geography term or empty string>", "company": "<company term or empty string>"},
  "queries": [
    {"query": "<search query, containing the sector term>", "targets_hypothesis": "<hypothesis id>"}
  ]
}
"""

SUMMARY_SYSTEM_PROMPT = """You are a research summarizer.

Given raw search results and fetched page excerpts, write concise
findings, each tagged with which hypothesis it bears on and whether it
supports or undermines that hypothesis.

Respond with ONLY a JSON object of this exact shape:
{
  "findings": [
    {
      "hypothesis_id": "<id>",
      "summary": "<1-3 sentence finding>",
      "supports_hypothesis": true,
      "source_url": "<url>"
    }
  ]
}
"""


@with_retry()
def _plan_queries(problem_statement: str, hypotheses: list[dict], revision_note: str | None) -> str:
    llm = get_llm(AGENT_NAME)
    content = f"Problem statement:\n{problem_statement}\n\nHypotheses:\n{hypotheses}"
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior research — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


@with_retry()
def _summarize_findings(raw_material: str) -> str:
    llm = get_llm(AGENT_NAME)
    response = llm.invoke(
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=raw_material)]
    )
    return response.content


def _hypothesis_statement(hypotheses: list[dict], hypothesis_id: str | None) -> str:
    for hyp in hypotheses:
        if hyp.get("id") == hypothesis_id:
            return hyp.get("statement", "")
    return ""


def _ensure_findings(
    findings: list[dict], search_error: str | None, pages_skipped: int, skip_reasons: list[str]
) -> list[dict]:
    """Guarantee a non-empty list so `not research_findings` reliably means
    "hasn't run" rather than "ran and found nothing usable" — those must
    not look the same to the ladder. Distinguishes *why* nothing came
    back: a dead search vs. pages that were fetched but filtered out."""
    if findings:
        return findings
    if search_error:
        reason = search_error
    elif pages_skipped:
        sample = "; ".join(skip_reasons[:3])
        reason = f"{pages_skipped} page(s) fetched but filtered out as unusable (e.g. {sample})"
    else:
        reason = "search ran but returned no usable results for any query"
    return [
        {
            "hypothesis_id": None,
            "summary": f"Researcher ran but found nothing: {reason}",
            "supports_hypothesis": None,
            "source_url": None,
            "unresolved": True,
        }
    ]


def researcher_node(state: dict) -> dict:
    """Search + fetch evidence targeting each hypothesis's kill conditions,
    then extract structured findings/facts into the evidence-base DB.

    Reads: state['problem_statement'], state['hypotheses'], state['run_id'], state['revision_notes']['researcher']
    Writes: state['research_findings'], state['messages'], state['run_path']
    """
    problem_statement = state.get("problem_statement", "")
    hypotheses = state.get("hypotheses", [])
    real_hypothesis_ids = {h.get("id") for h in hypotheses if h.get("id")}
    run_id = state.get("run_id")
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    plan_raw = _plan_queries(problem_statement, hypotheses, revision_note)
    plan = safe_extract_json(plan_raw)
    raw_anchors = plan.get("anchor_terms")
    anchor_terms = {
        "sector": str((raw_anchors or {}).get("sector") or "").strip() if isinstance(raw_anchors, dict) else "",
        "geography": str((raw_anchors or {}).get("geography") or "").strip() if isinstance(raw_anchors, dict) else "",
        "company": str((raw_anchors or {}).get("company") or "").strip() if isinstance(raw_anchors, dict) else "",
    }
    queries = plan.get("queries", [])[:MAX_QUERIES]

    raw_material_parts = []
    # (page dict, hypothesis_id) for every page that fetched cleanly AND
    # targets a real hypothesis — stage (b) extracts structured evidence
    # from these.
    fetched_pages: list[tuple[dict, str]] = []
    search_error: str | None = None
    pages_skipped = 0
    skip_reasons: list[str] = []

    for q in queries:
        query_text = q.get("query", "")
        target_hypothesis_id = q.get("targets_hypothesis")
        if not query_text:
            continue

        try:
            results = web_search.invoke({"query": query_text, "max_results": 5})
        except Exception as exc:  # noqa: BLE001 — any search failure (missing API key, network) must not crash the node
            search_error = str(exc)
            logger.warning("web_search failed for query %r: %s", query_text, exc)
            continue

        raw_material_parts.append(f"QUERY: {query_text} (targets {target_hypothesis_id})")

        for result in results[:MAX_FETCH_PER_QUERY]:
            page = fetch_page.invoke({"url": result["url"]})

            irrelevant = (
                not page.get("error")
                and not page.get("skipped")
                and page.get("text")
                and not is_relevant(page["text"], anchor_terms)
            )

            if page.get("skipped"):
                pages_skipped += 1
                skip_reasons.append(f"{result['url']}: {page.get('skip_reason')}")
            elif irrelevant:
                pages_skipped += 1
                skip_reasons.append(
                    f"{result['url']}: irrelevant (no match for sector {anchor_terms['sector']!r} "
                    f"or company {anchor_terms['company']!r})"
                )

            unusable = page.get("error") or page.get("skipped") or irrelevant
            excerpt = page["text"][:1500] if not unusable else result.get("content", "")
            raw_material_parts.append(f"SOURCE: {result['url']}\n{excerpt}")

            if not unusable and page.get("text") and target_hypothesis_id in real_hypothesis_ids:
                fetched_pages.append((page, target_hypothesis_id))

    raw_material = "\n\n".join(raw_material_parts) or "No search results were retrieved."
    summary_raw = _summarize_findings(raw_material)
    parsed = safe_extract_json(summary_raw)

    # Stage (b): structured extraction + persistence, one page at a time,
    # capped so a run stays bounded on Groq's free tier.
    inserted_findings = 0
    inserted_facts = 0
    rejected = 0
    extraction_errors = 0
    for page, hypothesis_id in fetched_pages[:MAX_EVIDENCE_EXTRACTIONS]:
        try:
            result = extract_evidence(
                page_text=page["text"],
                source_url=page["url"],
                source_name=page.get("title") or page["url"],
                run_id=run_id,
                hypothesis_id=hypothesis_id,
                hypothesis_statement=_hypothesis_statement(hypotheses, hypothesis_id),
                problem_statement=problem_statement,
            )
        except Exception as exc:  # noqa: BLE001 — one page's extraction failing (e.g. a timeout that outlasted with_retry) must not crash the whole run
            extraction_errors += 1
            logger.warning("extract_evidence failed for %r: %s", page["url"], exc)
            continue
        inserted_findings += len(result["inserted_finding_ids"])
        inserted_facts += len(result["inserted_fact_ids"])
        rejected += result["rejected_count"]

    findings = _ensure_findings(parsed.get("findings", []), search_error, pages_skipped, skip_reasons)

    run_logger.record(
        AGENT_NAME,
        queries=len(queries),
        anchor_terms=anchor_terms,
        pages_fetched=len(fetched_pages),
        pages_skipped=pages_skipped,
        skip_reasons=skip_reasons,
        findings_summary_count=len(findings),
        evidence_findings_inserted=inserted_findings,
        evidence_facts_inserted=inserted_facts,
        evidence_rejected=rejected,
        extraction_errors=extraction_errors,
        search_error=search_error,
    )

    return {
        "research_findings": findings,
        "messages": [AIMessage(content=summary_raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
