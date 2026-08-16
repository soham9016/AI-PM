"""Primary-research agent — PM path, stage 1 of 4 (replaces
hypothesis/researcher/analyst/synthesizer for routing_mode "pm").

Targets authoritative sources for THE ASK, not causes: statute and
regulation text, official guidance, company policy pages, product docs.
A PM problem's premise doesn't need diagnosing (see agents/structurer.py
for why routing_mode exists at all) — it needs grounding in what the
actual rules/product surface actually say.

Same provenance rules as the diagnostic path: source_url is NOT NULL,
attached by Python from the page actually fetched, never from the
model (see tools/evidence_extractor.py). Rows are tagged with a topic_id
("primary") in the hypothesis_id column — see tools/db.py's module
docstring for why that's not a schema mismatch, just column reuse.

Unlike agents/researcher.py, this agent does NOT derive its own anchor
terms — it reads state['anchor_terms'], computed once by
agents/structurer.py, since re-deriving them here would risk drifting
from what agents/competitive_audit.py uses for the same run.

CURRENT STATE, NOT JUST RULES: a recommendation is worthless if it
proposes something the company already ships. state['current_state_findings']
comes from queries CODE builds (_build_current_state_queries), not the
LLM — see CURRENT-STATE QUERIES ARE CODE-BUILT below. Their extracted
evidence is tagged with a separate topic_id (CURRENT_STATE_TOPIC_ID) so
it doesn't mix into the "what the rules say" findings, then pulled back
from the DB into state['current_state_findings'] — no extra LLM call
needed, since the extraction that already happened is the current-state
data. agents/solution_framing.py reads this to decide whether a
candidate solution is new, partial, or something the company already has.

CURRENT-STATE QUERIES ARE CODE-BUILT AND DOMAIN-FILTERED, NOT LLM-
PLANNED: observed failure — LLM-authored "current_state" queries like
"<company> app features" reliably surfaced vendor-marketing content
(SaaS tool blogs, listicle articles) that merely mentioned the topic,
never the company's own pages. _build_current_state_queries constructs
queries scoped at the company's own domain (a conservative slug guess —
a wrong guess just returns zero results, the safe failure mode), its app
store listing, and its help centre, and _is_company_owned then filters
EVERY current_state-tagged fetch result (code- or LLM-authored query, in
case the LLM emits one anyway) down to pages actually on that domain or
a recognized app-store domain — a page that merely mentions the company
does not qualify. If nothing survives that filter, current_state_findings
stays empty and tools/brief.py says so explicitly rather than silently
omitting the section (which read as "not checked" instead of "checked,
found nothing").

FUNNEL-TARGETED QUERIES ARE ALSO CODE-BUILT, NOT LLM-PLANNED: observed
failure — funnel_decomposition produces a specific, useful research plan
(each public stage's evidence_needed names exactly what to look for and
in what kind of source), but the LLM query planner, even when shown that
plan and told to prioritize it, consistently ignored it in favor of
generic queries — funnel_verdict was structurally unable to resolve to
anything but "undetermined" because the evidence the funnel asked for
was never actually searched for. _build_funnel_queries now builds one
query per PUBLIC-locatability stage directly from that stage's own
evidence_needed text (_public_only_funnel still filters to public stages
in code first, as before), combined with the company/sector anchor and a
site hint when evidence_needed names a source type (app store, reddit,
forum) — the LLM is no longer involved in this at all. Each query is
logged with the stage_name it came from (see run_logger.record's
funnel_queries_by_stage) so the mapping is inspectable. The verdict
judgment (_summarize_findings) still sees the FULL funnel, since
"undetermined" has to be judged against every stage, not just the
publicly-searchable ones. No new LLM call for the verdict itself:
SUMMARY_SYSTEM_PROMPT's existing summarization pass judges which funnel
stage the gathered evidence points to (or "undetermined"), validated in
code against the funnel's real stage names so an invented stage name can
never reach state — see _normalize_funnel_verdict.

With funnel-targeted and current-state queries both fully code-built,
the LLM query planner's only remaining job is "authoritative" queries
(regulation/official guidance) — see SYSTEM_PROMPT.
"""

import logging
import re
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import MAX_EVIDENCE_EXTRACTIONS, get_llm
from tools.evidence_extractor import extract_evidence
from tools.fetch import fetch_page
from tools.search import web_search
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.relevance import is_relevant
from utils.retry import with_retry
from utils.source_tier import TIER_RANK, source_tier

AGENT_NAME = "primary_research"
TOPIC_ID = "primary"
CURRENT_STATE_TOPIC_ID = "current_state"
MAX_QUERIES = 3
MAX_FETCH_PER_QUERY = 2
# App-store domains a current-state result can live on without the
# company's own name being IN the domain -- still must pass is_relevant
# (anchor terms in the page text) to count, this only exempts the URL
# check for domain ownership since every listing shares these hosts.
_APP_STORE_DOMAINS = ("play.google.com", "apps.apple.com")
# Keyword -> search-engine site hint, checked against a funnel stage's
# own evidence_needed text (which itself uses this vocabulary, since we
# seeded it into agents/funnel_decomposition.py's prompt) so the query
# actually targets the kind of source the stage named.
_SOURCE_HINTS = (
    (("app store", "play store", "app review"), "site:play.google.com OR site:apps.apple.com"),
    (("reddit", "subreddit"), "site:reddit.com"),
    (("forum", "community"), "forum"),
    (("complaint",), "complaints"),
)

logger = logging.getLogger("business_copilot.primary_research")

SYSTEM_PROMPT = """You are a primary-source research planner. Your job
is finding what the AUTHORITATIVE source says — not diagnosing a cause.
The problem statement already states its premise; you are not here to
verify it, only to find out what the actual rules/product/policy allow
or require. (Current-state "what does the company already ship" queries
and funnel-stage-targeted queries are built separately, in code, from
state that already exists — you do not need to plan those; just focus
on authoritative sources.)

Given a business problem statement, its issue-tree branches, and
sector/geography/company anchor terms, propose up to 3 search queries
for: statute/regulation text (the law itself, or an official regulator's
site — not commentary), official guidance from a standards body or
regulator, or the named company's own investor/official materials.

Prefer official/primary sources over commentary. Every query MUST
include the sector term (and company term, if one exists) so results
stay on topic. Tag each query with the issue-tree branch it targets.

Respond with ONLY a JSON object of this exact shape:
{
  "queries": [
    {"query": "<search query>", "branch": "<issue tree branch this targets>"}
  ]
}
"""

SUMMARY_SYSTEM_PROMPT = """You are a research summarizer. Given raw
search results and fetched page excerpts about what an authoritative
source (regulation, official guidance, company policy/docs) actually
says, write concise findings — plain statements of what the source says,
not a diagnosis of why anything happened.

If funnel stages are given below, also review the gathered evidence and
state which stage name it points to as the likely site of the drop-off
— or "undetermined" if the evidence doesn't clearly distinguish between
stages. "undetermined" is a legitimate, useful answer, not a failure —
say it plainly rather than guessing. If no funnel stages are given,
leave funnel_verdict as an empty string.

Respond with ONLY a JSON object of this exact shape:
{
  "findings": [
    {"branch": "<issue tree branch>", "summary": "<1-3 sentence finding>", "source_url": "<url>"}
  ],
  "funnel_verdict": "<exact stage name from the stages given, or \\"undetermined\\", or empty string if no funnel was given>"
}
"""


@with_retry()
def _plan_queries(problem_statement: str, issue_tree: dict, anchor_terms: dict, revision_note: str | None) -> str:
    llm = get_llm(AGENT_NAME)
    content = (
        f"Problem statement:\n{problem_statement}\n\n"
        f"Issue tree:\n{issue_tree}\n\n"
        f"Anchor terms:\n{anchor_terms}"
    )
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior research — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


@with_retry()
def _summarize_findings(raw_material: str, funnel: dict) -> str:
    llm = get_llm(AGENT_NAME)
    content = raw_material
    if funnel:
        content += f"\n\nFunnel stages to judge against:\n{funnel}"
    response = llm.invoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _public_only_funnel(funnel: dict) -> dict:
    """Only public-locatability stages are ever handed to the query
    planner -- code-enforced, not just prompted for (see module
    docstring). An internal-only stage is simply absent from what the
    planner sees; there's nothing for it to target even if it tried."""
    if not funnel:
        return {}
    public_stages = [s for s in funnel.get("stages", []) if s.get("evidence_locatability") == "public"]
    if not public_stages:
        return {}
    return {"metric": funnel.get("metric"), "stages": public_stages}


def _source_hint(evidence_needed: str) -> str:
    """Map a stage's evidence_needed text to a search-engine site hint
    for the kind of source it names -- app store reviews, subreddits,
    forums, complaint sites. Keyword-matched against the same vocabulary
    agents/funnel_decomposition.py's prompt was seeded with, so this
    isn't guessing at arbitrary free text."""
    text = evidence_needed.lower()
    for keywords, hint in _SOURCE_HINTS:
        if any(k in text for k in keywords):
            return hint
    return ""


def _build_funnel_queries(public_funnel: dict, anchor_terms: dict) -> list[dict]:
    """One query per PUBLIC-locatability funnel stage, built directly
    from that stage's own evidence_needed text -- not a general
    restatement of the problem, and not left to the LLM query planner
    (see module docstring: it was shown this exact plan and still
    ignored it). This is what lets funnel_verdict ever resolve to a
    named stage instead of "undetermined" being the only reachable
    answer."""
    company_or_sector = (anchor_terms.get("company") or anchor_terms.get("sector") or "").strip()
    queries = []
    for stage in public_funnel.get("stages", []):
        evidence_needed = (stage.get("evidence_needed") or "").strip()
        if not evidence_needed:
            continue
        hint = _source_hint(evidence_needed)
        query_text = " ".join(p for p in (company_or_sector, evidence_needed, hint) if p)
        queries.append({
            "query": query_text,
            "branch": None,
            "query_type": "funnel",
            "stage_name": stage.get("name"),
        })
    return queries


def _normalize_company_token(company: str) -> str:
    return re.sub(r"[^a-z0-9]", "", company.lower())


def _is_company_owned(url: str, company: str) -> bool:
    """Does this URL actually belong to the named company -- its own
    domain, or a recognized app-store listing (still subject to the
    normal is_relevant anchor-term check on the page text, since every
    app-store listing shares the same domain). A page that merely
    MENTIONS the company on someone else's domain does not qualify --
    see module docstring for the vendor-marketing-content failure this
    replaces."""
    token = _normalize_company_token(company)
    if not token:
        return False
    netloc = urlparse(url).netloc.lower()
    if any(d in netloc for d in _APP_STORE_DOMAINS):
        return True
    # ".".join stripped so a multi-label domain like "cult.fit" still
    # matches a company name normalized to "cultfit".
    return token in netloc.replace(".", "").replace("-", "")


def _build_current_state_queries(anchor_terms: dict) -> list[dict]:
    """Queries scoped at the company's OWN web presence -- a conservative
    domain-slug guess, its app store listing, its help centre -- built in
    code rather than trusted to an LLM-phrased query (see module
    docstring). A wrong domain guess just returns zero results, the safe
    failure mode; _is_company_owned then filters every current_state
    fetch result (from these queries or an LLM-authored one) down to
    pages that are actually the company's own."""
    company = (anchor_terms.get("company") or "").strip()
    if not company:
        return []
    domain_guess = f"{_normalize_company_token(company)}.com"
    return [
        {
            "query": f"site:{domain_guess} features OR products OR services OR support",
            "branch": None, "query_type": "current_state", "stage_name": None,
        },
        {
            "query": f'"{company}" app store OR play store listing',
            "branch": None, "query_type": "current_state", "stage_name": None,
        },
    ]


def _normalize_funnel_verdict(raw_value, funnel: dict) -> str:
    """Never trust an invented stage name -- validated against the
    funnel's real stage names + "undetermined". Empty funnel or empty/
    invalid verdict both normalize to "" (not yet determined) rather
    than "undetermined", which is a specific, meaningful claim (evidence
    was gathered and didn't distinguish) that only applies when a real
    funnel existed to judge against."""
    if not funnel or not isinstance(raw_value, str):
        return ""
    candidate = raw_value.strip()
    if candidate == "undetermined":
        return "undetermined"
    stage_names = {s.get("name") for s in funnel.get("stages", [])}
    if candidate in stage_names:
        return candidate
    if candidate:
        logger.warning("funnel_verdict %r is not a real stage name in %r — falling back to 'undetermined'", candidate, stage_names)
        return "undetermined"
    return ""


def _ensure_findings(findings: list[dict], search_error: str | None, pages_skipped: int, skip_reasons: list[str]) -> list[dict]:
    """Same "ran but empty" guarantee as agents/researcher.py's
    _ensure_findings — `not primary_research_findings` must reliably mean
    "hasn't run", not "ran and found nothing usable"."""
    if findings:
        return findings
    if search_error:
        reason = search_error
    elif pages_skipped:
        sample = "; ".join(skip_reasons[:3])
        reason = f"{pages_skipped} page(s) fetched but filtered out as unusable (e.g. {sample})"
    else:
        reason = "search ran but returned no usable results for any query"
    return [{"branch": None, "summary": f"Primary research ran but found nothing: {reason}", "source_url": None, "unresolved": True}]


def _format_current_state_findings(findings: list[dict], facts: list[dict]) -> list[dict]:
    """Uniform {claim, source_url, source_name} shape regardless of
    whether the underlying row was a finding or a fact, so downstream
    consumers (agents/solution_framing.py, tools/brief.py) don't need to
    know the difference."""
    formatted = [
        {"claim": f["claim"], "source_url": f["source_url"], "source_name": f.get("source_name")}
        for f in findings
    ]
    for f in facts:
        unit = f.get("unit") or ""
        metric_text = f"{f.get('metric', 'metric')}: {f.get('value')} {unit}".strip()
        # entity often carries the business line/segment a number belongs
        # to (e.g. "Instamart" vs. "Food delivery") -- dropping it here
        # would silently lose the one signal that lets a reader tell two
        # different businesses' numbers apart in tools/brief.py.
        entity = f.get("entity")
        claim = f"{entity} — {metric_text}" if entity else metric_text
        formatted.append({
            "claim": claim,
            "source_url": f["source_url"],
            "source_name": f.get("source_name"),
        })
    return formatted


def primary_research_node(state: dict) -> dict:
    """Search + fetch authoritative sources for the ask AND current-state
    evidence of what the company already ships, then extract structured
    findings/facts into the evidence-base DB tagged by topic_id
    ("primary" for authoritative, "current_state" for current-state).

    Reads: state['problem_statement'], state['issue_tree'], state['anchor_terms'],
    state['funnel'], state['run_id'], state['revision_notes']['primary_research']
    Writes: state['primary_research_findings'], state['current_state_findings'],
    state['funnel_verdict'], state['messages'], state['run_path']
    """
    problem_statement = state.get("problem_statement", "")
    issue_tree = state.get("issue_tree", {})
    anchor_terms = state.get("anchor_terms", {})
    funnel = state.get("funnel", {})
    run_id = state.get("run_id")
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    public_funnel = _public_only_funnel(funnel)
    funnel_queries = _build_funnel_queries(public_funnel, anchor_terms)
    current_state_queries = _build_current_state_queries(anchor_terms)

    plan_raw = _plan_queries(problem_statement, issue_tree, anchor_terms, revision_note)
    plan = safe_extract_json(plan_raw)
    llm_queries = [
        {**q, "query_type": "authoritative", "stage_name": None}
        for q in plan.get("queries", [])[:MAX_QUERIES] if q.get("query")
    ]

    # Funnel and current-state queries are code-built and go first --
    # they are guaranteed coverage, not subject to the LLM's own budget
    # or judgment (see module docstring for why that judgment failed).
    queries = funnel_queries + current_state_queries + llm_queries

    raw_material_parts = []
    fetched_pages: list[tuple[dict, str]] = []  # (page, query_type)
    search_error: str | None = None
    pages_skipped = 0
    skip_reasons: list[str] = []
    company = anchor_terms.get("company") or ""

    for q in queries:
        query_text = q.get("query", "")
        query_type = q.get("query_type") if q.get("query_type") in ("authoritative", "current_state", "funnel") else "authoritative"
        if not query_text:
            continue

        try:
            results = web_search.invoke({"query": query_text, "max_results": 5})
        except Exception as exc:  # noqa: BLE001 — any search failure must not crash the node
            search_error = str(exc)
            logger.warning("web_search failed for query %r: %s", query_text, exc)
            continue

        raw_material_parts.append(f"QUERY ({query_type}, stage={q.get('stage_name')}): {query_text}")

        for result in results[:MAX_FETCH_PER_QUERY]:
            page = fetch_page.invoke({"url": result["url"]})

            off_domain = (
                query_type == "current_state"
                and not page.get("error") and not page.get("skipped")
                and not _is_company_owned(result["url"], company)
            )

            irrelevant = (
                not page.get("error")
                and not page.get("skipped")
                and not off_domain
                and page.get("text")
                and not is_relevant(page["text"], anchor_terms)
            )

            if page.get("skipped"):
                pages_skipped += 1
                skip_reasons.append(f"{result['url']}: {page.get('skip_reason')}")
            elif off_domain:
                pages_skipped += 1
                skip_reasons.append(f"{result['url']}: not a company-owned source (current-state query)")
            elif irrelevant:
                pages_skipped += 1
                skip_reasons.append(f"{result['url']}: irrelevant (no anchor match)")

            unusable = page.get("error") or page.get("skipped") or off_domain or irrelevant
            excerpt = page["text"][:1500] if not unusable else result.get("content", "")
            raw_material_parts.append(f"SOURCE: {result['url']}\n{excerpt}")

            if not unusable and page.get("text"):
                fetched_pages.append((page, query_type))

    # Prefer filing/regulatory-tier pages for the (capped) structured
    # extraction pass — this agent's whole point is authoritative
    # sources, so when there are more usable pages than budget allows,
    # spend the extraction calls on the highest-tier ones first. Applies
    # across both query types; a high-tier current-state page (e.g. the
    # company's own official app-store listing) is worth prioritizing too.
    fetched_pages.sort(key=lambda item: TIER_RANK.get(source_tier(item[0]["url"]), 0), reverse=True)

    raw_material = "\n\n".join(raw_material_parts) or "No search results were retrieved."
    summary_raw = _summarize_findings(raw_material, funnel)
    parsed = safe_extract_json(summary_raw)
    funnel_verdict = _normalize_funnel_verdict(parsed.get("funnel_verdict"), funnel)

    inserted_findings = 0
    inserted_facts = 0
    rejected = 0
    extraction_errors = 0
    for page, query_type in fetched_pages[:MAX_EVIDENCE_EXTRACTIONS]:
        topic_id = CURRENT_STATE_TOPIC_ID if query_type == "current_state" else TOPIC_ID
        try:
            result = extract_evidence(
                page_text=page["text"],
                source_url=page["url"],
                source_name=page.get("title") or page["url"],
                run_id=run_id,
                hypothesis_id=topic_id,
                hypothesis_statement=problem_statement,
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

    # No extra LLM call for current-state findings -- the extraction that
    # already happened (tagged CURRENT_STATE_TOPIC_ID above) IS the
    # current-state data; just pull it back, already grounded/validated.
    from tools.db import get_facts, get_findings  # local import: same cycle-avoidance pattern as agents/competitive_audit.py

    current_state_findings = _format_current_state_findings(
        get_findings(run_id, CURRENT_STATE_TOPIC_ID) if run_id else [],
        get_facts(run_id, CURRENT_STATE_TOPIC_ID) if run_id else [],
    )

    run_logger.record(
        AGENT_NAME,
        queries=len(queries),
        funnel_queries_by_stage=[{"stage": q.get("stage_name"), "query": q.get("query")} for q in funnel_queries],
        current_state_queries=[q.get("query") for q in current_state_queries],
        pages_fetched=len(fetched_pages),
        pages_skipped=pages_skipped,
        current_state_findings_count=len(current_state_findings),
        funnel_verdict=funnel_verdict,
        evidence_findings_inserted=inserted_findings,
        evidence_facts_inserted=inserted_facts,
        evidence_rejected=rejected,
        extraction_errors=extraction_errors,
        search_error=search_error,
    )

    return {
        "primary_research_findings": findings,
        "current_state_findings": current_state_findings,
        "funnel_verdict": funnel_verdict,
        "messages": [AIMessage(content=summary_raw, name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
