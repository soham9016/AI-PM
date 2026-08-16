"""Evidence extractor — two LLM calls over fetched page text, returning
both evidence tracks (qualitative findings + numeric facts).

CRITICAL: the model is never asked for and never supplies source_url or
source_name — Python attaches those from the page that was actually
fetched, plus run_id/hypothesis_id from state. A model can hallucinate a
URL; it can't hallucinate a variable this module assigned. Everything is
then validated (utils in tools/db.py) before it's allowed into the
database — rejections are logged, not silently dropped.

TWO STAGES, DELIBERATELY SEPARATED — this is a fix for a real observed
failure ("citation laundering"): a single call that received both
page_text AND hypothesis/problem-statement text would, on a thin or
off-topic page, echo sentences from the hypothesis/problem-statement
context back as "findings" from the page. Every provenance guardrail
passed (source_url NOT NULL, source tiering, critic traceability) because
they only check that a URL is real, not that the URL's page actually
contains the claim being attributed to it.

  Stage 1 (_extract_raw): the ONLY call that sees page_text. Its prompt
  contains nothing else — no hypothesis, no problem statement, no prior
  findings — so there is nothing for it to copy from. It cannot judge
  stance (supports/contradicts a hypothesis) because it isn't told what
  hypothesis is under test; it only extracts claims/facts.

  Stage 2 (_classify_relevance_and_stance): given the already-extracted
  (grounded, deduplicated, capped) claim strings plus a context
  statement, both FILTERS for relevance and classifies stance in one
  call. Its output is {"kept": [{"index": i, "stance": ...}]} — index and
  a label only, never new claim text, so it is safe for this call to see
  hypothesis/problem-statement text: there is no field through which it
  could inject fabricated content into the DB.

RELEVANCE FILTERING MOVED HERE DELIBERATELY: isolating stage 1 from all
context (the citation-laundering fix above) also removed the only signal
stage 1 had for what was worth extracting, so it started extracting
every sentence on a page — including page furniture ("Then Covid arrived
and rewired consumer behaviour almost overnight") that's grounded (it's
really on the page) but not relevant to anything. Isolation had to be
paired with a relevance pass that sees context but can't inject text —
same index-only pattern already used for stance, extended to also decide
what's worth keeping at all.

CODE-LEVEL GROUNDING BACKSTOP (between the two stages): even with the
prompt split, a model can still hallucinate content from its own training
data on a thin page. So before a claim/fact is allowed to survive to
stage 2 / insertion, it must be positively located in page_text itself
(see _grounded_in_page / _fact_grounded_in_page) — not merely "doesn't
resemble the problem statement" (a negative check), but "actually appears
in the page" (a positive one). This catches both context-laundering and
hallucination from the model's general knowledge.

A findings-specific length gate also lives here: claims shorter than
MIN_FINDING_CHARS are dropped before insert, since slide-deck/carousel
pages tend to yield headings ("Retention Will Decide the Winner") that
look like claims but assert nothing checkable — a content-shape problem
specific to extraction, not a provenance problem tools/db.py's validators
are meant to catch.

PER-PAGE DEDUPLICATION + CAP: a degenerate page (repetitive boilerplate,
or a model stuck restating the same claim) can otherwise flood the DB
with near-identical rows — one observed case inserted the same claim
eight times verbatim from one page. Exact-normalized-text duplicates are
dropped before insert; MAX_FINDINGS_PER_PAGE caps how many claims survive
per page regardless, which also bounds the token cost of every downstream
consumer of this evidence (e.g. agents/competitive_audit.py's
classification call, which hit Groq's TPM ceiling when a single run's
findings volume ballooned after the isolation fix above).

tools/db.py separately enforces metric/unit coherence AND scale
plausibility on facts (a metric named "percentage of restaurants" stored
as unit "count" is rejected; a crore-labeled value that's also been
crore-multiplied is rejected) — this module's prompt is tuned to avoid
producing those in the first place, but the DB-layer gate is the actual
backstop, same defense-in-depth pattern as everywhere else in this
evidence base.
"""

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from tools.db import ALLOWED_UNITS, insert_fact, insert_finding
from utils.parsing import safe_extract_json
from utils.retry import with_retry
from utils.text_similarity import longest_run_ratio, tokenize

AGENT_NAME = "evidence_extractor"
MAX_PAGE_CHARS = 6000
MIN_FINDING_CHARS = 40
GROUNDING_OVERLAP_THRESHOLD = 0.7
GROUNDING_MIN_RUN_TOKENS = 6
MAX_FINDINGS_PER_PAGE = 8
DEGENERATE_DUPLICATE_THRESHOLD = 0.4
DEGENERATE_MIN_RAW_FINDINGS = 5

logger = logging.getLogger("business_copilot.evidence_extractor")

EXTRACT_SYSTEM_PROMPT = f"""You extract evidence from a single web page's
text. You are given ONLY the page text below — nothing else. Extract
only what this specific page actually says; do not add outside
knowledge.

Return BOTH tracks in one JSON object. Either array may be empty — a
page with no numbers on it is normal, not a failure. Do not force facts
out of a page that doesn't contain any.

Track 1 — findings (qualitative): each claim on the page worth
recording, as a COMPLETE ASSERTION that could be judged true or false —
a full sentence with a subject and a claim, not a slide heading, section
label, or fragment. Some pages (slide decks, LinkedIn carousels) are
full of short headings that look like claims but assert nothing
checkable:
  BAD (headings, not claims — do not extract these as findings):
  "Market Share > Immediate Profitability", "Cost Structures Will Be
  Stress-Tested", "Retention Will Decide the Winner" — none of these say
  what will happen, to whom, by how much, or why; they are slide titles.
  GOOD (a real, checkable claim): "Quick-commerce players are
  prioritizing market share over near-term profitability, accepting
  margin losses to fund expansion." — this asserts something specific
  that evidence could confirm or contradict.
If a claim is too short or vague to actually be checked against
anything, leave it out rather than including it as a finding.

Track 2 — facts (numeric): each concrete number on the page (a metric
with a value), with a unit chosen from this allowed list ONLY:
{sorted(ALLOWED_UNITS)}. Common phrasings are normalized automatically
before storage (e.g. "%"/"pct" -> "percent", "crore"/"cr"/"cr INR"/"INR
cr" -> "inr_cr", "billion USD"/"usd bn" -> "usd_bn", "mn" -> "million",
"bn" (alone, no currency) -> "billion", "min"/"mins" -> "minutes",
"hr"/"hrs" -> "hours", "x"/"times" -> "ratio") — prefer the canonical
spelling above when it's easy, but don't agonize over exact wording for
one of these common cases. If a number's unit genuinely isn't any of
these, omit the fact rather than forcing a wrong unit onto it.

CRITICAL — a fact MUST have a real numeric value. If the page discusses
a metric WITHOUT giving an actual number, that is a finding, not a fact
— do NOT emit a fact with value: null or a number you made up.
  Example: "revenue grew significantly last year" -> no number is given,
  so this is a FINDING only, NOT a fact.
  Example: "revenue grew 14% to $2.1B in FY24" -> TWO facts: one with
  value 14, unit "percent"; one with value 2.1, unit "usd_bn".
  Example: "margins expanded, with an 8.3% CAGR over the period" -> a
  fact with value 8.3, unit "percent" (a percentage is still "percent"
  even when it's a growth rate like CAGR, not just a static ratio).
  Example: "the platform processes 4.45 million orders daily" -> a fact
  with value 4.45, unit "million".
  Example: "average delivery time of 30 minutes" -> a fact with value
  30, unit "minutes".
If you are not looking at an actual number in the text, do not add
anything to the facts array for it.

CRITICAL — when a number is already expressed in a scale unit (crore,
billion, million), record the value IN that scale, not expanded to raw
units. The scale word and the magnitude of value must not both apply:
  BAD: page says "consolidated revenue of Rs 4,718 crore" -> extracting
  {{"metric": "consolidated revenue", "value": 4718000000, "unit": "inr_cr"}}
  is WRONG — the value has been multiplied out to raw rupees AND still
  labeled "inr_cr", double-counting the scale.
  GOOD: {{"metric": "consolidated revenue", "value": 4718, "unit": "inr_cr"}}
  — the number stays exactly as printed on the page; "inr_cr" alone
  already communicates the scale.
  Example: "revenue of Rs 8,625 cr INR, with a loss of Rs 1,092 cr" ->
  two facts: value 8625, unit "inr_cr"; value 1092, unit "inr_cr" — NOT
  8625000000 / 1092000000.

CRITICAL — a plain currency amount (no "crore"/"lakh"/"billion"/"million"
word attached) is a DIFFERENT unit from the scale-denominated ones above
— use "inr" or "usd", not "inr_cr"/"usd_bn", whenever the number itself
carries no scale word. This matters most for small, everyday amounts
(prices, fares, fees), which are never legitimately expressed in crore or
billion:
  BAD: page says "milk costs Rs 65 in Delhi" -> extracting
  {{"metric": "price of milk", "value": 65, "unit": "inr_cr"}} is WRONG —
  there is no "crore" in the source text; this is a plain rupee amount.
  GOOD: {{"metric": "price of milk", "value": 65, "unit": "inr"}}.

CRITICAL — the metric name must describe what THAT NUMBER itself
measures, not a word borrowed from elsewhere in the sentence. A sentence
mentioning a count and a percentage close together is a common source of
mislabeling:
  BAD: page says "500,000 restaurants are now live on the platform,
  giving it a 35% share of all listed restaurants" -> extracting
  {{"metric": "percentage of restaurants", "value": 500000, "unit": "count"}}
  is WRONG. The metric name says "percentage" but the number and unit
  are a plain count — the word "percentage" leaked in from later in the
  sentence even though this particular number isn't one.
  GOOD: the same sentence should produce TWO separate facts:
  {{"metric": "number of restaurants", "value": 500000, "unit": "count"}}
  and {{"metric": "restaurant market share", "value": 35, "unit": "percent"}}
  — each metric name matches its OWN number's unit, not the other one's.
Rows that fail this check are rejected before storage, so getting it
right the first time avoids a wasted extraction.

confidence is your own confidence that this claim/number was accurately
extracted from the page (not a statement about the underlying
business) — it must be exactly one of "high", "medium", "low".

Do NOT include a source_url or source_name field anywhere in your
output — that is attached separately from the page's real URL, not from
your output.

Respond with ONLY a JSON object of this exact shape:
{{
  "findings": [
    {{"claim": "<specific claim from THIS page>", "confidence": "high|medium|low"}}
  ],
  "facts": [
    {{"entity": "<what the number is about>", "metric": "<metric name>", "value": 12.5, "unit": "<one of the allowed units>", "period": "<e.g. 2025-Q4, or null>", "confidence": "high|medium|low"}}
  ]
}}
"""

RELEVANCE_AND_STANCE_SYSTEM_PROMPT = """You are given a context statement
and a numbered list of already-extracted candidate claims from a single
web page. You are NOT extracting new claims — only selecting from this
exact list and labeling. Two jobs:

1. RELEVANCE: keep only claims that actually bear on the context
   statement. Drop generic background, unrelated trivia, or claims about
   something else entirely, even if the claim is a real sentence from
   the page — a claim being true doesn't make it worth keeping if it has
   nothing to do with the context statement.
   BAD to keep: "Then Covid arrived and rewired consumer behaviour
   almost overnight" when the context statement is about a specific
   company's Q3 order growth — real sentence, irrelevant to this context.

2. STANCE: for each claim you KEEP, classify it as:
   "supports"    — is evidence for the context statement
   "contradicts" — is evidence against the context statement
   "context"     — relevant background, doesn't clearly support or contradict

Respond with ONLY a JSON object of this exact shape. Omit any claim
you're dropping as irrelevant — do not list it at all:
{
  "kept": [
    {"index": 0, "stance": "supports|contradicts|context"}
  ]
}
"""


def _grounded_in_page(claim: str, page_text: str) -> bool:
    """A claim is only real evidence if it can actually be located in the
    page it's being attributed to — the positive check, not "doesn't
    resemble the problem statement" (which only catches copying from
    context, not hallucination from the model's own training data)."""
    ratio, run_len = longest_run_ratio(tokenize(claim), tokenize(page_text))
    return ratio >= GROUNDING_OVERLAP_THRESHOLD and run_len >= GROUNDING_MIN_RUN_TOKENS


def _fact_grounded_in_page(value, page_text: str) -> bool:
    """Cheap and exact: the numeric value must appear, in some plausible
    formatting, as a literal substring of the page text. If the page
    doesn't contain "34" anywhere, a fact claiming 34 from that page is
    fabricated — no similarity math needed for numbers."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    candidates = {str(value)}
    if numeric_value == int(numeric_value):
        int_value = int(numeric_value)
        candidates.add(str(int_value))
        candidates.add(f"{int_value:,}")
    else:
        candidates.add(f"{numeric_value:.1f}")
        candidates.add(f"{numeric_value:.2f}")
        candidates.add(f"{numeric_value:,}")

    return any(candidate in page_text for candidate in candidates)


def _normalize_for_dedup(text: str) -> str:
    """Collapse case/punctuation/whitespace differences so "Zomato and
    Swiggy are the top food delivery apps in India." and a re-punctuated
    restatement of the exact same sentence dedup to the same key. Not
    intended to catch differently-WORDED near-duplicates (that's a much
    fuzzier problem) — only exact-content repeats."""
    return " ".join(tokenize(text))


def _laundering_note(claim: str, problem_statement: str) -> str:
    """Diagnostic only — when problem_statement is available, note
    whether a grounding-rejected claim ALSO overlaps it heavily, so the
    log line can distinguish "likely laundered from context" from
    "likely hallucinated outright". Never affects the reject decision."""
    if not problem_statement:
        return ""
    ratio, run_len = longest_run_ratio(tokenize(claim), tokenize(problem_statement))
    if ratio >= GROUNDING_OVERLAP_THRESHOLD and run_len >= GROUNDING_MIN_RUN_TOKENS:
        return " (also matches problem statement — likely laundered from context rather than hallucinated)"
    return ""


@with_retry()
def _extract_raw(page_text: str) -> str:
    llm = get_llm(AGENT_NAME, json_mode=True)
    content = f"Page text:\n{page_text[:MAX_PAGE_CHARS]}"
    response = llm.invoke([SystemMessage(content=EXTRACT_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


@with_retry()
def _classify_relevance_and_stance(claims: list[str], context_statement: str) -> str:
    llm = get_llm(AGENT_NAME, json_mode=True)
    numbered = "\n".join(f"{i}: {claim}" for i, claim in enumerate(claims))
    content = f"Context statement:\n{context_statement}\n\nClaims:\n{numbered}"
    response = llm.invoke([SystemMessage(content=RELEVANCE_AND_STANCE_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def extract_evidence(
    page_text: str,
    source_url: str,
    source_name: str,
    run_id: str,
    hypothesis_id: str,
    hypothesis_statement: str = "",
    problem_statement: str = "",
) -> dict:
    """Extract findings + facts from one fetched page and persist valid rows.

    hypothesis_statement is used ONLY in stage 2 (the combined relevance-
    filter + stance-classification call), never seen by the extraction
    call that produces claim/fact text. problem_statement is optional and
    used ONLY to enrich the rejection log message for grounding failures
    — it plays no role in extraction or in any accept/reject decision.

    Returns a summary dict: {inserted_finding_ids, inserted_fact_ids,
    rejected_count} — inserted_*_ids are the DB row ids that made it in;
    rows that failed validation are logged and counted in rejected_count,
    not raised as errors, since a page failing to yield clean evidence
    isn't fatal to the research pass.
    """
    if not source_url:
        raise ValueError("extract_evidence requires a non-empty source_url — provenance is not optional")

    raw = _extract_raw(page_text)
    parsed = safe_extract_json(raw)

    retrieved_at = datetime.now(timezone.utc).isoformat()

    rejected_count = 0

    # --- findings: length gate, then grounding backstop ---
    raw_finding_count = len(parsed.get("findings", []))
    grounded_claims: list[dict] = []
    for finding in parsed.get("findings", []):
        claim = (finding.get("claim") or "").strip()
        if len(claim) < MIN_FINDING_CHARS:
            rejected_count += 1
            logger.info(
                "Dropping finding too short to be a real assertion (%d chars, need >= %d): %r",
                len(claim), MIN_FINDING_CHARS, claim,
            )
            continue

        if not _grounded_in_page(claim, page_text):
            rejected_count += 1
            note = _laundering_note(claim, problem_statement)
            logger.warning(
                "ungrounded_claim_rejected: claim not locatable in the page it's attributed to (%s)%s: %r",
                source_url, note, claim,
            )
            continue

        grounded_claims.append({"claim": claim, "confidence": finding.get("confidence")})

    # --- per-page dedup on normalized claim text ---
    deduped_claims: list[dict] = []
    seen_normalized: set[str] = set()
    duplicates_dropped = 0
    for finding in grounded_claims:
        key = _normalize_for_dedup(finding["claim"])
        if key in seen_normalized:
            duplicates_dropped += 1
            rejected_count += 1
            logger.info("duplicate_claim_dropped: %s: %r", source_url, finding["claim"])
            continue
        seen_normalized.add(key)
        deduped_claims.append(finding)

    if raw_finding_count >= DEGENERATE_MIN_RAW_FINDINGS:
        duplicate_ratio = duplicates_dropped / raw_finding_count
        if duplicate_ratio >= DEGENERATE_DUPLICATE_THRESHOLD:
            logger.warning(
                "degenerate_extraction_detected: %s produced %d raw findings, %d were exact-normalized "
                "duplicates (ratio %.2f) — possible repetitive page or extraction loop",
                source_url, raw_finding_count, duplicates_dropped, duplicate_ratio,
            )

    # --- cap per page, bounding both this page's DB volume and every
    # downstream consumer's token cost (see module docstring) ---
    if len(deduped_claims) > MAX_FINDINGS_PER_PAGE:
        excess = len(deduped_claims) - MAX_FINDINGS_PER_PAGE
        rejected_count += excess
        logger.info(
            "findings_capped: %s produced %d claims after dedup, keeping first %d (dropping %d)",
            source_url, len(deduped_claims), MAX_FINDINGS_PER_PAGE, excess,
        )
        deduped_claims = deduped_claims[:MAX_FINDINGS_PER_PAGE]

    # --- stage 2: relevance filter + stance, over the bounded candidate set ---
    kept_by_index: dict[int, str] = {}
    if deduped_claims:
        stage2_raw = _classify_relevance_and_stance(
            [c["claim"] for c in deduped_claims], hypothesis_statement
        )
        stage2_parsed = safe_extract_json(stage2_raw)
        for entry in stage2_parsed.get("kept", []):
            idx = entry.get("index")
            stance = entry.get("stance")
            if isinstance(idx, int) and 0 <= idx < len(deduped_claims) and stance in ("supports", "contradicts", "context"):
                kept_by_index[idx] = stance

    inserted_finding_ids: list[int] = []
    for i, finding in enumerate(deduped_claims):
        if i not in kept_by_index:
            rejected_count += 1
            logger.info("relevance_filtered: %s: %r", source_url, finding["claim"])
            continue

        record = {
            "run_id": run_id,
            "hypothesis_id": hypothesis_id,
            "claim": finding["claim"],
            "stance": kept_by_index[i],
            "confidence": finding.get("confidence"),
            "source_url": source_url,
            "source_name": source_name,
            "retrieved_at": retrieved_at,
        }
        new_id = insert_finding(record)
        if new_id is None:
            rejected_count += 1
        else:
            inserted_finding_ids.append(new_id)

    # --- facts: null-value gate, then grounding backstop ---
    inserted_fact_ids: list[int] = []
    for fact in parsed.get("facts", []):
        if fact.get("value") is None:
            rejected_count += 1
            logger.info("Dropping fact with no value (belongs in findings, not facts): %r", fact)
            continue

        if not _fact_grounded_in_page(fact.get("value"), page_text):
            rejected_count += 1
            logger.warning(
                "ungrounded_fact_rejected: value %r not locatable in the page it's attributed to (%s): %r",
                fact.get("value"), source_url, fact,
            )
            continue

        record = {
            "run_id": run_id,
            "hypothesis_id": hypothesis_id,
            "entity": fact.get("entity"),
            "metric": fact.get("metric"),
            "value": fact.get("value"),
            "unit": fact.get("unit"),
            "period": fact.get("period"),
            "confidence": fact.get("confidence"),
            "source_url": source_url,
            "source_name": source_name,
            "retrieved_at": retrieved_at,
        }
        new_id = insert_fact(record)
        if new_id is None:
            rejected_count += 1
        else:
            inserted_fact_ids.append(new_id)

    if rejected_count:
        logger.info(
            "extract_evidence: %s -> %d findings, %d facts inserted, %d rejected",
            source_url, len(inserted_finding_ids), len(inserted_fact_ids), rejected_count,
        )

    return {
        "inserted_finding_ids": inserted_finding_ids,
        "inserted_fact_ids": inserted_fact_ids,
        "rejected_count": rejected_count,
    }
