# Architecture

## What this is

This is an agentic research-and-recommendation system: given a business problem statement, it produces either a diagnosis (why did something happen, backed by cited evidence), a product recommendation (what to build, backed by authoritative sources and competitive context), or both. It runs as a LangGraph state machine — a fixed set of specialist agents, each doing one job, coordinated by a single supervisor that decides what runs next based on what the current state is missing. Every factual claim the system makes is required to trace back to a real, fetched web page or PDF; there is no path by which an agent can assert something without a stored `source_url` behind it.

## The agents

Twelve worker agents do the actual information-processing work; one supervisor (`engagement_manager`) decides what runs next but doesn't produce content itself — it's covered separately in FLOW.md. (The original scaffold had seven workers; `primary_research`, `competitive_audit`, and `solution_framing` were added for the initial PM path; `funnel_decomposition` and `solution_review` were added after that to fix two further failure modes described below.)

| Agent | File | Job | Reads from state | Writes to state |
|---|---|---|---|---|
| structurer | `agents/structurer.py` | Breaks the problem into a MECE issue tree; classifies `problem_type`, `routing_mode`, and search anchor terms | `problem_statement` | `issue_tree`, `problem_type`, `routing_mode`, `anchor_terms` |
| hypothesis | `agents/hypothesis.py` | Writes falsifiable, single-claim hypotheses, one per issue-tree branch, each declaring what evidence would settle it | `problem_statement`, `issue_tree` | `hypotheses` |
| researcher | `agents/researcher.py` | Searches for evidence bearing on each hypothesis's kill conditions; extracts findings/facts into the evidence-base DB | `problem_statement`, `hypotheses`, `run_id` | `research_findings` (a summary; the real output is DB rows) |
| analyst | `agents/analyst.py` | Pulls numeric facts per hypothesis, groups them safely, flags disagreement, charts only what's comparable | `hypotheses`, `run_id` | `analysis_results` |
| synthesizer | `agents/synthesizer.py` | Rules a verdict per hypothesis from DB evidence (weighted by source quality), then writes an answer-first brief | `hypotheses`, `issue_tree`, `run_id` | `synthesis` |
| funnel_decomposition | `agents/funnel_decomposition.py` | The PM-mode analogue of hypothesis generation: decomposes the problem's stated metric into 3-5 funnel stages before any solution is proposed, so solutions target WHERE the drop is happening instead of scattering across the whole funnel; tags each stage's evidence as publicly findable or internal-only | `problem_statement`, `issue_tree` | `funnel` |
| primary_research | `agents/primary_research.py` | Searches for authoritative sources (regulation text, official guidance, company policy) for the ask itself, not a cause; separately builds funnel-stage-targeted and company-domain-scoped current-state queries in code (not left to the LLM) and judges `funnel_verdict` from what's found | `problem_statement`, `issue_tree`, `anchor_terms`, `funnel`, `run_id` | `primary_research_findings`, `current_state_findings`, `funnel_verdict` (summary; real output is DB rows) |
| competitive_audit | `agents/competitive_audit.py` | Checks what named competitors already ship per issue-tree branch; classifies each branch's competitive landscape | `problem_statement`, `issue_tree`, `anchor_terms`, `run_id` | `competitive_audit` |
| solution_framing | `agents/solution_framing.py` | Turns primary research + competitive audit + funnel verdict into candidate solutions, each tied to real evidence and a required/optional flag; guarantees an instrumentation solution whenever `funnel_verdict` is "undetermined" | `problem_statement`, `issue_tree`, `run_id`, `competitive_audit`, `funnel`, `funnel_verdict` | `solutions`, `dropped_solutions` |
| pm | `agents/pm.py` | Writes RICE-scored features from either hypotheses+verdicts or solutions+audit; labels each feature's evidence strength in code | `problem_statement`, `issue_tree`, `hypotheses`, `synthesis`, `solutions`, `competitive_audit` | `prd` |
| solution_review | `agents/solution_review.py` | Retroactively kills weak PM-path solutions (and their PRD feature) against the evidence actually gathered; recomputes MoSCoW/header over the surviving feature set, since both are relative to whatever feature set exists at the time — inheriting `pm`'s pre-kill values would describe a set that no longer exists | `solutions`, `prd` | `solutions`, `dropped_solutions`, `prd`, `solutions_reviewed` |
| critic | `agents/critic.py` | Runs deterministic checks appropriate to the mode that ran; filters them for materiality | `synthesis`, `hypotheses`, `issue_tree`, `solutions`, `prd`, `routing_mode` | `critique`, `critic_passed` |

Every agent also reads `state['revision_notes'][<its own name>]` (feedback from a prior critic pass, if this is a re-run) and writes `messages` + `run_path`, omitted from the table above since it's uniform across all twelve.

## The three routing modes

A single pipeline shape can't serve both "why did this happen" and "what should we build" — the first requires generating and testing causal hypotheses against evidence; the second either already knows its premise or is asking a non-causal question no hypothesis-testing machinery would resolve. `structurer` classifies which shape applies before any research happens, and the supervisor runs a genuinely different sequence of agents depending on the answer — not the same agents with different prompts.

| Mode | Agents that run | Agents that don't | Why the shape differs |
|---|---|---|---|
| **diagnostic** | structurer → hypothesis → researcher → analyst → synthesizer → (pm, if `problem_type` is product-flavored) → critic | funnel_decomposition, primary_research, competitive_audit, solution_framing, solution_review | A root cause must be found and verified. Every step exists to generate a testable causal claim and rule on it against evidence. |
| **pm** | structurer → funnel_decomposition → primary_research → competitive_audit → solution_framing → pm → solution_review → critic | hypothesis, researcher, analyst, synthesizer | The cause is either a stated premise (not something to verify) or the question isn't causal at all ("what should we build for DPDP compliance?"). Running the diagnostic chain here invents diagnostic questions nobody asked, can't find public evidence for them, and correctly reports inconclusive — the actual failure mode this mode exists to avoid. |
| **combined** | Every agent from both of the above, diagnostic steps first, then PM steps, then pm, then critic | — | The problem genuinely needs both a verified root cause and a set of recommended measures. Roughly double the LLM call volume of either mode alone — see the "wrong/fragile" notes at the end for why that's worth watching. |

## The evidence base

Two SQLite tables, defined in `tools/db.py`, are the durable record of everything the system found. Everything else — `research_findings`, `primary_research_findings`, `analysis_results` — is a scratch summary or an audit trail, not the data agents actually reason over (see the "used in a way its name doesn't suggest" note at the end).

| Table | Holds | Key columns |
|---|---|---|
| `findings` | Qualitative claims | `claim`, `stance` (supports/contradicts/context), `source_url`, `source_name`, `confidence` |
| `facts` | Numeric claims | `entity`, `metric`, `value`, `unit`, `period`, `source_url`, `source_name`, `confidence` |

Both tables share `run_id` (which graph run produced this row) and `hypothesis_id` (see below for what this actually means). Facts are never deduplicated — two sources reporting different numbers for the same metric is signal, not noise.

**The provenance rule**: no row can exist without a real `source_url`. This is enforced three times, deliberately redundant:
- **Prompt**: `tools/evidence_extractor.py`'s system prompt tells the model never to supply `source_url`/`source_name` at all.
- **Code**: Python attaches `source_url`/`source_name` itself, from the page it actually fetched — the model's output is never trusted for this field even if it tried to supply one.
- **DB constraint**: `source_url TEXT NOT NULL` on both tables. Even if the code path above were buggy, the database itself refuses a null.

**`hypothesis_id`** is not a foreign key to a real hypotheses table (hypotheses only ever live in graph state, never in this DB) — it's a plain grouping key. In diagnostic mode it holds a real hypothesis id (`H1`, `H2`...). In PM mode, where there's no hypothesis, `primary_research` and `competitive_audit` reuse the same column as a topic label (`"primary"`, `"competitor:zomato"`). `get_findings`/`get_facts` query by that key; `get_findings_for_run`/`get_facts_for_run` ignore it and pull everything for a run, which is what `solution_framing` needs. This reuse avoided a schema migration but is a real design compromise — see the "wrong/fragile" section.

**Source tiering**: `utils/source_tier.py` classifies every piece of evidence by domain pattern into `filing` > `analyst` > `news` > `blog` > `social`, computed fresh from `source_url` every time it's needed rather than stored — it's a pure function of the URL, so there's nothing to migrate when the domain lists are extended. An unrecognized domain defaults to `blog`, not `news`: absence of a bad signal isn't presence of a good one.

| Where tiering is enforced | Mechanism |
|---|---|
| Prompt | `synthesizer`'s verdict prompt is told the tier of every piece of evidence and instructed to weigh accordingly (e.g. prefer `inconclusive` over `supported` on social-tier-only evidence) |
| Code (backstop) | `synthesizer._sanity_check` downgrades any `supported`/`killed` verdict backed only by social-tier evidence to `inconclusive`, regardless of what the LLM ruled |

## Where LLMs are used, and where they deliberately aren't

| Used for | Not used for | Why |
|---|---|---|
| Classification (`problem_type`, `routing_mode`, `evidence_type`) | Validating that classification against the allowed enum | An LLM can misclassify; code normalizes/validates every classification against a fixed set with a safe default, logged when it falls back |
| Generating hypotheses, solutions, features, search queries | Deciding which agent runs next | Routing is pure Python in `engagement_manager.py` — a routing mistake would derail the whole run, and it's a small enough decision space that code is both cheaper and more reliable than a model call |
| Extracting claims/numbers from page text | Validating the extracted unit, value plausibility, or metric/unit coherence | `tools/db.py` rejects nonsense (a metric named "percentage of X" stored as a count) in code — the prompt is tuned to avoid producing it, but the gate doesn't trust that tuning |
| Ruling a verdict on a hypothesis from evidence | Deciding whether that verdict is *internally consistent* (e.g. "supported" with zero cited evidence) | `synthesizer._sanity_check` and `critic`'s deterministic checks re-derive consistency in code rather than trusting the model's self-report |
| Labeling a feature's evidence-strength narrative | Setting the actual `evidence_strength` value | `pm.py` computes this from the real verdict/solution data — the model is explicitly told not to self-assess it |
| Filtering critic candidates for materiality | Generating the candidate list itself | The four-to-six critic checks are deterministic (DB/state lookups); only the "is this worth surfacing" judgment is delegated to a model, and even then it can only drop candidates, never invent new ones |

The pattern throughout: generation and judgment-under-ambiguity go to the model; anything where a wrong answer would silently corrupt provenance, evidence-linkage, or control flow is deterministic code, usually placed *after* the model call as a backstop rather than instead of prompting for it — both layers exist together (defense in depth), not one or the other.

## Every guardrail, and what it prevents

**Provenance & storage** (`tools/db.py`, `tools/evidence_extractor.py`)

| Guardrail | Prevents |
|---|---|
| `source_url NOT NULL` (DB constraint) | Any claim entering storage without a real source |
| `stance`/`confidence` CHECK constraints (DB) | Invalid categorical values being stored raw |
| Unit normalization + `ALLOWED_UNITS` check (code) | Real evidence being rejected over phrasing ("cr INR" vs "crore"), while still rejecting genuinely unsupported units |
| Metric/unit coherence check (code) | A metric named "percentage of X" being stored with unit "count" (or any other name/unit contradiction) even though every individual field looks valid alone |
| `MIN_FINDING_CHARS` gate (code) | Slide headings/fragments ("Retention Will Decide the Winner") being stored as if they were checkable claims |
| Null-value fact drop (code) | A metric mentioned without a real number being stored as a fact with an invented or missing value |

**Evidence quality & relevance** (`utils/relevance.py`, `tools/fetch.py`)

| Guardrail | Prevents |
|---|---|
| Sector/company anchor relevance gate | Off-topic pages (matched by a generic query) from becoming "evidence" for the wrong company/industry |
| PDF/HTML garbage gate, charset handling, binary-extension pre-filter | Mangled or non-text content being fed to the extractor as if it were readable prose |

**Verdict & synthesis integrity** (`agents/synthesizer.py`)

| Guardrail | Prevents |
|---|---|
| Source-tier-weighted verdict prompt + code downgrade | A single social-tier opinion post carrying a "supported" verdict |
| Zero-evidence verdict downgrade | A "supported"/"killed" verdict with no cited evidence_ids |
| `answer`/`caveats` field separation | Disclosure of unexamined branches swallowing the entire actionable recommendation |
| `unexamined_branches` computed in code | The write-up silently implying full issue-tree coverage when hypotheses were capped |

**Critic** (`agents/critic.py`)

| Guardrail | Prevents |
|---|---|
| Deterministic candidate checks (traceability, verdict-evidence consistency, overreach, MECE gaps, solution traceability, regulation ranking) | Hallucinated evidence_ids, self-contradictory verdicts, inconclusive hypotheses presented as settled, silently uncovered issue-tree branches, evidence-free PM solutions, and regulation-mandated features ranked below optional ones by RICE |
| Fail-closed on materiality-filter parse failure | A JSON parse hiccup silently turning real flags into a clean pass |
| Materiality reconstruction from the original candidate, not the LLM's restated copy | A drifted `suggested_action` causing a flag to fall through to the wrong (or default) loop-back target |

**Chart integrity** (`agents/analyst.py`)

| Guardrail | Prevents |
|---|---|
| Never chart a `conflicting`-flagged group | Sources that disagree by an order of magnitude being drawn as one smooth trend |
| Entity partitioning | Different companies/scopes sharing a chart just because they share a metric+unit |
| Fiscal-period normalization + chronological sort | The same underlying period being plotted twice under different labels, out of time order |
| Minimum 3 comparable points | A 2-bar chart being presented as if it showed a trend |

**PM-path evidence discipline** (`agents/solution_framing.py`, `agents/pm.py`)

| Guardrail | Prevents |
|---|---|
| `finding_id` validation against `evidence_exists()` | A hallucinated finding_id making an exploratory solution look evidence-backed |
| `evidence_strength` set in code from real verdict/solution data | The PM agent grading its own recommendation's confidence |
| `reach_basis` required per feature | An invented reach number looking like a sourced estimate |
| Competitive impact capping in code | An LLM RICE score treating "we now match competitors" as "massive impact" |
| Guardrail-metric coverage check | A PRD shipping with no metric watched for regression, silently |

**Control flow** (`agents/engagement_manager.py`, `state.py`)

| Guardrail | Prevents |
|---|---|
| `problem_type`/`routing_mode` normalization + validated enum | An off-vocabulary classification silently breaking downstream mode dispatch |
| "Ran but empty" non-empty defaults (issue_tree, hypotheses, findings, solutions, ...) | "Hasn't run yet" and "ran, found nothing" looking identical to the ladder, which would otherwise loop on the same agent forever |
| `iteration_count` cap (`MAX_ITER`) | Infinite loop-backs — forces a disclosed-limitations finish instead |
| `critic_passed` reset on every loop-back | A stale pass-through short-circuiting the very re-review a loop-back exists to trigger |
| `DOWNSTREAM` invalidation on loop-back | Stale downstream state masking that rework didn't actually happen |

**Resilience** (`utils/parsing.py`, `utils/retry.py`, `config.py`)

| Guardrail | Prevents |
|---|---|
| Groq JSON mode (critic, extractor) | Malformed JSON being structurally possible in the first place |
| `safe_extract_json` (used almost everywhere) | A single bad LLM response crashing the entire run instead of degrading to "produced nothing" |
| Retry/backoff on rate-limit errors (`utils/retry.py`, covers both the Groq and OpenAI-compatible exception hierarchies) | A transient rate limit (Groq or Cerebras) failing a node outright |
| `config.invoke_llm`'s invocation-time provider fallback | A non-Groq provider's non-rate-limit failure (bad model name, quota/payment error) killing the run — logs clearly and retries once on that entry's Groq fallback model instead of propagating |
| `MAX_HYPOTHESES` / `MAX_EVIDENCE_EXTRACTIONS` / per-agent query caps | Unbounded LLM call volume on a free-tier budget |
