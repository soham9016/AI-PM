# Flow

This is a companion to ARCHITECTURE.md: that file describes what each piece is, this one describes how a request actually moves through them.

## Hub-and-spoke, and why routing lives in exactly one place

Every worker agent's LangGraph edge points to the same node: `engagement_manager`. No agent ever calls another agent directly, and no agent knows what runs after it — `researcher` has no idea `analyst` exists. The only node that knows the shape of a run is `engagement_manager`, and the only place that decision gets translated into an actual graph destination is `routing.py`:

```python
def route_from_manager(state):
    nxt = state["next_agent"]
    return END if nxt == "FINISH" else nxt
```

That function hasn't changed since the very first version of this system, even after three new agents were added for the PM path — because it doesn't know or care how many agents exist. It just reads `next_agent` and either ends the run or routes to whatever name `engagement_manager` wrote there.

This buys one concrete thing: adding `primary_research`, `competitive_audit`, `solution_framing` (and later `funnel_decomposition`, `solution_review`) required changes in exactly two files each time — `agents/engagement_manager.py` (teach it the new ladder position) and `graph.py` (register the node in the `WORKERS` dict, which generates its edge back to the hub automatically). None of the other agents, and nothing in `routing.py`, needed to be touched, five agents and two additions later. That's the actual payoff of a hub-and-spoke shape over a hand-wired chain: the cost of adding an agent doesn't grow with how many agents already exist.

## Walkthrough: one full diagnostic run

Problem statement: *"Why did Swiggy's Q3 order growth slow from 22% to 9% quarter-over-quarter?"*

| Hop | Node | File | What happens | State change | What EM checked to send it here |
|---|---|---|---|---|---|
| 0 | — | `state.py` | `new_state()` builds the initial state: empty issue_tree, empty hypotheses, `routing_mode=""`, fresh `run_id` | initial state | — |
| 1 | engagement_manager | `agents/engagement_manager.py` | First tick. `issue_tree` is empty, so this is the shared entry step for every mode | `next_agent="structurer"` | Rule: no issue tree yet → always `structurer`, regardless of mode, because `routing_mode` itself doesn't exist until structurer sets it |
| 2 | structurer | `agents/structurer.py` | One LLM call breaks the problem into branches (demand-side, supply-side, pricing...), classifies `problem_type="strategy"`, `routing_mode="diagnostic"`, and anchor terms (`company="Swiggy"`, `sector="quick commerce"`) | `issue_tree`, `problem_type`, `routing_mode`, `anchor_terms` set | — |
| 3 | engagement_manager | same | `routing_mode="diagnostic"` selects `_DIAGNOSTIC_STEPS`. First unfilled field in that ladder is `hypotheses` | `next_agent="hypothesis"` | Walks the diagnostic ladder top to bottom, stops at the first `(state_key, next_agent)` pair whose `state_key` is still falsy |
| 4 | hypothesis | `agents/hypothesis.py` | One LLM call, one hypothesis per branch, each with a `kill_conditions`/`evidence_needed`/`evidence_type` | `hypotheses` set | — |
| 5 | engagement_manager | same | `hypotheses` now truthy, next unfilled ladder field is `research_findings` | `next_agent="researcher"` | Ladder walk |
| 6 | researcher | `agents/researcher.py` | Plans queries per hypothesis, searches (Tavily), fetches pages, relevance-gates them against the anchor terms, extracts findings/facts into the DB tagged with the real `hypothesis_id` | `research_findings` (scratch summary); real output is new rows in `findings`/`facts` scoped to this `run_id` | — |
| 7 | engagement_manager | same | Next unfilled field: `analysis_results` | `next_agent="analyst"` | Ladder walk |
| 8 | analyst | `agents/analyst.py` | Pulls `facts` for this run+hypothesis from the DB, groups by (metric, unit), flags groups whose spread exceeds the conflict threshold, charts only the non-conflicting groups with ≥3 comparable points | `analysis_results` set | — |
| 9 | engagement_manager | same | Next unfilled field: `synthesis` | `next_agent="synthesizer"` | Ladder walk |
| 10 | synthesizer | `agents/synthesizer.py` | Per hypothesis, pulls findings/facts by `hypothesis_id`, tags each by source tier, LLM rules a verdict, code sanity-checks it (downgrades a zero-evidence or all-social-tier "supported"/"killed" to `inconclusive`), then a second LLM call writes the answer/caveats split | `synthesis` set | — |
| 11 | engagement_manager | same | Diagnostic ladder fully walked (issue_tree, hypotheses, research_findings, analysis_results, synthesis all truthy). PM gate: `problem_type` is `"strategy"`, not `"product"`, and `routing_mode` isn't `pm`/`combined` → skip `pm` | `next_agent="critic"` | Step 6 of the node: ladder exhausted → PM gate false → critic |
| 12 | critic | `agents/critic.py` | `routing_mode="diagnostic"` runs the four diagnostic checks against `synthesis`/`hypotheses`/`issue_tree`. Say it finds nothing material this time | `critique=[]`, `critic_passed=True` | — |
| 13 | engagement_manager | same | `critic_passed` is `True` and `critique` is empty | `next_agent="FINISH"` | Branch 3: critic passed → done. `_assemble` builds `final_brief` with the diagnostic fields (`verdicts`, `recommendation`, `caveats`) since `routing_mode` is diagnostic, and omits `solutions`/`competitive_audit` |
| 14 | — | `routing.py` | `route_from_manager` sees `"FINISH"` → returns `END` | graph run ends | — |

## Walkthrough: one full PM-mode run

Problem statement: *"What should we build to comply with the DPDP Act's consent requirements?"*

| Hop | Node | What happens | State change | What EM checked |
|---|---|---|---|---|
| 1 | engagement_manager | Empty issue tree | `next_agent="structurer"` | Shared entry rule |
| 2 | structurer | Classifies `problem_type="product"`, `routing_mode="pm"` (the ask isn't "why," it's "what," and there's no causal question to test), anchor terms | `issue_tree`, `problem_type`, `routing_mode`, `anchor_terms` set | — |
| 3 | engagement_manager | `routing_mode="pm"` selects `_PM_STEPS`, not `_DIAGNOSTIC_STEPS`. First unfilled field: `funnel` | `next_agent="funnel_decomposition"` | Ladder walk — note `hypothesis`/`researcher`/`analyst`/`synthesizer` are never even considered, they aren't in this ladder at all |
| 4 | funnel_decomposition | Decomposes the stated metric ("% of users who complete consent") into 3-5 ordered stages, each tagged `evidence_locatability: "public"` or `"internal"` | `funnel` set, `funnel_verdict` reset to `""` | — |
| 5 | engagement_manager | Next unfilled field: `primary_research_findings` | `next_agent="primary_research"` | Ladder walk |
| 6 | primary_research | Plans queries toward regulation text and official guidance (not toward proving/disproving a hypothesis, since there isn't one); separately builds one query per PUBLIC-locatability funnel stage directly from that stage's own `evidence_needed` text, and company-domain-scoped current-state queries, both in code — not left to the LLM; fetches, sorts by source tier so `.gov` pages get extraction budget first, extracts evidence into the DB tagged `hypothesis_id="primary"`/`"current_state"`, judges `funnel_verdict` from what was found | `primary_research_findings`, `current_state_findings`, `funnel_verdict` set (scratch summaries; real output is new DB rows) | — |
| 7 | engagement_manager | Next unfilled field: `competitive_audit` | `next_agent="competitive_audit"` | Ladder walk |
| 8 | competitive_audit | Identifies up to 3 named competitors, plans queries per competitor per issue-tree branch, fetches/extracts evidence tagged `hypothesis_id="competitor:<name>"`, then a second LLM call classifies each branch as `table_stakes`/`differentiator`/`parity_gap` per competitor, validated against a fixed allowed set | `competitive_audit={competitors, areas}` set | — |
| 9 | engagement_manager | Next unfilled field: `solutions` | `next_agent="solution_framing"` | Ladder walk |
| 10 | solution_framing | Pulls the *entire* evidence pool for this run (`get_findings_for_run`/`get_facts_for_run` — not scoped to one topic, since a solution can draw on both primary research and competitive findings), LLM proposes solutions each tagged `required` (regulation-mandated) or not, with `finding_ids` and (if a funnel exists) `addresses_funnel_stage`; code strips any `finding_id` that doesn't actually exist in the DB, and guarantees an `is_instrumentation` solution covering every stage whenever `funnel_verdict=="undetermined"` and the LLM didn't propose one itself | `solutions`, `dropped_solutions` set | — |
| 11 | engagement_manager | PM ladder fully walked. PM gate: `routing_mode="pm"` and `prd` is still `None` | `next_agent="pm"` | Step 6: ladder exhausted, PM gate true (this is the same gate the diagnostic walkthrough hit and skipped — here it fires) |
| 12 | pm | LLM proposes features, each declaring which solution it addresses; code computes `evidence_strength` from the real solution data (`required=True` → `regulation_backed`, non-empty `finding_ids` → `evidence_backed`, `is_instrumentation` → `instrumentation`, else `exploratory`), applies the competitive-audit impact cap, recomputes RICE scores, assigns MoSCoW, builds `header`/`north_star_metric` | `prd` set | — |
| 13 | engagement_manager | Next unfilled field: `solutions_reviewed` | `next_agent="solution_review"` | Ladder walk |
| 14 | solution_review | LLM judges each candidate solution's evidence against the north-star metric, killing weak ones (removed from `solutions` and their PRD feature, moved to `dropped_solutions`); MoSCoW and `header` are then RECOMPUTED over the surviving feature set, not inherited from `pm`'s pre-kill pass — both are relative to whichever feature set exists at computation time | `solutions`, `dropped_solutions`, `prd` updated, `solutions_reviewed=True` | — |
| 15 | engagement_manager | `solutions_reviewed` is now `True` | `next_agent="critic"` | — |
| 16 | critic | `routing_mode="pm"` runs the two PM checks (solution traceability, regulation ranking) instead of the four diagnostic ones — `synthesis` is empty here anyway, so the diagnostic checks wouldn't have anything to check | `critique`, `critic_passed` set | — |
| 17 | engagement_manager | Passed | `next_agent="FINISH"` | `_assemble` includes `solutions`/`competitive_audit`, omits `verdicts`/`recommendation`/`caveats` |

**Combined mode** is the concatenation of both ladders (`_DIAGNOSTIC_STEPS + _PM_STEPS`) walked in order, still ending at the same PM gate and critic step — every hop above happens, in the same relative order, just without a `FINISH` in between.

## The loop-back path

**Trigger**: `critic` finds one or more flags it judges material (survived its LLM materiality filter). `critique` comes back non-empty.

**What EM does with it** (this check runs *before* the ladder-walk logic, so a loop-back always pre-empts forward progress):
1. Picks a target agent via `_target_for(flags)` — the earliest agent in a fixed `ORDER` list that any flag's `suggested_action` names; defaults to `synthesizer` if none is recognized.
2. Increments `iteration_count`.
3. Writes `revision_notes` — a `{agent_name: [issue text, ...]}` dict — so the target agent's next run gets the specific critique, not just a generic "try again."
4. Resets every field in `DOWNSTREAM[target]` back to its `EMPTY` value.

**What gets reset, and why**: `DOWNSTREAM` is keyed by the loop-back target and lists every field that would be stale or misleading once that agent reruns. The rule is "everything downstream of the target, never the target's own output field" — looping back to `synthesizer` clears `prd`/`critique`/`critic_passed` (what comes *after* synthesis) but leaves `synthesis` itself alone, since synthesizer is about to overwrite it anyway. Looping back to `structurer` clears everything in both tracks, because `structurer` is what decides `routing_mode` in the first place — a revised issue tree could change which ladder even applies.

`critic_passed` is included in *every* `DOWNSTREAM` list, deliberately. This was a real bug until recently: `critic_passed` was reset to `False` in the `EMPTY` defaults dict but never appeared in any `DOWNSTREAM` list, so a loop-back would invalidate `prd`/`critique` but leave a stale `critic_passed=True` sitting in state from the *previous* critic pass. On the very next tick, EM's branch order checks "flags present → route to target" first, but that only fires once; on the tick *after* the target agent actually reruns, EM would see `critic_passed=True` left over from before and finish immediately — reporting success on a run that had just been told to fix something, before the revised output was ever re-reviewed. Adding `critic_passed` to every `DOWNSTREAM` list closes that: after a loop-back, the next critic run is guaranteed to be a genuine re-review, not a stale pass-through.

**What stops it looping forever**: `MAX_ITER = 3`. If `iteration_count` has already reached the cap and `critique` is *still* non-empty, EM force-finishes anyway — `final_brief` still includes the outstanding critique under `limitations`, so the run discloses what it couldn't resolve rather than looping indefinitely or silently hiding the gap.

## Where each guardrail sits in the flow

| Flow stage | Guardrails active there |
|---|---|
| Page fetch (researcher / primary_research / competitive_audit) | Relevance gate, PDF/HTML garbage gate, charset handling |
| Extraction into DB (`tools/evidence_extractor.py`) | `source_url` provenance (prompt + code + DB constraint), `MIN_FINDING_CHARS`, null-value fact drop, unit/coherence checks |
| Analyst | Conflict detection, entity/period-safe charting, minimum-points gate |
| Synthesizer | Source-tier weighting, zero-evidence/social-only verdict downgrade, answer/caveats separation, `unexamined_branches` |
| Funnel decomposition | Public-locatability stages code-filtered before the query planner ever sees them (`_public_only_funnel`) |
| Primary research | Funnel-stage and current-state queries built in code from `evidence_needed`/company anchor, not left to the LLM; `_is_company_owned` domain filter on every current-state result |
| Solution framing | `finding_id` validation against `evidence_exists()`; `instrumentation_plan.stage` forced to cover every funnel stage when `funnel_verdict=="undetermined"` |
| PM | Code-computed `evidence_strength`, competitive impact cap, `reach_basis` requirement, guardrail-metric coverage check |
| Solution review | MoSCoW/header recomputed over the post-kill feature set, not inherited from `pm`'s pre-kill pass |
| Critic | Deterministic mode-gated checks, fail-closed materiality filter, candidate reconstruction (not LLM's restated copy) |
| Engagement manager | Enum normalization (upstream, in structurer, but enforced by EM trusting only validated values), ran-but-empty non-empty defaults, `iteration_count` cap, `critic_passed`/`DOWNSTREAM` reset pairing |
| Every LLM call | `safe_extract_json` parse-failure containment, retry/backoff, JSON mode where used |
