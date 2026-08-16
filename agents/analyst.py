"""Analyst agent — queries ONLY the facts table, scoped to the current
run_id, and only ever aggregates numbers that share the same metric and
unit (mismatched units are reported separately, never combined).

For each hypothesis: declare what's being looked for, check whether any
matching facts exist for this run, and either analyze them or report
"insufficient data" — it never runs statistics over unrelated rows just
because a table happens to have data in it. Every computed figure keeps
the source fact ids it was derived from, so a downstream reader (or the
critic) can trace a number back to where it came from.

No dedup, ever — two sources reporting different numbers for the same
metric is signal, not noise. But a wide spread (three sources citing
CAGR as 71%, 18%, and 5% for the same market) shouldn't be silently
averaged or resolved to one value either — CONFLICT_RELATIVE_SPREAD_THRESHOLD
marks a (metric, unit) group as "conflicting" when its spread is large
relative to its mean, and both the summarizer prompt and a code-generated
message line are told to surface that disagreement explicitly rather
than pick a winner.

CHART SAFETY: (metric, unit) grouping is the right granularity for the
narrative summary, but it's too loose for a chart — a group can span
multiple entities (different companies, different scopes) whose numbers
were never meant to be compared, and periods get labelled inconsistently
across sources ("Q3FY23" vs "3QFY24", "FY24" vs "2024"), which can put
one source's figure between two other sources' figures on the x-axis as
if it were a later data point in the same trend when it's really an
unrelated one. _chart_for_group therefore, in order: (1) never charts a
group already flagged conflicting — that alone catches the worst case,
a group whose values differ by orders of magnitude; (2) partitions by
entity, since different entities sharing (metric, unit) is normal and
NOT a conflict, just not chartable together; (3) normalizes fiscal
period labels to one canonical form per entity-partition so the same
underlying period isn't plotted twice under different names; (4) drops
any partition left with fewer than MIN_CHART_POINTS comparable points —
a 2-bar chart isn't a trend.
"""

import re
from collections import defaultdict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_llm
from tools.chart import generate_chart
from tools.db import get_facts
from utils.logger import run_logger
from utils.parsing import safe_extract_json
from utils.retry import with_retry

AGENT_NAME = "analyst"
CONFLICT_RELATIVE_SPREAD_THRESHOLD = 1.0
MIN_CHART_POINTS = 3

_QUARTER_RE = re.compile(r"^Q([1-4])FY(\d{2,4})$")
_QUARTER_ALT_RE = re.compile(r"^([1-4])QFY(\d{2,4})$")
_FY_RE = re.compile(r"^FY(\d{2,4})$")
_YEAR_RE = re.compile(r"^(\d{4})$")

DECLARE_SYSTEM_PROMPT = """You are a data analyst about to test a
hypothesis against numeric evidence. Before looking at any data, state
in one sentence what kind of metric(s)/comparison would actually test
this hypothesis. This is a declaration of intent, not a query — it will
be logged so a reader can see what you were looking for before you saw
the results, which is what keeps the analysis honest.

Respond with ONLY a JSON object of this exact shape:
{"data_needed": "<1 sentence describing the metric/comparison you're looking for>"}
"""

SUMMARY_SYSTEM_PROMPT = """You are a data analyst. Given grouped numeric
facts (already aggregated within matching metric+unit groups — never
combined across units) for one hypothesis, describe what the numbers
show and whether they support or undermine the hypothesis. If groups use
incompatible units, say so explicitly instead of picking one.

Some groups are marked "conflicting": true — this means sources sharing
the same metric and unit disagree by a wide margin (e.g. three sources
citing CAGR as 71%, 18%, and 5% for the same market). Do NOT average
these into one number and do NOT silently pick the value that best
supports the hypothesis — call out the disagreement explicitly in your
summary (e.g. "sources disagree sharply on X, ranging from A to B") and
treat that disagreement itself as a relevant finding, not noise to
resolve.

Respond with ONLY a JSON object of this exact shape:
{
  "summary": "<1-3 sentence finding, grounded only in the numbers given>",
  "supports_hypothesis": true
}
"""


@with_retry()
def _declare_data_needed(hypothesis: dict, revision_note: str | None) -> str:
    llm = get_llm(AGENT_NAME)
    content = f"Hypothesis:\n{hypothesis}"
    if revision_note:
        content += f"\n\nA previous review flagged this issue with your prior analysis — address it:\n{revision_note}"
    response = llm.invoke([SystemMessage(content=DECLARE_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


@with_retry()
def _summarize_groups(hypothesis: dict, groups: list[dict]) -> str:
    llm = get_llm(AGENT_NAME)
    content = f"Hypothesis:\n{hypothesis}\n\nGrouped facts (metric, unit, aggregate, fact_ids):\n{groups}"
    response = llm.invoke([SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=content)])
    return response.content


def _group_facts(facts: list[dict]) -> list[dict]:
    """Group facts by (metric, unit) so aggregation never mixes units.

    Returns a list of {metric, unit, count, sum, avg, min, max, fact_ids,
    entities, periods, conflicting}. `conflicting` is True when the
    group's spread (max - min) exceeds CONFLICT_RELATIVE_SPREAD_THRESHOLD
    times the group's mean — sources disagreeing sharply on the same
    (metric, unit) is itself a finding, not something to average away.
    """
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for fact in facts:
        key = (fact.get("metric"), fact.get("unit"))
        buckets[key].append(fact)

    groups = []
    for (metric, unit), rows in buckets.items():
        values = [row["value"] for row in rows]
        avg = sum(values) / len(values)
        low, high = min(values), max(values)
        conflicting = len(values) > 1 and avg != 0 and (high - low) / abs(avg) > CONFLICT_RELATIVE_SPREAD_THRESHOLD
        groups.append(
            {
                "metric": metric,
                "unit": unit,
                "count": len(values),
                "sum": sum(values),
                "avg": avg,
                "min": low,
                "max": high,
                "conflicting": conflicting,
                "fact_ids": [row["id"] for row in rows],
                "entities": sorted({row.get("entity") for row in rows if row.get("entity")}),
                "periods": sorted({row.get("period") for row in rows if row.get("period")}),
            }
        )
    return groups


def _normalize_period(period) -> str | None:
    """Canonicalize a fiscal-period label so "Q1FY24" / "1QFY24" and
    "FY24" / "2024" are treated as the same period rather than distinct
    points on a chart's x-axis. This is an approximation — a company's
    fiscal year doesn't always equal the calendar year — but the
    alternative (a bare 4-digit year sitting unordered next to
    differently-formatted fiscal labels) is what produced a chart with a
    "FY24" data point wedged between "2024" and "2029" as if it were its
    own later period.
    """
    if not period or not isinstance(period, str):
        return None
    p = period.strip().upper().replace(" ", "").replace("-", "")
    m = _QUARTER_RE.match(p) or _QUARTER_ALT_RE.match(p)
    if m:
        return f"Q{m.group(1)}FY{m.group(2)[-2:]}"
    m = _FY_RE.match(p)
    if m:
        return f"FY{m.group(1)[-2:]}"
    m = _YEAR_RE.match(p)
    if m:
        return f"FY{m.group(1)[-2:]}"
    return p


def _period_sort_key(period: str) -> tuple[int, int]:
    """Chronological sort key for normalized period labels, so a chart's
    x-axis reads left-to-right in time order rather than DB-insertion
    order (which is what put "FY24" between "2024" and "2029" visually,
    on top of it being a mislabeled duplicate of "2024" in the first
    place)."""
    m = re.match(r"^Q(\d)FY(\d{2})$", period)
    if m:
        return (int(m.group(2)), int(m.group(1)))
    m = re.match(r"^FY(\d{2})$", period)
    if m:
        return (int(m.group(1)), 0)
    return (999, 0)


def _chart_for_group(hypothesis_id: str, group: dict, facts_by_id: dict[int, dict]) -> list[str]:
    """Produce zero or more chart paths for one (metric, unit) group —
    zero or more because a group spanning several entities now yields at
    most one chart PER entity, not one chart mixing all of them. See the
    module docstring's CHART SAFETY section for the full rationale.
    """
    if group.get("conflicting"):
        return []

    rows = [facts_by_id[fid] for fid in group["fact_ids"]]

    by_entity: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_entity[row.get("entity") or ""].append(row)

    chart_paths = []
    for entity, entity_rows in by_entity.items():
        if not entity:
            continue

        if any(r.get("period") for r in entity_rows):
            x_key = "period"
            # First-wins dedup: two facts normalizing to the same period
            # (e.g. one labelled "FY24", one labelled "2024") can't both
            # occupy the same x-axis position — this only affects what
            # the CHART shows, not what's stored or used in the summary.
            seen: dict[str, dict] = {}
            for row in entity_rows:
                norm = _normalize_period(row.get("period"))
                if norm and norm not in seen:
                    seen[norm] = {**row, "period": norm}
            chart_rows = sorted(seen.values(), key=lambda r: _period_sort_key(r["period"]))
        else:
            x_key = "source_name"
            chart_rows = [r for r in entity_rows if r.get("source_name")]

        if len(chart_rows) < MIN_CHART_POINTS:
            continue

        chart_result = generate_chart.invoke(
            {
                "data": chart_rows,
                "chart_type": "bar",
                "x": x_key,
                "y": "value",
                "title": f"{group['metric']} ({group['unit']}) — {entity} — {hypothesis_id}",
            }
        )
        path = chart_result.get("path")
        if path:
            chart_paths.append(path)

    return chart_paths


def analyst_node(state: dict) -> dict:
    """Analyze facts scoped to the current run_id, per hypothesis.

    Reads: state['hypotheses'], state['run_id'], state['revision_notes']['analyst']
    Writes: state['analysis_results'], state['messages'], state['run_path']
    """
    hypotheses = state.get("hypotheses", [])
    run_id = state.get("run_id")
    revision_note = state.get("revision_notes", {}).get(AGENT_NAME)

    analysis_results = []
    message_parts = []

    for hypothesis in hypotheses:
        hypothesis_id = hypothesis.get("id")

        declare_raw = _declare_data_needed(hypothesis, revision_note)
        data_needed = safe_extract_json(declare_raw).get("data_needed", "")
        message_parts.append(f"[{hypothesis_id}] looking for: {data_needed}")

        if not run_id or not hypothesis_id:
            analysis_results.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "summary": f"insufficient data for {hypothesis_id}: no run_id available yet.",
                    "supports_hypothesis": None,
                    "fact_ids": [],
                    "groups": [],
                    "chart_paths": [],
                }
            )
            continue

        facts = get_facts(run_id, hypothesis_id)
        if not facts:
            analysis_results.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "summary": f"insufficient data for {hypothesis_id}: no facts recorded for this run.",
                    "supports_hypothesis": None,
                    "fact_ids": [],
                    "groups": [],
                    "chart_paths": [],
                }
            )
            continue

        facts_by_id = {fact["id"]: fact for fact in facts}
        groups = _group_facts(facts)

        conflicting_groups = [g for g in groups if g["conflicting"]]
        if conflicting_groups:
            message_parts.append(
                f"[{hypothesis_id}] CONFLICTING FACTS: "
                + "; ".join(
                    f"{g['metric']} ({g['unit']}) ranges {g['min']}-{g['max']} across {g['count']} sources"
                    for g in conflicting_groups
                )
            )

        chart_paths = [
            path
            for group in groups
            for path in _chart_for_group(hypothesis_id, group, facts_by_id)
        ]

        summary_raw = _summarize_groups(hypothesis, groups)
        summary_parsed = safe_extract_json(summary_raw)
        message_parts.append(summary_raw)

        all_fact_ids = [fid for group in groups for fid in group["fact_ids"]]
        analysis_results.append(
            {
                "hypothesis_id": hypothesis_id,
                "summary": summary_parsed.get("summary", ""),
                "supports_hypothesis": summary_parsed.get("supports_hypothesis"),
                "fact_ids": all_fact_ids,
                "groups": groups,
                "chart_paths": chart_paths,
            }
        )

    run_logger.record(
        AGENT_NAME,
        hypotheses_analyzed=len(analysis_results),
        insufficient_data_count=sum(1 for r in analysis_results if not r["fact_ids"]),
        conflicting_group_count=sum(len([g for g in r["groups"] if g.get("conflicting")]) for r in analysis_results),
    )

    return {
        "analysis_results": analysis_results,
        "messages": [AIMessage(content="\n\n".join(message_parts), name=AGENT_NAME)],
        "run_path": [AGENT_NAME],
    }
