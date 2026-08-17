"""Renders a full graph result into a readable markdown brief — the
human-facing output of a run, replacing run.py's old raw-dict print
block entirely.

Operates on the FULL graph result (the state dict app.invoke() returns),
not just the curated state['final_brief'], since it needs fields
final_brief doesn't carry (current_state_findings, constraint,
hypotheses, dropped_solutions, competitive_audit's raw areas). Every
section is mode-aware by construction: a section reads whatever state
field it needs and renders nothing when that field is empty/default,
rather than branching explicitly on routing_mode — diagnostic-mode runs
naturally have empty solutions/competitive_audit/current_state_findings,
PM-mode runs naturally have empty synthesis, so the sections that don't
apply just produce no output.

No raw dicts, no bare finding IDs anywhere in the output — every citation
is resolved via tools/db.py's get_evidence_by_id into a real markdown
link with the actual source URL. A claim that can't be resolved (a stale
or malformed evidence_id) is skipped rather than rendered as broken
output — the DB is the source of truth, not whatever a solution/verdict
dict happened to claim.
"""

MOSCOW_ORDER = {"Must": 0, "Should": 1, "Could": 2, "Won't": 3}

# Current-state findings: how many items to show per source before
# collapsing the rest -- one press release full of financial detail
# otherwise dominates the whole section (observed: 12 of 13 items from a
# single release). See _current_state_section.
MAX_CURRENT_STATE_ITEMS_PER_SOURCE = 4

# Crude but sufficient: prefer findings that describe what the company
# SHIPS (features/capabilities) over financial results when a source has
# more items than the per-source cap allows -- the point of this section
# is "what already exists that we might be about to re-recommend," not a
# P&L summary.
_FINANCIAL_KEYWORDS = (
    "revenue", "margin", "ebitda", "gov", "gmv", "aov", "average order value",
    "mtu", "mau", "yoy", "growth", "grew", "% of", "improved to", "plateaued",
    "contribution margin",
)


def _looks_like_financial(claim: str) -> bool:
    lowered = (claim or "").lower()
    return any(kw in lowered for kw in _FINANCIAL_KEYWORDS)


def _link(source_name, source_url) -> str:
    name = source_name or source_url or "source"
    return f"[{name}]({source_url})" if source_url else name


def _resolve(evidence_id, db_path=None) -> dict | None:
    from tools.db import get_evidence_by_id  # local import: keeps this module import-light for tests that stub tools.db

    if not evidence_id:
        return None
    row = get_evidence_by_id(evidence_id, db_path=db_path)
    if row is None:
        return None
    claim = row.get("claim")
    if claim is None:  # a fact row, not a finding row
        unit = row.get("unit") or ""
        claim = f"{row.get('metric', 'metric')}: {row.get('value')} {unit}".strip()
    return {
        "claim": claim,
        "source_name": row.get("source_name"),
        "source_url": row.get("source_url"),
        "stance": row.get("stance"),
    }


def _header(result: dict) -> str:
    problem_statement = result.get("problem_statement", "")
    mode = result.get("routing_mode", "")
    problem_type = result.get("problem_type", "")
    constraint = result.get("constraint") or "(none stated)"
    return (
        f"# {problem_statement}\n\n"
        f"**Mode:** {mode} | **Problem type:** {problem_type} | **Constraint:** {constraint}"
    )


def _instrumentation_feature(result: dict) -> dict | None:
    prd = (result.get("final_brief") or {}).get("prd") or {}
    for f in prd.get("features", []) or []:
        if f.get("is_instrumentation"):
            return f
    return None


def _funnel_section(result: dict) -> str:
    funnel = result.get("funnel") or {}
    stages = funnel.get("stages") or []
    if not stages:
        return ""
    verdict = result.get("funnel_verdict") or ""

    lines = [f"## Where's the drop? — {funnel.get('metric', '')}"]
    lines.append("\n| Stage | Definition | Evidence needed | Locatability |")
    lines.append("|---|---|---|---|")
    for s in stages:
        lines.append(
            f"| {s.get('name', '')} | {s.get('definition', '')} | "
            f"{s.get('evidence_needed', '')} | {s.get('evidence_locatability', '')} |"
        )

    if verdict == "undetermined":
        feature = _instrumentation_feature(result)
        # Name it, don't just say "below" -- a promise this section can't
        # keep if the referenced recommendation doesn't actually exist is
        # worse than no pointer at all (observed failure: this callout
        # used to say "below" with nothing there to find).
        pointer = (
            f"See **{feature['name']}** in Recommendations before committing to a fix."
            if feature else
            "Instrumentation is required before committing to a fix on any stage."
        )
        lines.append(
            "\n> **UNDETERMINED** — the gathered evidence does not distinguish between these "
            f"stages. Recommending which stage to build for without knowing where the drop "
            f"actually is would be a guess, not a diagnosis. {pointer}"
        )
    elif verdict:
        lines.append(f"\n**Where the drop is:** {verdict}")
    else:
        lines.append("\n_(not yet determined — evidence gathering is still in progress)_")

    return "\n".join(lines)


def _collect_cited_evidence(result: dict) -> dict:
    """theme (issue-tree branch) -> ordered list of evidence_ids that
    actually drove a recommendation, gathered mode-agnostically: solution
    finding_ids for PM/combined, verdict evidence_ids (via the linked
    hypothesis's branch) for diagnostic/combined. Grouping by branch, not
    raw hypothesis_id, since a branch is a real MECE theme a reader can
    follow — a hypothesis_id string is a database key."""
    by_theme: dict[str, list[str]] = {}

    for s in result.get("solutions", []) or []:
        theme = s.get("branch") or "General"
        for eid in s.get("finding_ids", []) or []:
            bucket = by_theme.setdefault(theme, [])
            if eid not in bucket:
                bucket.append(eid)

    hypotheses_by_id = {h.get("id"): h for h in result.get("hypotheses", []) or [] if h.get("id")}
    for v in (result.get("synthesis") or {}).get("verdicts", []) or []:
        hyp = hypotheses_by_id.get(v.get("hypothesis_id"))
        theme = (hyp or {}).get("branch") or "General"
        for eid in v.get("evidence_ids", []) or []:
            bucket = by_theme.setdefault(theme, [])
            if eid not in bucket:
                bucket.append(eid)

    return by_theme


_STANCE_NOTE = {"supports": "supports this", "contradicts": "contradicts this", "context": "background context"}


def _findings_section(result: dict, db_path=None) -> str:
    by_theme = _collect_cited_evidence(result)
    if not by_theme:
        return ""
    lines = ["## What we found"]
    for theme, evidence_ids in by_theme.items():
        lines.append(f"\n### {theme}")
        for eid in evidence_ids:
            resolved = _resolve(eid, db_path=db_path)
            if resolved is None:
                continue
            note = _STANCE_NOTE.get(resolved.get("stance"), "")
            suffix = f" ({note})" if note else ""
            lines.append(f"- {resolved['claim']} — {_link(resolved['source_name'], resolved['source_url'])}{suffix}")
    return "\n".join(lines)


def _competitors_section(result: dict, db_path=None) -> str:
    audit = result.get("competitive_audit") or {}
    areas = audit.get("areas", [])
    no_evidence = audit.get("competitors_with_no_evidence", [])
    unclassified_branches = audit.get("unclassified_branches", [])
    if not areas and not no_evidence and not unclassified_branches:
        return ""
    lines = ["## What competitors do"]
    for area in areas:
        lines.append(f"\n### {area.get('branch')} — {area.get('classification')}")
        if area.get("rationale"):
            lines.append(f"_{area['rationale']}_")
        for c in area.get("competitors", []) or []:
            resolved = _resolve(c.get("evidence_id"), db_path=db_path)
            source = f" — {_link(resolved['source_name'], resolved['source_url'])}" if resolved else ""
            lines.append(f"- **{c.get('name')}**: {c.get('does_it')}{source}")
    if unclassified_branches:
        lines.append("\n### Branches with no competitive evidence")
        for branch in unclassified_branches:
            lines.append(f"- **{branch}**: no evidence found to classify this branch")
    if no_evidence:
        lines.append("\n### Competitors with no evidence found")
        for name in no_evidence:
            lines.append(f"- **{name}**: no evidence found")
    return "\n".join(lines)


def _current_state_section(result: dict) -> str:
    findings = result.get("current_state_findings") or []
    company = (result.get("anchor_terms") or {}).get("company") or ""
    heading_name = company or "the company"
    # Current-state research (agents/primary_research.py) only runs in
    # "pm"/"combined" routing_mode -- diagnostic-mode runs naturally have
    # current_state_findings=[], but that means "never attempted", not
    # "attempted and found nothing", so it must stay silent rather than
    # claim a search that never happened. This is the one deliberate
    # exception to this file's usual "render nothing when the field is
    # empty, no routing_mode branching" convention (see module
    # docstring) -- an empty list is genuinely ambiguous between those
    # two states here, unlike every other section.
    current_state_applicable = result.get("routing_mode") in ("pm", "combined")
    if not findings:
        if not company or not current_state_applicable:
            # No company named, or this mode never runs current-state
            # research at all -- distinct from "we looked, found
            # nothing" below.
            return ""
        return (
            f"## What {heading_name} already has\n\n"
            f"No current-state evidence found — nothing on {heading_name}'s own "
            f"domain, app store listing, or help centre surfaced anything "
            f"relevant (see agents/primary_research.py's domain filter)."
        )

    by_source: dict[str, list[dict]] = {}
    order: list[str] = []
    for f in findings:
        key = f.get("source_url") or f.get("source_name") or "unknown source"
        if key not in by_source:
            by_source[key] = []
            order.append(key)
        by_source[key].append(f)

    lines = [f"## What {heading_name} already has"]
    for key in order:
        items = by_source[key]
        # Feature/capability claims first (what this section is actually
        # for), financial-result claims last -- when a source has more
        # items than the cap, the financial ones are what gets dropped.
        items_sorted = sorted(items, key=lambda f: _looks_like_financial(f.get("claim", "")))
        capped = items_sorted[:MAX_CURRENT_STATE_ITEMS_PER_SOURCE]
        source_name = capped[0].get("source_name") or key
        lines.append(f"\n### {source_name}")
        for f in capped:
            lines.append(f"- {f.get('claim')} — {_link(f.get('source_name'), f.get('source_url'))}")
        omitted = len(items) - len(capped)
        if omitted > 0:
            lines.append(f"_(+{omitted} more from this source, omitted for brevity)_")
    return "\n".join(lines)


def _dropped_section(result: dict, db_path=None) -> str:
    dropped = result.get("dropped_solutions") or []
    if not dropped:
        return ""
    exists = [s for s in dropped if s.get("drop_reason") == "already exists"]
    killed = [s for s in dropped if (s.get("drop_reason") or "").startswith("solution review:")]
    constraint_violations = [
        s for s in dropped
        if s.get("drop_reason") != "already exists" and not (s.get("drop_reason") or "").startswith("solution review:")
    ]

    lines = ["## Considered and dropped"]
    if exists:
        lines.append("\n### Already exists")
        for s in exists:
            resolved = _resolve(s.get("current_state_evidence_id"), db_path=db_path)
            source = _link(resolved["source_name"], resolved["source_url"]) if resolved else "source not captured"
            lines.append(f"- **{s.get('name')}**: {s.get('problem_addressed', '')} — {source}")
    if killed:
        lines.append("\n### Killed on review")
        for s in killed:
            reason = (s.get("drop_reason") or "").removeprefix("solution review:").strip()
            lines.append(f"- **{s.get('name')}**: {reason}")
    if constraint_violations:
        lines.append("\n### Doesn't fit the stated constraint")
        for s in constraint_violations:
            lines.append(f"- **{s.get('name')}**: {s.get('drop_reason')}")
    return "\n".join(lines)


def _feature_evidence_lines(feature: dict, result: dict, db_path=None) -> list[str]:
    lines: list[str] = []

    solution_id = feature.get("addresses_solution")
    if solution_id:
        solutions_by_id = {s.get("id"): s for s in result.get("solutions", []) or []}
        solution = solutions_by_id.get(solution_id)
        for eid in (solution or {}).get("finding_ids", []) or []:
            resolved = _resolve(eid, db_path=db_path)
            if resolved:
                lines.append(f"  - {resolved['claim']} — {_link(resolved['source_name'], resolved['source_url'])}")

    hypothesis_id = feature.get("addresses_hypothesis")
    if hypothesis_id:
        verdicts_by_id = {v.get("hypothesis_id"): v for v in (result.get("synthesis") or {}).get("verdicts", []) or []}
        verdict = verdicts_by_id.get(hypothesis_id)
        for eid in (verdict or {}).get("evidence_ids", []) or []:
            resolved = _resolve(eid, db_path=db_path)
            if resolved:
                lines.append(f"  - {resolved['claim']} — {_link(resolved['source_name'], resolved['source_url'])}")

    if not lines:
        lines.append("  - (exploratory — no direct evidence link)")
    return lines


def _render_feature(feature: dict, result: dict, db_path=None) -> list[str]:
    marker = "🔍 INSTRUMENT — " if feature.get("is_instrumentation") else ""
    lines = [f"\n### {marker}[{feature.get('moscow', '?')}] {feature.get('name', '')}", feature.get("description", "")]

    if feature.get("is_instrumentation"):
        plan = feature.get("instrumentation_plan") or {}
        lines.append(
            f"\n**What to measure:** {plan.get('measure', '')}  \n"
            f"**At stage(s):** {plan.get('stage', '')}  \n"
            f"**Result that would identify the drop:** {plan.get('identifying_result', '')}"
        )
        # No "Why this feature" evidence block -- an instrumentation
        # feature is deliberately evidence-free by construction (the gap
        # itself is the justification, see agents/solution_review.py's
        # auto-promotion); the generic fallback line ("exploratory — no
        # direct evidence link") would misleadingly undersell it.
    else:
        lines.append("\n**Why this feature:**")
        lines.extend(_feature_evidence_lines(feature, result, db_path))

    funnel_stage = feature.get("addresses_funnel_stage")
    if funnel_stage:
        lines.append(f"\n**Funnel stage:** {funnel_stage}")

    if feature.get("works_within_constraint"):
        lines.append(f"\n**Within the stated constraint:** {feature['works_within_constraint']}")

    differentiation = feature.get("differentiation")
    if differentiation:
        if differentiation.strip().lower() == "parity":
            lines.append("\n**Parity with competitors** — does not differentiate")
        else:
            lines.append(f"\n**Differentiation:** {differentiation}")

    lines.append(
        f"\n**RICE:** reach={feature.get('reach')}, impact={feature.get('impact')}, "
        f"confidence={feature.get('confidence')}, effort={feature.get('effort')} → "
        f"**{feature.get('rice_score')}**"
    )
    if feature.get("impact_capped_reason"):
        lines.append(f"_{feature['impact_capped_reason']}_")

    lines.append(
        f"\n**Success metric:** {feature.get('success_metric', '')}  \n"
        f"**Guardrail metric:** {feature.get('guardrail_metric') or '(none)'}"
    )

    vp = feature.get("validation_plan") or {}
    if vp:
        lines.append(
            f"\n**Validation plan:** cohort — {vp.get('cohort', '?')}; duration — {vp.get('duration', '?')}; "
            f"success threshold — {vp.get('success_threshold', '?')}  \n"
            f"Tests: {vp.get('hypothesis_tested', '')}"
        )

    lines.append(f"\n**Evidence strength:** {feature.get('evidence_strength', '')}")
    signal_strength = feature.get("signal_strength")
    if signal_strength:
        lines.append(f"  \n**Signal strength:** {signal_strength}")
    return lines


def _recommendations_section(result: dict, db_path=None) -> str:
    prd = (result.get("final_brief") or {}).get("prd")
    if not prd:
        return ""
    features = prd.get("features", [])
    if not features:
        return ""

    # Instrumentation features render FIRST, ahead of every build feature,
    # regardless of MoSCoW/RICE -- pm.py already forces moscow="Must" for
    # these, but that's not sufficient on its own (a regulation_backed
    # Must could still out-RICE it within the same tier); this is the
    # actual guarantee.
    instrumentation = [f for f in features if f.get("is_instrumentation")]
    shipped = [f for f in features if f.get("moscow") != "Won't" and not f.get("is_instrumentation")]
    wont = [f for f in features if f.get("moscow") == "Won't" and not f.get("is_instrumentation")]
    shipped.sort(key=lambda f: (MOSCOW_ORDER.get(f.get("moscow"), 9), -(f.get("rice_score") or 0)))
    wont.sort(key=lambda f: -(f.get("rice_score") or 0))

    lines = ["## Recommendations"]
    north_star = prd.get("north_star_metric", "")
    if north_star:
        lines.append(f"North-star metric: **{north_star}**")

    for feature in instrumentation:
        lines.extend(_render_feature(feature, result, db_path))
    for feature in shipped:
        lines.extend(_render_feature(feature, result, db_path))

    if wont:
        lines.append("\n### Considered, not prioritized (Won't)")
        for feature in wont:
            lines.append(f"- **{feature.get('name')}** (RICE {feature.get('rice_score')}): {feature.get('moscow_reason', '')}")

    return "\n".join(lines)


def _gaps_section(result: dict) -> str:
    limitations = (result.get("final_brief") or {}).get("limitations") or []
    unexamined = (result.get("synthesis") or {}).get("unexamined_branches") or []
    no_evidence = (result.get("competitive_audit") or {}).get("competitors_with_no_evidence") or []
    if not (limitations or unexamined or no_evidence):
        return ""

    lines = ["## What we could not establish"]
    for item in limitations:
        issue = item.get("issue") if isinstance(item, dict) else str(item)
        lines.append(f"- {issue}")
    for branch in unexamined:
        lines.append(f"- Issue-tree branch not examined: {branch}")
    for name in no_evidence:
        lines.append(f"- No evidence found for competitor: {name}")
    return "\n".join(lines)


def build_markdown_brief(result: dict, db_path=None) -> str:
    sections = [
        _header(result),
        _funnel_section(result),
        _findings_section(result, db_path),
        _competitors_section(result, db_path),
        _current_state_section(result),
        _dropped_section(result, db_path),
        _recommendations_section(result, db_path),
        _gaps_section(result),
    ]
    return "\n\n".join(s for s in sections if s.strip())
