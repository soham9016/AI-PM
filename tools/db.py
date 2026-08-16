"""Shared schema + access layer for the evidence base.

Two tables:
  findings — qualitative evidence (a claim + stance), the primary track.
  facts    — numeric evidence, opportunistic, not required.

"No source, no storage": both tables enforce source_url NOT NULL at the
DB level (a CHECK/NOT NULL constraint, not just a prompt instruction),
so provenance cannot be skipped by an upstream bug or a model that
forgets to mention where a claim came from. Facts are never deduped —
two sources reporting the same figure is signal, not noise.

hypothesis_id is a plain grouping key, not a foreign key to a
"hypotheses" table (hypotheses only ever live in graph state, never in
this DB) — the diagnostic path uses real hypothesis ids there; the PM
path (agents/primary_research.py, agents/competitive_audit.py) reuses
the same column as a topic_id ("primary", "competitor:<name>") since
there's no hypothesis to key on. get_findings_for_run/get_facts_for_run
pull the whole evidence pool for a run regardless of that column, which
is what agents/solution_framing.py needs.
"""

import logging
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "business.db"

logger = logging.getLogger("business_copilot.db")

ALLOWED_STANCES = {"supports", "contradicts", "context"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

# Extend as new metric types show up in practice.
# "inr" (plain rupees) is distinct from "inr_cr" (rupees-in-crore) -- a
# consumer price like "milk costs Rs 65" is a plain-currency amount with
# no scale word, and was previously falling through to "inr_cr" for lack
# of a plain-currency Indian-rupee unit, turning Rs 65 into Rs 65 crore.
ALLOWED_UNITS = {
    "usd",
    "inr",
    "usd_bn",
    "inr_cr",
    "percent",
    "count",
    "million",
    "billion",
    "ratio",
    "days",
    "months",
    "minutes",
    "hours",
    "users",
    "score",
    "usd_per_user",
    "usd_per_month",
    "requests_per_day",
}

# Common ways a model (or a source page) phrases a unit that isn't the
# exact canonical string above. Rejecting "%" for a percentage just
# because it isn't spelled "percent" throws away real evidence for no
# reason — normalize known synonyms before validating against
# ALLOWED_UNITS rather than after.
UNIT_ALIASES: dict[str, str] = {
    "%": "percent",
    "pct": "percent",
    "percentage": "percent",
    # Plain rupee amounts -- NOT crore-scaled. Distinct from the
    # "crore"/"cr"/"cr inr" family below, which all map to inr_cr.
    "₹": "inr",
    "rs": "inr",
    "rs.": "inr",
    "rupee": "inr",
    "rupees": "inr",
    "inr rupees": "inr",
    "usd_billion": "usd_bn",
    "usd billion": "usd_bn",
    "billion usd": "usd_bn",
    "usd bn": "usd_bn",
    "$bn": "usd_bn",
    "bn usd": "usd_bn",
    "crore": "inr_cr",
    "inr crore": "inr_cr",
    "inr_crore": "inr_cr",
    "rs crore": "inr_cr",
    "rs. crore": "inr_cr",
    "cr": "inr_cr",
    # Word-order/multi-token crore variants -- "cr INR" (value-then-currency
    # order, common in Indian financial reporting) was falling through the
    # single "crore"/"cr" aliases above and getting rejected outright,
    # taking the actual revenue/loss figures with it.
    "cr inr": "inr_cr",
    "inr cr": "inr_cr",
    "rs cr": "inr_cr",
    "crore inr": "inr_cr",
    # Generic scale multipliers (not currency-specific -- e.g. "4.45 million
    # orders/day"). "bn"/"billion" alone (no currency word) map to the
    # generic "billion", distinct from the usd_bn/inr_cr currency units above.
    "mn": "million",
    "millions": "million",
    "bn": "billion",
    "billions": "billion",
    "times": "ratio",
    "x": "ratio",
    "multiple": "ratio",
    # Duration -- a legitimate operational metric (e.g. delivery time).
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    stance TEXT NOT NULL CHECK(stance IN ('supports', 'contradicts', 'context')),
    source_url TEXT NOT NULL,
    source_name TEXT,
    confidence TEXT CHECK(confidence IS NULL OR confidence IN ('high', 'medium', 'low')),
    retrieved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    entity TEXT,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    period TEXT,
    source_url TEXT NOT NULL,
    source_name TEXT,
    confidence TEXT CHECK(confidence IS NULL OR confidence IN ('high', 'medium', 'low')),
    retrieved_at TEXT NOT NULL
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection, self-initializing the schema on every call.

    The app doesn't depend on anyone remembering to run scripts/seed_db.py
    first — CREATE TABLE IF NOT EXISTS is idempotent and cheap enough to
    run per-connection for a two-table SQLite DB.
    """
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the findings/facts tables if they don't already exist.

    Kept for scripts/seed_db.py's explicit, no-surprises init step —
    get_connection() above now does this on every call too, so the app
    is self-initializing even if this is never called.
    """
    get_connection(db_path).close()


def _normalize_confidence(record: dict) -> str | None:
    """confidence is categorical ("high"/"medium"/"low"), not a numeric
    score — an LLM's self-reported 0-1 float isn't a calibrated
    probability, and treating it like one is a type mismatch waiting to
    happen downstream. Confidence is supplementary metadata, not a
    load-bearing field like source_url/stance/unit, so an invalid value
    is dropped (logged) rather than rejecting the whole row over it."""
    confidence = record.get("confidence")
    if confidence is None:
        return None
    if isinstance(confidence, str) and confidence.strip().lower() in ALLOWED_CONFIDENCE:
        return confidence.strip().lower()
    logger.warning("Dropping invalid confidence %r (expected one of %s)", confidence, sorted(ALLOWED_CONFIDENCE))
    return None


def _normalize_unit(raw_unit) -> str | None:
    """Map common unit synonyms to the canonical ALLOWED_UNITS spelling
    before validating. Only string input can be normalized; anything else
    is passed through unchanged and will fail the ALLOWED_UNITS check."""
    if not isinstance(raw_unit, str):
        return raw_unit
    cleaned = raw_unit.strip().lower()
    if cleaned in ALLOWED_UNITS:
        return cleaned
    return UNIT_ALIASES.get(cleaned, raw_unit)


# Metric-name keywords implying the number itself is a percentage/rate.
_PERCENT_METRIC_KEYWORDS = ("percentage", "percent", "rate", "share", "margin", "cagr")
# Metric-name keywords implying the number is a plain count, not a rate.
_COUNT_METRIC_KEYWORDS = ("number of", "count of", "total")
_PERCENT_VALUE_MIN, _PERCENT_VALUE_MAX = -100, 1000

# A scale-denominated unit already carries a multiplier in its name — a
# value that ALSO looks fully expanded (e.g. 4718000000 stored as
# "inr_cr", meaning the model both multiplied by 1e7 AND kept the crore
# label) is definitionally implausible at that unit's own scale. These
# ceilings are deliberately generous (well above any real single-company
# or single-sector figure this system would encounter) so they only
# catch the double-counting failure mode, not large-but-real numbers.
_SCALE_UNIT_MAX_VALUE = {
    "usd_bn": 10_000,       # already billions; >10,000 implies >$10 trillion
    "inr_cr": 10_000_000,   # already crores; generous headroom above any single company/sector figure
    "million": 1_000_000,   # already millions; >1,000,000 implies >1 trillion units
    "billion": 10_000,
}


def _scale_plausibility_error(unit: str, value: float) -> str | None:
    """Catch a value that's been expanded to raw units while its unit
    still carries a scale multiplier (crore, billion, million) — e.g.
    "Rs 4,718 crore" stored as value=4718000000, unit="inr_cr" instead of
    value=4718. The row is fully sourced and unit-coherent by every other
    check and still off by roughly the scale factor. Returns a rejection
    reason, or None if the value is plausible at that unit's scale."""
    ceiling = _SCALE_UNIT_MAX_VALUE.get(unit)
    if ceiling is not None and abs(value) > ceiling:
        return (
            f"unit {unit!r} already denotes a scale; value {value!r} is implausible at that "
            f"scale (ceiling {ceiling!r}) — likely double-counted (expanded the number AND kept the scale unit)"
        )
    return None


def _metric_unit_coherence_error(metric: str, unit: str, value: float) -> str | None:
    """Catch a metric NAME that contradicts its own unit/value — e.g. a
    metric called "percentage of restaurants" stored with unit "count"
    and value 500000 (the actual bug: the page said "500,000
    restaurants," and the word "percentage" leaked in from elsewhere in
    the sentence). Provenance (a real source_url) doesn't catch this —
    the row is fully sourced and still nonsense. Returns a rejection
    reason, or None if the metric/unit/value are coherent.

    Checked in this order so a metric like "total market share" (matches
    both the percent and count keyword lists) doesn't false-positive:
    percent-flavored naming wins if both match, since "share"/"margin"/
    etc. are stronger, less ambiguous signals than a bare "total".
    """
    metric_lower = (metric or "").lower()
    is_percent_name = any(kw in metric_lower for kw in _PERCENT_METRIC_KEYWORDS)
    is_count_name = any(kw in metric_lower for kw in _COUNT_METRIC_KEYWORDS)

    if is_percent_name and unit != "percent":
        return f"metric name {metric!r} implies a percentage but unit is {unit!r}, not 'percent'"

    if unit == "percent" and not (_PERCENT_VALUE_MIN <= value <= _PERCENT_VALUE_MAX):
        return (
            f"unit is 'percent' but value {value!r} is not a plausible percentage "
            f"({_PERCENT_VALUE_MIN} to {_PERCENT_VALUE_MAX})"
        )

    if is_count_name and not is_percent_name and unit == "percent":
        return f"metric name {metric!r} implies a count but unit is 'percent'"

    return None


# A per-unit consumer price/cost/fare is never legitimately reported in a
# scale-denominated currency unit -- "milk costs Rs 65" is a plain
# amount, not "Rs 65 crore". Distinct from _scale_plausibility_error
# above, which catches the value being too LARGE for its unit; this
# catches the unit itself being the wrong currency unit even when the
# value looks individually plausible (65 is well under the inr_cr
# ceiling, so the plausibility check alone can't catch this case).
_PRICE_METRIC_KEYWORDS = ("price", "cost", "mrp", "fare", "fee", "charge")
_SCALE_CURRENCY_UNITS = {"inr_cr", "usd_bn", "million", "billion"}


def _currency_scale_error(metric: str, unit: str) -> str | None:
    metric_lower = (metric or "").lower()
    if unit in _SCALE_CURRENCY_UNITS and any(kw in metric_lower for kw in _PRICE_METRIC_KEYWORDS):
        return (
            f"metric name {metric!r} looks like a per-unit consumer price/cost, which should use "
            f"a plain currency unit ('inr'/'usd'), not the scale-denominated unit {unit!r}"
        )
    return None


# When a metric name explicitly names a unit word (not just implies a
# category like "percentage"/"count" above), the stored unit must match
# it exactly -- e.g. a metric literally called "number of minutes" stored
# with unit "hours" is contradicting its own name. Compound rate units
# (unit contains "_per_") are excluded: "cost per month" legitimately
# uses unit "usd_per_month", and "month" appearing in the metric name
# there isn't a conflict with that compound unit.
_METRIC_UNIT_WORD_MAP = {
    "minutes": "minutes", "minute": "minutes", "mins": "minutes", "min": "minutes",
    "hours": "hours", "hour": "hours", "hrs": "hours", "hr": "hours",
    "days": "days", "day": "days",
    "months": "months", "month": "months",
    "percent": "percent", "percentage": "percent",
    "crore": "inr_cr",
    "million": "million",
    "billion": "billion",
    "users": "users",
}


def _explicit_unit_conflict_error(metric: str, unit: str) -> str | None:
    if "_per_" in (unit or ""):
        return None
    metric_lower = (metric or "").lower()
    for word, canonical in _METRIC_UNIT_WORD_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", metric_lower) and canonical != unit:
            return (
                f"metric name {metric!r} explicitly names unit {canonical!r} but the stored unit is {unit!r}"
            )
    return None


def insert_finding(record: dict, db_path: Path | None = None) -> int | None:
    """Validate and insert one finding row.

    Required: run_id, hypothesis_id, claim, stance (in ALLOWED_STANCES),
    source_url. Returns the new row id, or None (and logs a rejection)
    if validation fails.
    """
    if not record.get("source_url"):
        logger.warning("Rejected finding (no source_url): %r", record)
        return None
    if not record.get("claim"):
        logger.warning("Rejected finding (empty claim): %r", record)
        return None
    if not record.get("run_id") or not record.get("hypothesis_id"):
        logger.warning("Rejected finding (missing run_id/hypothesis_id): %r", record)
        return None
    if record.get("stance") not in ALLOWED_STANCES:
        logger.warning("Rejected finding (invalid stance %r): %r", record.get("stance"), record)
        return None

    row = {
        "run_id": record["run_id"],
        "hypothesis_id": record["hypothesis_id"],
        "claim": record["claim"],
        "stance": record["stance"],
        "source_url": record["source_url"],
        "source_name": record.get("source_name"),
        "confidence": _normalize_confidence(record),
        "retrieved_at": record["retrieved_at"],
    }
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO findings (run_id, hypothesis_id, claim, stance, source_url, source_name, confidence, retrieved_at) "
            "VALUES (:run_id, :hypothesis_id, :claim, :stance, :source_url, :source_name, :confidence, :retrieved_at)",
            row,
        )
        conn.commit()
        new_id = cursor.lastrowid

    # Rejections were already logged above; without a matching success log
    # there was no durable record of what the evidence base actually
    # contains short of querying it live — this is that record.
    logger.info(
        "Inserted finding %s (run_id=%s, hypothesis_id=%s, stance=%s): %r <- %s",
        new_id, row["run_id"], row["hypothesis_id"], row["stance"], row["claim"], row["source_url"],
    )
    return new_id


def insert_fact(record: dict, db_path: Path | None = None) -> int | None:
    """Validate and insert one fact row.

    Required: run_id, hypothesis_id, metric, value (castable to float),
    unit (in ALLOWED_UNITS), source_url, and the metric name must be
    coherent with the unit/value (see _metric_unit_coherence_error — a
    metric named "percentage of X" with unit "count" is rejected even
    though every individual field looks valid on its own). For
    scale-denominated units (inr_cr/usd_bn/million/billion), the value
    must also be plausible at that scale (see _scale_plausibility_error —
    a value that's been expanded to raw units while the unit still
    carries the scale multiplier is rejected); a price/cost-flavored
    metric name must not use a scale-denominated currency unit (see
    _currency_scale_error); and a metric name that explicitly names a
    unit word must match the stored unit (see
    _explicit_unit_conflict_error). Returns the new row id, or None (and
    logs a rejection) if validation fails.
    """
    if not record.get("source_url"):
        logger.warning("Rejected fact (no source_url): %r", record)
        return None
    if not record.get("metric"):
        logger.warning("Rejected fact (empty metric): %r", record)
        return None
    if not record.get("run_id") or not record.get("hypothesis_id"):
        logger.warning("Rejected fact (missing run_id/hypothesis_id): %r", record)
        return None
    try:
        value = float(record.get("value"))
    except (TypeError, ValueError):
        logger.warning("Rejected fact (value not castable to float): %r", record)
        return None

    unit = _normalize_unit(record.get("unit"))
    if unit not in ALLOWED_UNITS:
        logger.warning("Rejected fact (unit %r not in ALLOWED_UNITS, even after normalization): %r", record.get("unit"), record)
        return None

    coherence_error = _metric_unit_coherence_error(record.get("metric", ""), unit, value)
    if coherence_error:
        logger.warning("Rejected fact (metric/unit incoherent: %s): %r", coherence_error, record)
        return None

    scale_error = _scale_plausibility_error(unit, value)
    if scale_error:
        logger.warning("Rejected fact (scale implausible: %s): %r", scale_error, record)
        return None

    currency_scale_error = _currency_scale_error(record.get("metric", ""), unit)
    if currency_scale_error:
        logger.warning("Rejected fact (currency/scale mismatch: %s): %r", currency_scale_error, record)
        return None

    unit_conflict_error = _explicit_unit_conflict_error(record.get("metric", ""), unit)
    if unit_conflict_error:
        logger.warning("Rejected fact (metric names a conflicting unit: %s): %r", unit_conflict_error, record)
        return None

    row = {
        "run_id": record["run_id"],
        "hypothesis_id": record["hypothesis_id"],
        "entity": record.get("entity"),
        "metric": record["metric"],
        "value": value,
        "unit": unit,
        "period": record.get("period"),
        "source_url": record["source_url"],
        "source_name": record.get("source_name"),
        "confidence": _normalize_confidence(record),
        "retrieved_at": record["retrieved_at"],
    }
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO facts (run_id, hypothesis_id, entity, metric, value, unit, period, source_url, source_name, confidence, retrieved_at) "
            "VALUES (:run_id, :hypothesis_id, :entity, :metric, :value, :unit, :period, :source_url, :source_name, :confidence, :retrieved_at)",
            row,
        )
        conn.commit()
        new_id = cursor.lastrowid

    logger.info(
        "Inserted fact %s (run_id=%s, hypothesis_id=%s): %s (%s) = %s %s <- %s",
        new_id, row["run_id"], row["hypothesis_id"], row["entity"], row["metric"], row["value"], row["unit"], row["source_url"],
    )
    return new_id


def get_findings(run_id: str, hypothesis_id: str, db_path: Path | None = None) -> list[dict]:
    """All findings rows for one (run_id, hypothesis_id), parameterized (no string-built SQL)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM findings WHERE run_id = ? AND hypothesis_id = ? ORDER BY id",
            (run_id, hypothesis_id),
        ).fetchall()
    return [dict(row) for row in rows]


def get_facts(run_id: str, hypothesis_id: str, db_path: Path | None = None) -> list[dict]:
    """All facts rows for one (run_id, hypothesis_id), parameterized (no string-built SQL)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE run_id = ? AND hypothesis_id = ? ORDER BY id",
            (run_id, hypothesis_id),
        ).fetchall()
    return [dict(row) for row in rows]


def get_findings_for_run(run_id: str, db_path: Path | None = None) -> list[dict]:
    """All findings rows for a run, regardless of hypothesis_id.

    For the PM path: primary_research/competitive_audit tag rows with a
    topic_id in the hypothesis_id column (there's no hypothesis to key
    on), and solution_framing needs to pull the WHOLE evidence pool for
    the run rather than one topic at a time — this is that query.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_facts_for_run(run_id: str, db_path: Path | None = None) -> list[dict]:
    """All facts rows for a run, regardless of hypothesis_id. See
    get_findings_for_run — same rationale, numeric track."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM facts WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def evidence_exists(evidence_id: str, db_path: Path | None = None) -> bool:
    """Check a "finding:<id>" / "fact:<id>" reference against the DB.

    Used by the critic to verify a cited evidence_id is real, not just
    internally consistent within the synthesis object.
    """
    if not isinstance(evidence_id, str):
        return False
    kind, _, raw_id = evidence_id.partition(":")
    if kind not in ("finding", "fact") or not raw_id.isdigit():
        return False
    table = "findings" if kind == "finding" else "facts"
    with get_connection(db_path) as conn:
        row = conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (int(raw_id),)).fetchone()
    return row is not None


def get_evidence_by_id(evidence_id: str, db_path: Path | None = None) -> dict | None:
    """Single-row lookup by exact "finding:<id>" / "fact:<id>" reference.

    Every other accessor is scoped by (run_id, hypothesis_id) or run_id —
    this is the one a report/brief builder needs, since it only has the
    exact evidence_ids cited in solutions[].finding_ids /
    verdicts[].evidence_ids / competitive-audit citations, not a scope to
    query by. Returns the full row (including source_url/source_name) or
    None if the id is malformed or doesn't exist.
    """
    if not isinstance(evidence_id, str):
        return None
    kind, _, raw_id = evidence_id.partition(":")
    if kind not in ("finding", "fact") or not raw_id.isdigit():
        return None
    table = "findings" if kind == "finding" else "facts"
    with get_connection(db_path) as conn:
        row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (int(raw_id),)).fetchone()
    return dict(row) if row is not None else None
