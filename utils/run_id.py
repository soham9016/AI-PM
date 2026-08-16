"""Run ID generator.

Stamps a single graph invocation with a unique id used to scope evidence-
base rows (findings/facts) to that run, so facts from a previous run
never leak into the current one's analysis.
"""

import uuid
from datetime import datetime, timezone


def generate_run_id() -> str:
    """Return a unique, sortable run id, e.g. '20260809T142233Z-3f9a1c'."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:6]}"
