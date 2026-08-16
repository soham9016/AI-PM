"""Chart generation tool — turns tabular data into a saved PNG chart."""

import time
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool

CHARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "charts"
SUPPORTED_TYPES = {"bar", "line", "scatter"}


@tool
def generate_chart(data: list[dict], chart_type: str, x: str, y: str, title: str = "") -> dict:
    """Generate a chart PNG from tabular data and save it to data/charts/.

    Args:
        data: list of row dicts, e.g. output of run_sql_query()["rows"].
        chart_type: one of "bar", "line", "scatter".
        x: column name to use for the x-axis.
        y: column name to use for the y-axis.
        title: optional chart title.

    Returns a dict: {path, error}. On failure `error` is set and `path`
    is None.
    """
    if chart_type not in SUPPORTED_TYPES:
        return {"path": None, "error": f"chart_type must be one of {sorted(SUPPORTED_TYPES)}"}
    if not data:
        return {"path": None, "error": "No data provided."}

    df = pd.DataFrame(data)
    if x not in df.columns or y not in df.columns:
        return {"path": None, "error": f"Columns {x!r}/{y!r} not found in data columns {list(df.columns)}"}

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{chart_type}_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    out_path = CHARTS_DIR / filename

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if chart_type == "bar":
            ax.bar(df[x].astype(str), df[y])
        elif chart_type == "line":
            ax.plot(df[x].astype(str), df[y], marker="o")
        elif chart_type == "scatter":
            ax.scatter(df[x], df[y])

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f"{y} by {x}")
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)

    return {"path": str(out_path), "error": None}
