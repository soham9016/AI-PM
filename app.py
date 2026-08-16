"""Streamlit UI — thin wrapper only: build_graph(), stream, render.

No business logic lives here. Every domain rule (RICE scoring, MoSCoW,
evidence validation, provider fallback, brief formatting) lives in
agents/tools/utils, same as run.py. This file's only two jobs are (1)
drive app.stream() and surface progress as it lands, since a run takes
5-10 minutes with rate-limit backoff and must not look frozen, and (2)
render what agents/tools already produced — build_markdown_brief for the
brief, source_tier for the evidence explorer's tier column, run_sql_query
for ad-hoc evidence browsing. Nothing here reimplements what those
already do.
"""

import re
from pathlib import Path

import streamlit as st
import os
for _key in ("GROQ_API_KEY", "TAVILY_API_KEY", "CEREBRAS_API_KEY"):
    if _key in st.secrets and not os.environ.get(_key):
        os.environ[_key] = str(st.secrets[_key])
import config
from graph import build_graph
from state import new_state
from tools.brief import build_markdown_brief
from tools.sql_query import run_sql_query
from utils.logger import run_logger
from utils.source_tier import source_tier

DATA_DIR = Path(__file__).parent / "data"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

EXAMPLES = {
    "Cult.fit — live classes adoption": (
        "Only 11% of Cult.fit's gym-booking users have ever tried a live class, "
        "even though live classes are prominently featured in the app. What "
        "should Cult.fit build to increase live-class adoption among existing "
        "gym-booking users?"
    ),
    "Nykaa — checkout abandonment": (
        "Nykaa's app sees high browsing and add-to-cart activity, but a large "
        "share of users abandon before completing checkout. Recommend what "
        "Nykaa should build to improve checkout completion among users who "
        "have already added items to their cart."
    ),
    "Zepto — Tier-2 retention": (
        "Why did Zepto's Tier-2 weekly retention drop, and what should we "
        "build to fix it — given we are not adding new dark stores in the "
        "next two quarters?"
    ),
    "Zomato — DPDP consent": (
        "India's Digital Personal Data Protection Act (DPDP) requires "
        "explicit, granular consent before collecting user data. What should "
        "Zomato build to comply with DPDP's consent requirements?"
    ),
}


def _missing_keys() -> list[str]:
    missing = []
    if not config.GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not config.TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    return missing


def _past_briefs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("brief_*.md"), reverse=True)


def _run_ids() -> list[str]:
    result = run_sql_query.invoke({
        "query": "SELECT DISTINCT run_id FROM findings UNION SELECT DISTINCT run_id FROM facts ORDER BY run_id DESC",
    })
    if result.get("error"):
        return []
    return [row["run_id"] for row in result["rows"]]


def _evidence_for_run(run_id: str) -> tuple[dict, dict]:
    if not RUN_ID_RE.match(run_id):
        return {"error": "invalid run_id"}, {"error": "invalid run_id"}
    findings = run_sql_query.invoke({
        "query": f"SELECT id, hypothesis_id, claim, stance, source_name, source_url FROM findings WHERE run_id = '{run_id}' ORDER BY id",
    })
    facts = run_sql_query.invoke({
        "query": f"SELECT id, hypothesis_id, entity, metric, value, unit, source_name, source_url FROM facts WHERE run_id = '{run_id}' ORDER BY id",
    })
    return findings, facts


def _run_analysis(problem_statement: str, progress_area) -> dict | None:
    graph_app = build_graph()
    init = new_state(problem_statement)
    run_logger.reset()
    full_state = dict(init)
    log_seen = 0

    try:
        for update in graph_app.stream(init, config={"recursion_limit": 50}, stream_mode="updates"):
            for node_name, partial in update.items():
                if not partial:
                    continue
                for key, value in partial.items():
                    if key == "run_path":
                        full_state["run_path"] = full_state.get("run_path", []) + value
                    else:
                        full_state[key] = value

                new_entries = run_logger.full_path[log_seen:]
                log_seen = len(run_logger.full_path)

                if node_name == "engagement_manager":
                    continue  # routing ticks -- not an agent completing work
                with progress_area:
                    st.markdown(f"✅ **{node_name}**")
                    for entry in new_entries:
                        metrics = {k: v for k, v in entry.items() if k not in ("agent", "timestamp")}
                        if metrics:
                            st.json(metrics, expanded=False)
    except Exception as exc:  # noqa: BLE001 -- surface the message + partial progress, not a stack trace
        st.error(f"Run failed: {exc}")
        st.write("Partial run path:", full_state.get("run_path", []))
        return None

    return full_state


st.set_page_config(page_title="Business Copilot", layout="wide")
st.title("Business Research & Analysis Copilot")

missing = _missing_keys()
if missing:
    st.error(f"Missing required key(s) in .env: {', '.join(missing)}. Copy .env.example to .env and fill them in.")

if "problem_statement" not in st.session_state:
    st.session_state["problem_statement"] = ""
if "brief_markdown" not in st.session_state:
    st.session_state["brief_markdown"] = None
if "brief_run_id" not in st.session_state:
    st.session_state["brief_run_id"] = None

with st.sidebar:
    st.subheader("Example problems")
    for label, text in EXAMPLES.items():
        if st.button(label, width="stretch"):
            st.session_state["problem_statement"] = text
            st.session_state["brief_markdown"] = None

    st.subheader("Past runs")
    past = _past_briefs()
    if past:
        chosen = st.selectbox("View a past brief", options=["—"] + [p.name for p in past])
        if chosen != "—":
            st.session_state["brief_markdown"] = (DATA_DIR / chosen).read_text(encoding="utf-8")
            st.session_state["brief_run_id"] = chosen.removeprefix("brief_").removesuffix(".md")
    else:
        st.caption("No past runs yet.")

run_tab, evidence_tab = st.tabs(["Run", "Evidence explorer"])

with run_tab:
    st.text_area("Business problem statement", key="problem_statement", height=120)
    run_clicked = st.button("Run analysis", type="primary", disabled=bool(missing))
    st.caption("A run typically takes 5-10 minutes (rate-limit backoff on the free tier). Progress streams below as each agent completes.")

    progress_area = st.container()

    if run_clicked and st.session_state["problem_statement"].strip():
        result = _run_analysis(st.session_state["problem_statement"], progress_area)
        if result is not None:
            markdown = build_markdown_brief(result)
            out_path = DATA_DIR / f"brief_{result['run_id']}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")
            st.session_state["brief_markdown"] = markdown
            st.session_state["brief_run_id"] = result["run_id"]

    if st.session_state["brief_markdown"]:
        st.divider()
        st.download_button(
            "Download brief (.md)",
            data=st.session_state["brief_markdown"],
            file_name=f"brief_{st.session_state['brief_run_id']}.md",
            mime="text/markdown",
        )
        st.markdown(st.session_state["brief_markdown"])

with evidence_tab:
    st.caption("Provenance made visible — every claim traces to a real fetched page. Browse by run_id.")
    run_ids = _run_ids()
    if not run_ids:
        st.caption("No evidence in the database yet — run an analysis first.")
    else:
        default_index = 0
        if st.session_state["brief_run_id"] in run_ids:
            default_index = run_ids.index(st.session_state["brief_run_id"])
        selected_run_id = st.selectbox("run_id", options=run_ids, index=default_index)

        findings, facts = _evidence_for_run(selected_run_id)

        st.markdown("#### Findings")
        if findings.get("error"):
            st.error(findings["error"])
        elif not findings["rows"]:
            st.caption("No findings for this run.")
        else:
            for row in findings["rows"]:
                row["tier"] = source_tier(row.get("source_url"))
            st.dataframe(
                [{"claim": r["claim"], "stance": r["stance"], "source_name": r["source_name"],
                  "source_url": r["source_url"], "tier": r["tier"]} for r in findings["rows"]],
                width="stretch",
            )

        st.markdown("#### Facts")
        if facts.get("error"):
            st.error(facts["error"])
        elif not facts["rows"]:
            st.caption("No facts for this run.")
        else:
            for row in facts["rows"]:
                row["tier"] = source_tier(row.get("source_url"))
            st.dataframe(
                [{"entity": r["entity"], "metric": r["metric"], "value": r["value"], "unit": r["unit"],
                  "source_name": r["source_name"], "source_url": r["source_url"], "tier": r["tier"]} for r in facts["rows"]],
                width="stretch",
            )
