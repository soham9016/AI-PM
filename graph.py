"""Graph assembly: wires the agent nodes + routing into a compiled
LangGraph StateGraph — flat supervisor pattern, every worker returns to
engagement_manager, which decides what runs next (or FINISH -> END).

WORKERS is the single source of truth for which nodes exist: node
registration, the worker-back-to-EM edges, and the conditional-edges
destination mapping are all built generically from WORKERS.items()/keys
below, so adding a new agent (e.g. the three PM-path agents —
primary_research, competitive_audit, solution_framing) only ever means
adding one entry to this dict, an import, and the corresponding routing
logic in agents/engagement_manager.py — nothing here needs to change
node-by-node.

Callers do:
    app = build_graph()
    result = app.invoke(new_state("..."))

build_graph() also calls config.validate_models() -- catches a stale/
deprecated/nonexistent model name in config.py's MODELS at launch,
not three agents into a run (see config.py's module docstring for why:
four distinct provider failures in three days, this one specifically
answering "Groq deprecated a model out from under us"). Cached after the
first call in a process, so this doesn't add a network round-trip on
every graph build (e.g. the Streamlit app rebuilding the graph on every
click).
"""
from langgraph.graph import StateGraph, END

from config import validate_models
from state import EngagementState
from routing import route_from_manager
from agents.engagement_manager import engagement_manager_node
from agents.structurer import structurer_node
from agents.hypothesis import hypothesis_node
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.synthesizer import synthesizer_node
from agents.funnel_decomposition import funnel_decomposition_node
from agents.primary_research import primary_research_node
from agents.competitive_audit import competitive_audit_node
from agents.solution_framing import solution_framing_node
from agents.pm import pm_node
from agents.solution_review import solution_review_node
from agents.critic import critic_node

WORKERS = {
    "structurer":  structurer_node,
    "hypothesis":  hypothesis_node,
    "researcher":  researcher_node,
    "analyst":     analyst_node,
    "synthesizer": synthesizer_node,
    "funnel_decomposition": funnel_decomposition_node,
    "primary_research":  primary_research_node,
    "competitive_audit": competitive_audit_node,
    "solution_framing":  solution_framing_node,
    "pm":          pm_node,
    "solution_review": solution_review_node,
    "critic":      critic_node,
}

def build_graph(checkpointer=None):
    validate_models()
    graph = StateGraph(EngagementState)
    graph.add_node("engagement_manager", engagement_manager_node)
    for name, fn in WORKERS.items():
        graph.add_node(name, fn)
        graph.add_edge(name, "engagement_manager")
    graph.set_entry_point("engagement_manager")

    # Manager decides: routing.py names the destination
    graph.add_conditional_edges(
        "engagement_manager",
        route_from_manager,
        {**{name: name for name in WORKERS}, END: END},
    )

    return graph.compile(checkpointer=checkpointer)
