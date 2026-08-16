"""
Conditional edge functions.

route_from_manager is the only one: the engagement manager has already
decided who works next and written it to state["next_agent"], so this
just translates that decision into a destination LangGraph understands.

The pm gate lives in engagement_manager.py, not here — one place for
routing policy, not two.
"""
from langgraph.graph import END
from state import EngagementState


def route_from_manager(state: EngagementState) -> str:
    """
    Input:  EngagementState, post engagement_manager_node
    Output: a registered node name, or END
    Reads:  state["next_agent"]
    """
    nxt = state["next_agent"]
    return END if nxt == "FINISH" else nxt