from langgraph.graph import END, START, StateGraph

from langchain_runtime.model_node import call_model
from langchain_runtime.state import ToolCallingState
from langchain_runtime.tool_node import execute_tools


MAX_TOOL_STEPS = 8


def route_after_model(state):
    messages = state.get("messages", [])
    if not messages:
        return END

    if state.get("step_count", 0) >= MAX_TOOL_STEPS:
        return END

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END


def build_graph():
    graph = StateGraph(ToolCallingState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_model,
        {
            "tools": "tools",
            END: END,
        },
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
