from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph.message import add_messages


class ToolCallingState(TypedDict, total=False):
    messages: Annotated[List[Any], add_messages]
    task_id: str
    tool_logs: List[dict]
    cache_logs: List[dict]
    step_count: int
    tool_replay_mode: str
    tool_result_cache: Dict[str, str]
    method_name: str
    context_mode: str
    keep_last_tool_steps: int