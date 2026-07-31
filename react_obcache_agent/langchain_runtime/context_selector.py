from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


VALID_CONTEXT_MODES = {"full", "window", "segment", "obcache_v0"}


def select_messages(
    messages: List[Any],
    mode: str = "full",
    keep_last_tool_steps: int = 1,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Select prompt messages without changing the complete LangGraph state."""
    if mode not in VALID_CONTEXT_MODES:
        raise ValueError(
            f"Unknown tool context mode {mode!r}; expected one of "
            f"{sorted(VALID_CONTEXT_MODES)}"
        )

    keep_last_tool_steps = max(0, int(keep_last_tool_steps))
    full_tool_results = sum(isinstance(message, ToolMessage) for message in messages)

    if mode == "full":
        selected = list(messages)
        cached_tool_results = 0
        omitted_tool_results = 0
    elif mode == "window":
        selected = _select_window(messages, keep_last_tool_steps)
        selected_tool_results = sum(
            isinstance(message, ToolMessage) for message in selected
        )
        cached_tool_results = 0
        omitted_tool_results = full_tool_results - selected_tool_results
    else:
        use_placeholders = mode == "obcache_v0"
        selected, omitted_tool_results = _select_segment(
            messages,
            keep_last_tool_steps,
            use_placeholders=use_placeholders,
            keep_first_tool_result=mode == "obcache_v0",
        )
        cached_tool_results = omitted_tool_results if use_placeholders else 0

    metadata = {
        "context_mode": mode,
        "keep_last_tool_steps": keep_last_tool_steps,
        "full_message_count": len(messages),
        "selected_message_count": len(selected),
        "total_tool_results": full_tool_results,
        "kept_tool_results": full_tool_results - omitted_tool_results,
        "cached_tool_results": cached_tool_results,
        "omitted_tool_results": omitted_tool_results,
        "is_obcache_simulation": mode == "obcache_v0",
    }
    return selected, metadata


def _select_window(messages: List[Any], keep_last_tool_steps: int) -> List[Any]:
    prefix = [
        message
        for message in messages
        if isinstance(message, (HumanMessage, SystemMessage))
    ]
    tool_indices = [
        index for index, message in enumerate(messages) if isinstance(message, ToolMessage)
    ]

    if not tool_indices or keep_last_tool_steps <= 0:
        return prefix
    if len(tool_indices) <= keep_last_tool_steps:
        return list(messages)

    first_kept_tool_index = tool_indices[-keep_last_tool_steps]
    start_index = first_kept_tool_index
    if (
        first_kept_tool_index > 0
        and isinstance(messages[first_kept_tool_index - 1], AIMessage)
        and getattr(messages[first_kept_tool_index - 1], "tool_calls", None)
    ):
        start_index -= 1

    selected = list(prefix)
    for message in messages[start_index:]:
        if isinstance(message, (HumanMessage, SystemMessage)):
            continue
        selected.append(message)
    return selected


def _select_segment(
    messages: List[Any],
    keep_last_tool_steps: int,
    use_placeholders: bool,
    keep_first_tool_result: bool = False,
) -> Tuple[List[Any], int]:
    tool_indices = [
        index for index, message in enumerate(messages) if isinstance(message, ToolMessage)
    ]
    if keep_last_tool_steps <= 0:
        kept_indices = set()
    else:
        kept_indices = set(tool_indices[-keep_last_tool_steps:])
    if keep_first_tool_result and tool_indices:
        kept_indices.add(tool_indices[0])

    selected = []
    omitted_tool_results = 0
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage) or index in kept_indices:
            selected.append(message)
            continue

        omitted_tool_results += 1
        if use_placeholders:
            selected.append(
                ToolMessage(
                    content="[cached tool result omitted from prompt]",
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )

    return selected, omitted_tool_results