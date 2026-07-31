import json
import time

from langchain_core.messages import ToolMessage

from tools.lc_tools import TOOL_BY_NAME


_FAILURE_MARKERS = [
    "could not find",
    "no current page",
    "calculator error",
    "tool execution error",
]


def execute_tools(state):
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    tool_messages = []
    tool_logs = list(state.get("tool_logs", []))
    replay_mode = state.get("tool_replay_mode", "live")
    tool_result_cache = state.get("tool_result_cache") or {}

    for tool_call in tool_calls:
        name = tool_call.get("name")
        args = tool_call.get("args") or {}
        call_id = tool_call.get("id") or f"call_{len(tool_logs)}"
        cache_key = _cache_key(name, args)
        start_time = time.time()
        success = True
        was_cached = False
        refreshed_tool_state = False

        cached_result = tool_result_cache.get(cache_key)
        has_usable_cache = cached_result is not None and not _looks_like_failed_result(str(cached_result))

        if replay_mode in {"record", "replay"} and has_usable_cache:
            result = cached_result
            latency = 0.0
            was_cached = True

            if replay_mode == "record" and name == "wiki_search":
                refreshed_tool_state = _refresh_stateful_tool(name, args)
        elif replay_mode == "replay":
            success = False
            result = f"Tool execution error for {name}: missing replay cache key {cache_key}"
            latency = 0.0
        else:
            result, success, latency = _invoke_tool(name, args, start_time)

        result_text = str(result)
        if _looks_like_failed_result(result_text):
            success = False

        if replay_mode == "record" and not was_cached and success:
            tool_result_cache[cache_key] = result_text

        tool_messages.append(
            ToolMessage(
                content=result_text,
                tool_call_id=call_id,
                name=name,
            )
        )
        tool_logs.append(
            {
                "step": len(tool_logs) + 1,
                "tool_name": name,
                "arguments": args,
                "success": success,
                "latency_seconds": round(latency, 4),
                "result_chars": len(result_text),
                "result_tokens_approx": len(result_text.split()),
                "result_preview": result_text[:300],
                "was_cached": was_cached,
                "refreshed_tool_state": refreshed_tool_state,
                "replay_mode": replay_mode,
                "cache_key": cache_key,
            }
        )

    return {
        "messages": tool_messages,
        "tool_logs": tool_logs,
        "tool_result_cache": tool_result_cache,
    }


def _invoke_tool(name: str, args: dict, start_time: float):
    try:
        tool_item = TOOL_BY_NAME[name]
        result = tool_item.invoke(args)
        success = True
    except Exception as exc:
        result = f"Tool execution error for {name}: {exc}"
        success = False
    return result, success, time.time() - start_time


def _refresh_stateful_tool(name: str, args: dict) -> bool:
    try:
        tool_item = TOOL_BY_NAME[name]
        result = str(tool_item.invoke(args))
    except Exception:
        return False
    return not _looks_like_failed_result(result)


def _cache_key(name: str, args: dict) -> str:
    return json.dumps(
        {"tool": name, "args": args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _looks_like_failed_result(result_text: str) -> bool:
    lowered = result_text.lower()
    return any(marker in lowered for marker in _FAILURE_MARKERS)