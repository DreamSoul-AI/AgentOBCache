import json
from typing import Any, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


SYSTEM_INSTRUCTION = """
You are a tool-calling research agent. You must use tools before giving a final answer unless the user explicitly asks only for arithmetic and calculator is available.
Always output exactly one JSON object and no extra text.

If there is no Tool result in the conversation yet, do not answer from memory. Call one tool first.
If the user asks to search, find, look up, or identify factual information, call wiki_search first.
If the user asks to calculate or compute, call calculator.
After receiving useful Tool result messages, either call another needed tool or provide final_answer.
Use paper_search for questions about papers, methods, publications, or their authors.
For multi-hop questions, gather evidence for every requested hop before answering.
Answer the requested entity only; do not substitute an intermediate entity from an earlier hop.
Every factual final answer must be directly supported by a Tool result.

Tool call JSON format:
{"thought":"brief reason","tool_calls":[{"name":"wiki_search","arguments":{"query":"Apple Remote"}}]}

Calculator JSON example:
{"thought":"I need to calculate the difference.","tool_calls":[{"name":"calculator","arguments":{"expression":"4096 - 1536"}}]}

Final answer JSON format:
{"thought":"The evidence is enough.","final_answer":"answer"}

Rules:
- Use at most one tool call per turn.
- Tool names must exactly match one of the available tool names.
- arguments must be a JSON object.
- Do not output markdown fences.
- Do not output explanatory text outside the JSON object.
""".strip()


def serialize_messages_with_tools(messages: List[Any], tools) -> Tuple[str, dict]:
    tool_schema_text = _serialize_tools(tools)
    message_text = _serialize_messages(messages)
    has_tool_result = any(isinstance(message, ToolMessage) for message in messages)

    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\nAvailable tools:\n"
        + tool_schema_text
        + "\n\nConversation:\n"
        + message_text
        + "\n\nCurrent requirement:\n"
        + _current_requirement(has_tool_result)
        + "\n\nReturn JSON now:"
    )

    metadata = {
        "tool_schema_chars": len(tool_schema_text),
        "tool_schema_text": tool_schema_text,
        "message_chars": len(message_text),
        "has_tool_result": has_tool_result,
        "segment_metadata": [
            {"type": "immutable_context", "name": "system_instruction", "protect": True},
            {"type": "immutable_context", "name": "tool_schema", "protect": True},
            {"type": "active_context", "name": "conversation", "protect": True},
        ],
    }
    return prompt, metadata


def _current_requirement(has_tool_result: bool) -> str:
    if has_tool_result:
        return (
            "You have tool evidence. If it is sufficient, return final_answer JSON. "
            "If not sufficient, call exactly one more tool using tool_calls JSON."
        )
    return "No tool has been called yet. You must call exactly one tool using tool_calls JSON."


def _serialize_tools(tools) -> str:
    chunks = []
    for tool_item in tools:
        schema = {}
        args_schema = getattr(tool_item, "args_schema", None)
        if args_schema is not None:
            try:
                schema = args_schema.model_json_schema()
            except Exception:
                schema = {}
        chunks.append(
            json.dumps(
                {
                    "name": tool_item.name,
                    "description": tool_item.description,
                    "args_schema": schema.get("properties", schema),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(chunks)


def _serialize_messages(messages: List[Any]) -> str:
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                lines.append(f"Assistant tool calls: {message.tool_calls}")
            else:
                lines.append(f"Assistant: {message.content}")
        elif isinstance(message, ToolMessage):
            lines.append(f"Tool result ({message.name}): {message.content}")
        elif isinstance(message, SystemMessage):
            lines.append(f"System: {message.content}")
        else:
            lines.append(str(message))
    return "\n".join(lines)
