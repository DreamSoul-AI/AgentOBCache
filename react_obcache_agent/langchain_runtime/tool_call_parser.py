import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ParsedModelOutput:
    thought: str = ""
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None
    parse_error: Optional[str] = None
    raw_output: str = ""


def parse_tool_json(raw_output: str) -> ParsedModelOutput:
    raw_output = raw_output.strip()
    json_text = _extract_json_object(raw_output)

    if json_text is None:
        fallback = _regex_fallback(raw_output)
        if fallback.tool_calls:
            fallback.parse_error = "no_json_object_found_regex_tool_fallback"
            return fallback
        if fallback.final_answer and fallback.final_answer != raw_output:
            fallback.parse_error = "no_json_object_found_regex_final_answer_fallback"
            return fallback
        return ParsedModelOutput(
            final_answer=raw_output,
            parse_error="no_json_object_found",
            raw_output=raw_output,
        )

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        fallback = _regex_fallback(raw_output)
        fallback.parse_error = f"json_decode_error: {exc}"
        return fallback

    thought = str(data.get("thought", "")).strip()
    final_answer = data.get("final_answer")
    if final_answer is not None:
        return ParsedModelOutput(
            thought=thought,
            final_answer=_stringify_final_answer(final_answer),
            raw_output=raw_output,
        )

    tool_calls = []
    raw_tool_calls = data.get("tool_calls") or []
    if isinstance(raw_tool_calls, dict):
        raw_tool_calls = [raw_tool_calls]

    if not isinstance(raw_tool_calls, list):
        return ParsedModelOutput(
            thought=thought,
            final_answer=raw_output,
            parse_error="tool_calls_not_list",
            raw_output=raw_output,
        )

    for item in raw_tool_calls:
        call = _parse_tool_call_item(item)
        if call is not None:
            tool_calls.append(call)

    if not tool_calls:
        fallback = _regex_fallback(raw_output)
        if fallback.tool_calls:
            fallback.thought = thought or fallback.thought
            fallback.parse_error = "no_valid_tool_calls_json_regex_tool_fallback"
            return fallback
        return ParsedModelOutput(
            thought=thought,
            final_answer=raw_output,
            parse_error="no_valid_tool_calls_or_final_answer",
            raw_output=raw_output,
        )

    return ParsedModelOutput(
        thought=thought,
        tool_calls=tool_calls[:1],
        raw_output=raw_output,
    )


def _stringify_final_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        preferred_keys = [
            "answer",
            "final_answer",
            "author",
            "authors",
            "creator",
            "designer",
            "inventor",
            "other_device",
            "origin_of_python_name",
            "writers_directors",
            "result",
        ]
        if len(value) == 1:
            return _stringify_final_answer(next(iter(value.values())))
        for key in preferred_keys:
            if key in value:
                answer = _stringify_final_answer(value[key])
                if answer:
                    return answer
        parts = [_stringify_final_answer(item) for item in value.values()]
        return "; ".join(part for part in parts if part)
    if isinstance(value, list):
        parts = [_stringify_final_answer(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()
def _parse_tool_call_item(item) -> Optional[ParsedToolCall]:
    if not isinstance(item, dict):
        return None

    name = item.get("name") or item.get("tool") or item.get("tool_name")
    arguments = item.get("arguments") or item.get("args") or {}
    if isinstance(arguments, str):
        arguments = _coerce_argument_string(name, arguments)
    if name and isinstance(arguments, dict):
        return ParsedToolCall(name=str(name), arguments=arguments)
    return None


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _regex_fallback(text: str) -> ParsedModelOutput:
    final_answer = _extract_loose_final_answer(text)
    if final_answer:
        return ParsedModelOutput(final_answer=final_answer, raw_output=text)

    action_call = _parse_action_style_call(text)
    if action_call is not None:
        return ParsedModelOutput(tool_calls=[action_call], raw_output=text)

    name_match = re.search(r'"(?:name|tool|tool_name)"\s*:\s*"([a-zA-Z_][\w]*)"', text)
    if not name_match:
        return ParsedModelOutput(final_answer=text, raw_output=text)

    tool_name = name_match.group(1)
    args = {}
    for key in ["query", "keyword", "expression"]:
        value_match = re.search(rf'"{key}"\s*:\s*"(.*?)"', text, re.DOTALL)
        if value_match:
            args[key] = value_match.group(1).strip()

    return ParsedModelOutput(
        tool_calls=[ParsedToolCall(name=tool_name, arguments=args)],
        raw_output=text,
    )


def _extract_loose_final_answer(text: str) -> str:
    start_match = re.search(r'"final_answer"\s*:\s*"?', text)
    if not start_match:
        return ""

    answer = text[start_match.end():]
    end_match = re.search(r'"\s*[,}]?\s*$', answer, re.DOTALL)
    if end_match:
        answer = answer[:end_match.start()]

    return answer.strip().strip('"}')


def _parse_action_style_call(text: str) -> Optional[ParsedToolCall]:
    match = re.search(
        r'(?:Action\s*:\s*)?(wiki_search|wiki_lookup|paper_search|calculator)\s*\[(.*?)\]',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None

    name = match.group(1).strip()
    argument = match.group(2).strip()
    return ParsedToolCall(name=name, arguments=_coerce_argument_string(name, argument))


def _coerce_argument_string(name: Optional[str], value: str) -> Dict[str, str]:
    if name == "wiki_search":
        return {"query": value}
    if name == "wiki_lookup":
        return {"keyword": value}
    if name == "paper_search":
        return {"query": value}
    if name == "calculator":
        return {"expression": value}
    return {"input": value}
