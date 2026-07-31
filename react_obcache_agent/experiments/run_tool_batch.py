import ast
import json
import os
import re
import sys
import time
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import HumanMessage

from langchain_runtime.graph import build_graph
from langchain_runtime.model_node import initialize_model_backend
from tools.lc_tools import reset_tool_state


DATA_PATH = "data/tool_tasks.jsonl"
OUTPUT_PATH = "logs/tool_batch_results_langgraph_json.jsonl"
SUMMARY_PATH = "logs/tool_batch_summary.json"
TOOL_RESULT_CACHE_PATH = "logs/tool_result_cache.json"
QUESTION_LIMIT_TEXT = os.environ.get("QUESTION_LIMIT", "").strip()
QUESTION_LIMIT = int(QUESTION_LIMIT_TEXT) if QUESTION_LIMIT_TEXT else None
TOOL_REPLAY_MODE = os.environ.get("TOOL_REPLAY_MODE", "replay").strip().lower()
TOOL_METHODS_TEXT = os.environ.get("TOOL_METHODS", "").strip()
TASK_IDS_TEXT = os.environ.get("TASK_IDS", "").strip()

# OBCache v0 is a prompt-level simulation. It does not prune past_key_values yet.
METHOD_CONFIGS = [
    {
        "name": "full_tool_react",
        "context_mode": "full",
        "keep_last_tool_steps": 0,
    },
    {
        "name": "window_tool_react_1",
        "context_mode": "window",
        "keep_last_tool_steps": 1,
    },
    {
        "name": "window_tool_react_2",
        "context_mode": "window",
        "keep_last_tool_steps": 2,
    },
    {
        "name": "window_tool_react_3",
        "context_mode": "window",
        "keep_last_tool_steps": 3,
    },
    {
        "name": "segment_tool_react",
        "context_mode": "segment",
        "keep_last_tool_steps": 1,
    },
    {
        "name": "obcache_tool_react_v0",
        "context_mode": "obcache_v0",
        "keep_last_tool_steps": 1,
    },
]


def load_tasks(path: str):
    tasks = []
    with open(path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as error:
                snippet = line.strip()[:120]
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}. "
                    f"Line starts with: {snippet!r}"
                ) from error
    return tasks


def load_tool_result_cache(path: str) -> dict:
    if TOOL_REPLAY_MODE == "live" or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def save_tool_result_cache(path: str, cache: dict) -> None:
    if TOOL_REPLAY_MODE != "record":
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2, sort_keys=True)


def select_method_configs():
    if not TOOL_METHODS_TEXT:
        return METHOD_CONFIGS

    requested = [name.strip() for name in TOOL_METHODS_TEXT.split(",") if name.strip()]
    configs_by_name = {config["name"]: config for config in METHOD_CONFIGS}
    unknown = [name for name in requested if name not in configs_by_name]
    if unknown:
        raise ValueError(
            f"Unknown TOOL_METHODS values: {unknown}. "
            f"Available methods: {list(configs_by_name)}"
        )
    return [configs_by_name[name] for name in requested]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> bool:
    answer = normalize_answer(answer)
    if not answer:
        return False
    return normalize_answer(prediction) == answer


def contains_match(prediction: str, answer: str) -> bool:
    answer = normalize_answer(answer)
    if not answer:
        return False
    return answer in normalize_answer(prediction)


def token_f1(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def summarize(results, config):
    total = len(results)
    if total == 0:
        return {}

    cache_logs = [log for item in results for log in item["cache_logs"]]
    tool_logs = [log for item in results for log in item["tool_logs"]]
    model_steps = len(cache_logs)
    successful_tool_calls = sum(1 for log in tool_logs if log.get("success"))
    parse_errors = sum(1 for log in cache_logs if log.get("parse_error"))
    forced_steps = sum(1 for log in cache_logs if log.get("forced_tool_call"))
    source_counts = Counter(
        item.get("answer_source", "unfinished")
        for item in results
        if item.get("finished")
    )
    total_full_prompt_tokens = sum(
        log.get("full_prompt_tokens", 0) for log in cache_logs
    )
    total_pruned_tokens = sum(
        log.get("context_pruned_tokens", 0) for log in cache_logs
    )

    return {
        "num_tasks": total,
        "tool_replay_mode": TOOL_REPLAY_MODE,
        "context_mode": config["context_mode"],
        "keep_last_tool_steps": config["keep_last_tool_steps"],
        "obcache_status": (
            "prompt_context_simulation_only"
            if config["context_mode"] == "obcache_v0"
            else None
        ),
        "exact_match_accuracy": _rate(results, "exact_match_correct"),
        "contains_match_accuracy": _rate(results, "contains_match_correct"),
        "avg_token_f1": _avg(item.get("token_f1") for item in results),
        "runtime_finish_rate": _rate(results, "finished"),
        "true_model_finish_rate": _rate(results, "true_model_finished"),
        "evidence_extractor_finish_rate": _rate(results, "evidence_extractor_finished"),
        "max_step_stop_rate": _rate(results, "stopped_with_pending_tool_call"),
        "tool_execution_success_rate": (
            successful_tool_calls / len(tool_logs) if tool_logs else None
        ),
        "tool_selection_exact_match_rate": _rate(
            results, "tool_selection_exact_match"
        ),
        "tool_selection_set_match_rate": _rate(results, "tool_selection_set_match"),
        "avg_tool_calls_per_task": round(len(tool_logs) / total, 2),
        "raw_parse_success_rate": (
            (model_steps - parse_errors) / model_steps if model_steps else None
        ),
        "tool_call_parse_error_rate": (
            parse_errors / model_steps if model_steps else None
        ),
        "forced_tool_call_rate": forced_steps / model_steps if model_steps else None,
        "replay_cache_hit_rate": _avg_bool(log.get("was_cached") for log in tool_logs),
        "answer_source_counts": dict(source_counts),
        "tool_result_cache_entries": _cache_entry_count(results),
        "avg_prompt_tokens": _avg(log.get("prompt_tokens") for log in cache_logs),
        "avg_full_prompt_tokens": _avg(
            log.get("full_prompt_tokens") for log in cache_logs
        ),
        "avg_selected_prompt_tokens": _avg(
            log.get("selected_prompt_tokens") for log in cache_logs
        ),
        "avg_context_pruned_tokens": _avg(
            log.get("context_pruned_tokens") for log in cache_logs
        ),
        "avg_context_pruned_ratio": _avg(
            (log.get("context_pruned_ratio") for log in cache_logs),
            digits=4,
        ),
        "total_context_pruned_ratio": (
            round(total_pruned_tokens / total_full_prompt_tokens, 4)
            if total_full_prompt_tokens
            else 0.0
        ),
        "avg_output_tokens": _avg(log.get("output_tokens") for log in cache_logs),
        "avg_generation_time": _avg(
            (log.get("generation_time") for log in cache_logs),
            digits=4,
        ),
        "avg_generation_time_per_question": _avg(
            sum(log.get("generation_time", 0.0) for log in item["cache_logs"])
            for item in results
        ),
        "avg_tool_latency_per_question": _avg(
            sum(log.get("latency_seconds", 0.0) for log in item["tool_logs"])
            for item in results
        ),
        "avg_end_to_end_time_per_question": _avg(
            item.get("end_to_end_time") for item in results
        ),
        "avg_gpu_memory_allocated_mb": _avg(
            log.get("gpu_memory_allocated_mb") for log in cache_logs
        ),
        "avg_gpu_memory_reserved_mb": _avg(
            log.get("gpu_memory_reserved_mb") for log in cache_logs
        ),
        "avg_full_message_count": _avg(
            log.get("full_message_count") for log in cache_logs
        ),
        "avg_selected_message_count": _avg(
            log.get("selected_message_count") for log in cache_logs
        ),
        "avg_total_tool_results": _avg(
            log.get("total_tool_results") for log in cache_logs
        ),
        "avg_kept_tool_results": _avg(
            log.get("kept_tool_results") for log in cache_logs
        ),
        "avg_cached_tool_results": _avg(
            log.get("cached_tool_results") for log in cache_logs
        ),
        "avg_omitted_tool_results": _avg(
            log.get("omitted_tool_results") for log in cache_logs
        ),
        "avg_tool_result_tokens_approx": _avg(
            log.get("result_tokens_approx") for log in tool_logs
        ),
    }


def _rate(results, key: str):
    if not results:
        return None
    return sum(1 for item in results if item.get(key)) / len(results)


def _avg(values, digits: int = 2):
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), digits)


def _avg_bool(values):
    values = [value for value in values if isinstance(value, bool)]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def _cache_entry_count(results):
    for item in reversed(results):
        count = item.get("tool_result_cache_entries")
        if isinstance(count, int):
            return count
    return None


def run_method(graph, config, tasks, out, tool_result_cache):
    method_name = config["name"]
    results = []
    print(
        f"\n=== {method_name} | context={config['context_mode']} "
        f"keep_last_tool_steps={config['keep_last_tool_steps']} tasks={len(tasks)} ==="
    )

    for task in tasks:
        reset_tool_state()
        question_start = time.perf_counter()
        final_state = graph.invoke(
            {
                "task_id": task["id"],
                "method_name": method_name,
                "context_mode": config["context_mode"],
                "keep_last_tool_steps": config["keep_last_tool_steps"],
                "messages": [HumanMessage(content=task["question"])],
                "tool_logs": [],
                "cache_logs": [],
                "step_count": 0,
                "tool_replay_mode": TOOL_REPLAY_MODE,
                "tool_result_cache": tool_result_cache,
            }
        )
        end_to_end_time = time.perf_counter() - question_start
        tool_result_cache = final_state.get("tool_result_cache", tool_result_cache)
        save_tool_result_cache(TOOL_RESULT_CACHE_PATH, tool_result_cache)

        messages = final_state.get("messages", [])
        last_message_content = _clean_answer_text(messages[-1].content) if messages else ""
        finished = _is_final_answer_message(messages)
        prediction = last_message_content if finished else ""
        expected_tools = task.get("expected_tools", [])
        actual_tools = [
            log.get("tool_name") for log in final_state.get("tool_logs", [])
        ]
        answer_source = _answer_source(final_state.get("cache_logs", []), finished)
        stopped_with_pending_tool_call = _has_pending_tool_call(messages)
        gold_answer = task.get("answer", "")

        result = {
            "method": method_name,
            "context_mode": config["context_mode"],
            "keep_last_tool_steps": config["keep_last_tool_steps"],
            "id": task["id"],
            "question": task["question"],
            "gold_answer": gold_answer,
            "prediction": prediction,
            "last_message_content": last_message_content,
            "finished": finished,
            "finish_reason": (
                "final_answer" if finished else _unfinished_reason(messages)
            ),
            "answer_source": answer_source,
            "true_model_finished": finished and answer_source == "model",
            "evidence_extractor_finished": (
                finished and answer_source == "evidence_extractor"
            ),
            "stopped_with_pending_tool_call": stopped_with_pending_tool_call,
            "exact_match_correct": finished and exact_match(prediction, gold_answer),
            "contains_match_correct": (
                finished and contains_match(prediction, gold_answer)
            ),
            "token_f1": token_f1(prediction, gold_answer) if finished else 0.0,
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "tool_selection_exact_match": actual_tools == expected_tools,
            "tool_selection_set_match": set(actual_tools) == set(expected_tools),
            "end_to_end_time": round(end_to_end_time, 4),
            "tool_logs": final_state.get("tool_logs", []),
            "cache_logs": final_state.get("cache_logs", []),
            "tool_result_cache_entries": len(tool_result_cache),
        }
        results.append(result)
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            method_name,
            task["id"],
            prediction or f"<unfinished: {result['finish_reason']}>",
        )

    return results, tool_result_cache


def warmup_runtime(graph, tasks, tool_result_cache):
    if TOOL_REPLAY_MODE != "replay" or not tasks:
        return tool_result_cache

    reset_tool_state()
    task = tasks[0]
    final_state = graph.invoke(
        {
            "task_id": "runtime_warmup",
            "method_name": "runtime_warmup",
            "context_mode": "full",
            "keep_last_tool_steps": 0,
            "messages": [HumanMessage(content=task["question"])],
            "tool_logs": [],
            "cache_logs": [],
            "step_count": 0,
            "tool_replay_mode": TOOL_REPLAY_MODE,
            "tool_result_cache": tool_result_cache,
        }
    )
    return final_state.get("tool_result_cache", tool_result_cache)


def main():
    if TOOL_REPLAY_MODE not in {"live", "record", "replay"}:
        raise ValueError("TOOL_REPLAY_MODE must be one of: live, record, replay")

    os.chdir(PROJECT_ROOT)
    tasks = load_tasks(DATA_PATH)
    if TASK_IDS_TEXT:
        requested_ids = {task_id.strip() for task_id in TASK_IDS_TEXT.split(",") if task_id.strip()}
        tasks = [task for task in tasks if task.get("id") in requested_ids]
        missing_ids = sorted(requested_ids - {task.get("id") for task in tasks})
        if missing_ids:
            raise ValueError(f"Unknown TASK_IDS values: {missing_ids}")
    if QUESTION_LIMIT is not None:
        tasks = tasks[:QUESTION_LIMIT]

    method_configs = select_method_configs()
    print(f"Loaded {len(tasks)} tasks from {DATA_PATH}")
    print(f"Tool replay mode: {TOOL_REPLAY_MODE}")
    if QUESTION_LIMIT is not None:
        print(f"Question limit: {QUESTION_LIMIT}")
    if TOOL_METHODS_TEXT:
        print(f"TOOL_METHODS filter: {TOOL_METHODS_TEXT}")
    if TASK_IDS_TEXT:
        print(f"TASK_IDS filter: {TASK_IDS_TEXT}")
    print("Methods: " + ", ".join(config["name"] for config in method_configs))
    graph = build_graph()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tool_result_cache = load_tool_result_cache(TOOL_RESULT_CACHE_PATH)
    tool_result_cache = warmup_runtime(graph, tasks, tool_result_cache)
    summaries = {}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for config in method_configs:
            results, tool_result_cache = run_method(
                graph,
                config,
                tasks,
                out,
                tool_result_cache,
            )
            summaries[config["name"]] = summarize(results, config)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for method_name, summary in summaries.items():
        print(
            f"{method_name}: "
            f"exact={summary['exact_match_accuracy']:.4f}, "
            f"contains={summary['contains_match_accuracy']:.4f}, "
            f"prompt={summary['avg_prompt_tokens']:.2f}, "
            f"pruned={summary['total_context_pruned_ratio']:.4f}, "
            f"gen/q={summary['avg_generation_time_per_question']:.2f}s, "
            f"e2e/q={summary['avg_end_to_end_time_per_question']:.2f}s"
        )
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Detailed results written to {OUTPUT_PATH}")


def _clean_answer_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return _answer_value_to_text(value)
    if isinstance(value, list):
        return "; ".join(_clean_answer_text(item) for item in value if item is not None)

    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, (dict, list)):
            return _clean_answer_text(parsed)
    return text


def _answer_value_to_text(value: dict) -> str:
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
        return _clean_answer_text(next(iter(value.values())))
    for key in preferred_keys:
        if key in value:
            answer = _clean_answer_text(value[key])
            if answer:
                return answer
    return "; ".join(_clean_answer_text(item) for item in value.values() if item is not None)
def _is_final_answer_message(messages) -> bool:
    if not messages:
        return False
    last_message = messages[-1]
    if last_message.__class__.__name__ != "AIMessage":
        return False
    if getattr(last_message, "tool_calls", None):
        return False
    return bool(str(last_message.content).strip())


def _has_pending_tool_call(messages) -> bool:
    if not messages:
        return False
    return bool(getattr(messages[-1], "tool_calls", None))


def _unfinished_reason(messages) -> str:
    if not messages:
        return "empty_messages"
    if _has_pending_tool_call(messages):
        return "max_steps_with_pending_tool_call"
    if messages[-1].__class__.__name__ == "ToolMessage":
        return "ended_after_tool_result"
    return "empty_final_answer"


def _answer_source(cache_logs, finished: bool) -> str:
    if not finished:
        return "unfinished"
    for log in reversed(cache_logs):
        if log.get("has_final_answer"):
            return log.get("answer_source") or "unknown"
    return "unknown"


if __name__ == "__main__":
    main()