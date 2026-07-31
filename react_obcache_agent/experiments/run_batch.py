import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.react_agent import ReActAgent
from memory.full_context import FullContextManager
from memory.segment_context import SegmentContextManager
from memory.window_context import WindowContextManager
from models.hf_backend import PyTorchHFBackend
from models.obcache_backend import OBCacheBackend


DATA_PATH = "data/questions.jsonl"
QUESTION_LIMIT = None
BASELINE_CONFIGS = [
    {
        "name": "full_react",
        "context_factory": FullContextManager,
        "model_factory": PyTorchHFBackend,
        "output_path": "logs/batch_results_full_react.jsonl",
    },
    {
        "name": "window_react_1",
        "context_factory": lambda: WindowContextManager(keep_last_steps=1),
        "model_factory": PyTorchHFBackend,
        "output_path": "logs/batch_results_window_1.jsonl",
    },
    {
        "name": "window_react_2",
        "context_factory": lambda: WindowContextManager(keep_last_steps=2),
        "model_factory": PyTorchHFBackend,
        "output_path": "logs/batch_results_window_2.jsonl",
    },
    {
        "name": "window_react_3",
        "context_factory": lambda: WindowContextManager(keep_last_steps=3),
        "model_factory": PyTorchHFBackend,
        "output_path": "logs/batch_results_window_3.jsonl",
    },
    {
        "name": "obcache_react_no_prune",
        "context_factory": FullContextManager,
        "model_factory": OBCacheBackend,
        "output_path": "logs/batch_results_obcache_no_prune.jsonl",
    },
    {
        "name": "obcache_react_context_prune_3",
        "context_factory": lambda: WindowContextManager(keep_last_steps=3),
        "model_factory": OBCacheBackend,
        "output_path": "logs/batch_results_obcache_context_prune_3.jsonl",
    },
    {
        "name": "obcache_react_segment_keep_obs_0",
        "context_factory": lambda: SegmentContextManager(keep_recent_observations=0),
        "model_factory": OBCacheBackend,
        "output_path": "logs/batch_results_obcache_segment_keep_obs_0.jsonl",
    },
    {
        "name": "obcache_react_segment_keep_obs_1",
        "context_factory": lambda: SegmentContextManager(keep_recent_observations=1),
        "model_factory": OBCacheBackend,
        "output_path": "logs/batch_results_obcache_segment_keep_obs_1.jsonl",
    },
    {
        "name": "obcache_react_segment_keep_obs_2",
        "context_factory": lambda: SegmentContextManager(keep_recent_observations=2),
        "model_factory": OBCacheBackend,
        "output_path": "logs/batch_results_obcache_segment_keep_obs_2.jsonl",
    },
    {
        "name": "obcache_react_segment_prune",
        "context_factory": lambda: SegmentContextManager(keep_recent_observations=1),
        "model_factory": OBCacheBackend,
        "output_path": "logs/batch_results_obcache_segment_prune.jsonl",
    },
]


def load_questions(path: str):
    questions = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_exact_match(prediction: str, answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


def is_contains_match(prediction: str, answer: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_answer = normalize_answer(answer)

    if not normalized_answer:
        return False

    return normalized_answer in normalized_prediction


def _average_numeric(values, digits: int = 2):
    numeric_values = [value for value in values if isinstance(value, (int, float))]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), digits)


def summarize(results):
    total = len(results)
    exact_matches = sum(1 for item in results if item["is_exact_match"])
    contains_matches = sum(1 for item in results if item["is_contains_match"])
    finished = sum(1 for item in results if item["is_finished"])
    all_step_logs = [log for item in results for log in item["logs"]]
    valid_logs = [log for log in all_step_logs if log.get("prompt_tokens") is not None]
    cache_logs = [log["cache_stats"] for log in all_step_logs if log.get("cache_stats")]
    context_logs = [log["context_stats"] for log in all_step_logs if log.get("context_stats")]
    segment_logs = [log["segment_stats"] for log in all_step_logs if log.get("segment_stats")]

    if not valid_logs:
        return {
            "exact_match": exact_matches / total if total else 0,
            "contains_match": contains_matches / total if total else 0,
            "finish_rate": finished / total if total else 0,
            "avg_prompt_tokens": None,
            "avg_output_tokens": None,
            "avg_generation_time": None,
            "avg_generation_time_per_question": None,
            "avg_steps_per_question": None,
            "action_format_error_rate": None,
            "avg_cache_tokens": None,
            "avg_pruned_tokens": None,
            "avg_kv_cache_memory_mb": None,
            "avg_full_prompt_tokens": None,
            "avg_selected_prompt_tokens": None,
            "avg_context_pruned_tokens": None,
            "avg_context_pruned_ratio": None,
            "avg_total_segments": None,
            "avg_kept_segments": None,
            "avg_cached_segments": None,
            "avg_omitted_segments": None,
        }

    action_errors = sum(1 for log in all_step_logs if not log["whether_action_valid"])
    generation_time_by_question = [
        sum(log["generation_time"] for log in item["logs"])
        for item in results
    ]
    return {
        "exact_match": exact_matches / total if total else 0,
        "contains_match": contains_matches / total if total else 0,
        "finish_rate": finished / total if total else 0,
        "avg_prompt_tokens": round(
            sum(log["prompt_tokens"] for log in valid_logs) / len(valid_logs),
            2,
        ),
        "avg_output_tokens": round(
            sum(log["output_tokens"] for log in valid_logs) / len(valid_logs),
            2,
        ),
        "avg_generation_time": round(
            sum(log["generation_time"] for log in valid_logs) / len(valid_logs),
            4,
        ),
        "avg_generation_time_per_question": round(
            sum(generation_time_by_question) / total,
            4,
        ) if total else 0,
        "avg_steps_per_question": round(
            sum(len(item["logs"]) for item in results) / total,
            2,
        ) if total else 0,
        "action_format_error_rate": round(
            action_errors / len(all_step_logs),
            4,
        ) if all_step_logs else 0,
        "avg_cache_tokens": _average_numeric(
            stats.get("cache_tokens") for stats in cache_logs
        ),
        "avg_pruned_tokens": _average_numeric(
            stats.get("pruned_tokens") for stats in cache_logs
        ),
        "avg_kv_cache_memory_mb": _average_numeric(
            stats.get("kv_cache_memory_mb") for stats in cache_logs
        ),
        "avg_full_prompt_tokens": _average_numeric(
            stats.get("full_prompt_tokens") for stats in context_logs
        ),
        "avg_selected_prompt_tokens": _average_numeric(
            stats.get("selected_prompt_tokens") for stats in context_logs
        ),
        "avg_context_pruned_tokens": _average_numeric(
            stats.get("context_pruned_tokens") for stats in context_logs
        ),
        "avg_context_pruned_ratio": _average_numeric(
            (stats.get("context_pruned_ratio") for stats in context_logs),
            digits=4,
        ),
        "avg_total_segments": _average_numeric(
            stats.get("total_segments") for stats in segment_logs
        ),
        "avg_kept_segments": _average_numeric(
            stats.get("kept_segments") for stats in segment_logs
        ),
        "avg_cached_segments": _average_numeric(
            stats.get("cached_segments") for stats in segment_logs
        ),
        "avg_omitted_segments": _average_numeric(
            stats.get("omitted_segments") for stats in segment_logs
        ),
    }


def run_baseline(config: dict, questions):
    name = config["name"]
    output_path = config["output_path"]
    model = config["model_factory"]()
    results = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for item in questions:
            agent = ReActAgent(
                model=model,
                context_manager=config["context_factory"](),
                log_path=None,
            )
            prediction = agent.run(item["question"])
            gold_answer = item.get("answer", "")
            finished = any(log["whether_finished"] for log in agent.logs)

            result = {
                "baseline": name,
                "id": item["id"],
                "question": item["question"],
                "gold_answer": gold_answer,
                "prediction": prediction,
                "is_exact_match": is_exact_match(prediction, gold_answer),
                "is_contains_match": is_contains_match(prediction, gold_answer),
                "is_finished": finished,
                "logs": agent.logs,
            }

            cache_stats = getattr(model, "get_cache_stats", None)
            if cache_stats is not None:
                result["cache_stats"] = cache_stats()

            results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(name, item["id"], prediction)

    summary = summarize(results)
    print(name, summary)
    return summary


def main():
    os.chdir(PROJECT_ROOT)
    questions = load_questions(DATA_PATH)
    if QUESTION_LIMIT is not None:
        questions = questions[:QUESTION_LIMIT]

    summaries = {}
    for config in BASELINE_CONFIGS:
        summaries[config["name"]] = run_baseline(config, questions)

    summary_path = "logs/batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


