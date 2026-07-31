from dataclasses import asdict, dataclass
from typing import Optional

import time
import torch


@dataclass
class Action:
    name: str
    argument: str


@dataclass
class ReActStep:
    step: int
    action_name: Optional[str]
    action_argument: Optional[str]
    prompt_tokens: int
    output_tokens: int
    observation_tokens: int
    total_trajectory_tokens: int
    generation_time: float
    gpu_memory_allocated_mb: Optional[float]
    gpu_memory_reserved_mb: Optional[float]
    whether_action_valid: bool
    whether_finished: bool
    model_output: str
    observation: str
    cache_stats: Optional[dict] = None
    context_stats: Optional[dict] = None
    segment_stats: Optional[dict] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if data["cache_stats"] is None:
            data.pop("cache_stats")
        if data["context_stats"] is None:
            data.pop("context_stats")
        if data["segment_stats"] is None:
            data.pop("segment_stats")
        return data


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def get_gpu_memory_mb() -> dict:
    if not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
        }

    return {
        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
        "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
    }


def now_time() -> float:
    return time.time()


def make_step_log(
    step: int,
    model_output: str,
    action: Optional[Action],
    observation: str,
    prompt_tokens: int,
    output_tokens: int,
    observation_tokens: int,
    total_trajectory_tokens: int,
    generation_time: float,
    whether_action_valid: bool,
    whether_finished: bool,
    cache_stats: Optional[dict] = None,
    context_stats: Optional[dict] = None,
    segment_stats: Optional[dict] = None,
) -> dict:
    gpu_memory = get_gpu_memory_mb()
    step_log = ReActStep(
        step=step,
        action_name=action.name if action else None,
        action_argument=action.argument if action else None,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        observation_tokens=observation_tokens,
        total_trajectory_tokens=total_trajectory_tokens,
        generation_time=round(generation_time, 4),
        gpu_memory_allocated_mb=gpu_memory["gpu_memory_allocated_mb"],
        gpu_memory_reserved_mb=gpu_memory["gpu_memory_reserved_mb"],
        whether_action_valid=whether_action_valid,
        whether_finished=whether_finished,
        model_output=model_output,
        observation=observation,
        cache_stats=cache_stats,
        context_stats=context_stats,
        segment_stats=segment_stats,
    )
    return step_log.to_dict()
