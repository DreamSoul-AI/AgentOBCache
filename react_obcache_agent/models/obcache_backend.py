import torch

from models.base_backend import BaseModelBackend
from models.hf_backend import PyTorchHFBackend


class OBCacheBackend(BaseModelBackend):
    """Observable backend for OBCache-style experiments.

    This backend still delegates generation to the normal HuggingFace backend.
    It does not perform true low-level KV cache pruning yet. It records cache
    and pruning-shaped metrics so ReAct experiments can compare no-prune,
    window-prune, and segment-prune policies through one backend interface.
    """

    def __init__(self):
        self.backend = PyTorchHFBackend()
        self.cache_enabled = True
        self.cache_name = "obcache_observable"
        self.generation_count = 0
        self.pending_context_stats = None
        self.pending_segment_stats = None
        self.last_cache_stats = self._empty_cache_stats()

    def set_pruning_context(self, context_stats=None, segment_stats=None) -> None:
        self.pending_context_stats = context_stats or {}
        self.pending_segment_stats = segment_stats or {}

    def generate(self, prompt: str) -> str:
        prompt_tokens = self.count_tokens(prompt)
        pruned_tokens = self._get_pending_pruned_tokens()
        self.generation_count += 1

        output = self.backend.generate(prompt)

        self.last_cache_stats = {
            "cache_enabled": self.cache_enabled,
            "cache_name": self.cache_name,
            "cache_policy": self._get_cache_policy(),
            "generation_count": self.generation_count,
            "prompt_tokens": prompt_tokens,
            "cache_tokens": prompt_tokens,
            "pruned_tokens": pruned_tokens,
            "cache_hit": None,
            "cache_miss": None,
            "kv_cache_memory_mb": self._get_kv_cache_memory_mb(),
        }
        return output

    def count_tokens(self, text: str) -> int:
        return self.backend.count_tokens(text)

    def get_cache_stats(self) -> dict:
        return dict(self.last_cache_stats)

    def _get_pending_pruned_tokens(self) -> int:
        if not self.pending_context_stats:
            return 0
        return int(self.pending_context_stats.get("context_pruned_tokens") or 0)

    def _get_cache_policy(self) -> str:
        if self.pending_segment_stats:
            return self.pending_segment_stats.get("segment_policy", "segment_prune")
        if self.pending_context_stats and self.pending_context_stats.get("context_pruned_tokens", 0) > 0:
            return "context_prune"
        return "no_prune"

    def _empty_cache_stats(self) -> dict:
        return {
            "cache_enabled": self.cache_enabled,
            "cache_name": self.cache_name,
            "cache_policy": "no_prune",
            "generation_count": 0,
            "prompt_tokens": None,
            "cache_tokens": None,
            "pruned_tokens": 0,
            "cache_hit": None,
            "cache_miss": None,
            "kv_cache_memory_mb": None,
        }

    def _get_kv_cache_memory_mb(self):
        if not torch.cuda.is_available():
            return None

        return round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
