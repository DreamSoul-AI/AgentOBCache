class SegmentContextManager:
    """Segment-aware context manager for OBCache-style simulation.

    The trajectory is represented as alternating items:
    ("model", "Thought... Action...") and ("observation", "Observation: ...").

    This manager keeps every model Thought/Action item, keeps recent observations
    verbatim, and replaces older observations with compact cache placeholders.
    It simulates the idea that old long observations can live in an external or
    KV-cache-like segment instead of being fully repeated in the prompt.
    """

    def __init__(self, keep_recent_observations: int = 1):
        self.keep_recent_observations = keep_recent_observations
        self.last_segment_stats = {}

    def select(self, trajectory):
        observation_indices = [
            index for index, (role, _) in enumerate(trajectory)
            if role == "observation"
        ]
        if self.keep_recent_observations <= 0:
            kept_observation_indices = set()
        else:
            kept_observation_indices = set(observation_indices[-self.keep_recent_observations:])

        selected = []
        kept_segments = 0
        cached_segments = 0
        omitted_segments = 0

        for index, (role, content) in enumerate(trajectory):
            if role == "model":
                selected.append((role, content))
                kept_segments += 1
                continue

            if role == "observation" and index in kept_observation_indices:
                selected.append((role, content))
                kept_segments += 1
                continue

            if role == "observation":
                cached_segments += 1
                omitted_segments += 1
                selected.append((
                    role,
                    self._make_cached_observation_placeholder(index),
                ))
                continue

            selected.append((role, content))
            kept_segments += 1

        self.last_segment_stats = {
            "segment_context_enabled": True,
            "segment_policy": "keep_all_model_keep_recent_observations",
            "keep_recent_observations": self.keep_recent_observations,
            "total_segments": len(trajectory),
            "kept_segments": kept_segments,
            "cached_segments": cached_segments,
            "omitted_segments": omitted_segments,
        }
        return selected

    def get_segment_stats(self) -> dict:
        return dict(self.last_segment_stats)

    def _make_cached_observation_placeholder(self, index: int) -> str:
        return "Observation: [cached]"
