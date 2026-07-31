import json
import os
from typing import List, Optional, Tuple

from agent.action_parser import ActionParser
from agent.prompt_builder import PromptBuilder
from agent.trajectory import Action, make_step_log, now_time
from memory.full_context import FullContextManager
from models.hf_backend import PyTorchHFBackend
from tools.wiki_tool import WikipediaTool


MAX_STEPS = 8
LOG_PATH = "logs/react_wiki_log.json"


class ReActAgent:
    def __init__(
        self,
        model=None,
        context_manager=None,
        log_path: Optional[str] = LOG_PATH,
    ):
        self.model = model or PyTorchHFBackend()
        self.prompt_builder = PromptBuilder()
        self.parser = ActionParser()
        self.wiki = WikipediaTool()
        self.context_manager = context_manager or FullContextManager()
        self.log_path = log_path
        self.logs: List[dict] = []

    def run(self, question: str) -> str:
        trajectory: List[Tuple[str, str]] = []

        for step in range(1, MAX_STEPS + 1):
            selected_trajectory = self.context_manager.select(trajectory)
            prompt = self.prompt_builder.build(question, selected_trajectory)
            context_stats = self._build_context_stats(
                question,
                trajectory,
                selected_trajectory,
                prompt,
            )
            segment_stats = self._get_segment_stats()
            self._set_pruning_context(context_stats, segment_stats)

            start_time = now_time()
            model_output = self.model.generate(prompt)
            elapsed = now_time() - start_time

            action = self.parser.parse(model_output)

            if action is None:
                observation = (
                    "Observation: Invalid action format. "
                    "Please output one valid action, such as Action: search[Apple Remote]."
                )

                trajectory.append(("model", model_output))
                trajectory.append(("observation", observation))

                self._log(
                    step,
                    prompt,
                    model_output,
                    None,
                    observation,
                    trajectory,
                    elapsed,
                    whether_action_valid=False,
                    whether_finished=False,
                    context_stats=context_stats,
                    segment_stats=segment_stats,
                )
                self._print_step(step, model_output, observation)
                continue

            observation = self._execute_action(action)
            whether_finished = action.name == "finish"

            trajectory.append(("model", model_output))
            trajectory.append(("observation", observation))

            self._log(
                step,
                prompt,
                model_output,
                action,
                observation,
                trajectory,
                elapsed,
                whether_action_valid=True,
                whether_finished=whether_finished,
                context_stats=context_stats,
                segment_stats=segment_stats,
            )
            self._print_step(step, model_output, observation)

            if whether_finished:
                self._save_logs()
                return action.argument

        self._save_logs()
        return "Failed to finish within max steps."

    def _execute_action(self, action: Action) -> str:
        if action.name == "search":
            result = self.wiki.search(action.argument)
            return f"Observation: {result}"

        if action.name == "lookup":
            result = self.wiki.lookup(action.argument)
            return f"Observation: {result}"

        if action.name == "finish":
            return f"Observation: Episode finished. Final answer: {action.argument}"

        return f"Observation: Unknown action [{action.name}]."

    def _log(
        self,
        step: int,
        prompt: str,
        model_output: str,
        action: Optional[Action],
        observation: str,
        trajectory: List[Tuple[str, str]],
        elapsed: float,
        whether_action_valid: bool,
        whether_finished: bool,
        context_stats: Optional[dict],
        segment_stats: Optional[dict],
    ) -> None:
        total_trajectory_text = "\n".join(content for _, content in trajectory)
        cache_stats = self._get_cache_stats()
        self.logs.append(
            make_step_log(
                step=step,
                model_output=model_output,
                action=action,
                observation=observation,
                prompt_tokens=self.model.count_tokens(prompt),
                output_tokens=self.model.count_tokens(model_output),
                observation_tokens=self.model.count_tokens(observation),
                total_trajectory_tokens=self.model.count_tokens(total_trajectory_text),
                generation_time=elapsed,
                whether_action_valid=whether_action_valid,
                whether_finished=whether_finished,
                cache_stats=cache_stats,
                context_stats=context_stats,
                segment_stats=segment_stats,
            )
        )

    def _build_context_stats(
        self,
        question: str,
        full_trajectory: List[Tuple[str, str]],
        selected_trajectory: List[Tuple[str, str]],
        selected_prompt: str,
    ) -> dict:
        full_prompt = self.prompt_builder.build(question, full_trajectory)
        full_prompt_tokens = self.model.count_tokens(full_prompt)
        selected_prompt_tokens = self.model.count_tokens(selected_prompt)
        context_pruned_tokens = max(full_prompt_tokens - selected_prompt_tokens, 0)
        context_pruned_ratio = (
            round(context_pruned_tokens / full_prompt_tokens, 4)
            if full_prompt_tokens > 0
            else 0
        )

        return {
            "context_manager_name": self.context_manager.__class__.__name__,
            "full_prompt_tokens": full_prompt_tokens,
            "selected_prompt_tokens": selected_prompt_tokens,
            "context_pruned_tokens": context_pruned_tokens,
            "context_pruned_ratio": context_pruned_ratio,
        }

    def _set_pruning_context(
        self,
        context_stats: Optional[dict],
        segment_stats: Optional[dict],
    ) -> None:
        set_pruning_context = getattr(self.model, "set_pruning_context", None)
        if set_pruning_context is not None:
            set_pruning_context(
                context_stats=context_stats,
                segment_stats=segment_stats,
            )

    def _get_cache_stats(self) -> Optional[dict]:
        get_cache_stats = getattr(self.model, "get_cache_stats", None)
        if get_cache_stats is None:
            return None
        return get_cache_stats()

    def _get_segment_stats(self) -> Optional[dict]:
        get_segment_stats = getattr(self.context_manager, "get_segment_stats", None)
        if get_segment_stats is None:
            return None
        stats = get_segment_stats()
        return stats if stats else None

    def _save_logs(self) -> None:
        if not self.log_path:
            return

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(self.log_path, "w", encoding="utf-8") as file:
            json.dump(self.logs, file, ensure_ascii=False, indent=2)

    def _print_step(self, step: int, model_output: str, observation: str) -> None:
        print("=" * 80)
        print(f"Step {step}")
        print(model_output)
        print(observation)
