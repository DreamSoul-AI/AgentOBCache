import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base_backend import BaseModelBackend


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

MAX_NEW_TOKENS = 256


class PyTorchHFBackend(BaseModelBackend):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

    def generate(self, prompt: str) -> str:
        return self._generate_text(prompt, enforce_react_format=True)

    def generate_raw(self, prompt: str) -> str:
        return self._generate_text(prompt, enforce_react_format=False)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def warmup(self) -> None:
        inputs = self.tokenizer("warmup", return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _generate_text(self, prompt: str, enforce_react_format: bool) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        if not enforce_react_format:
            return text

        if not text.startswith("Thought:"):
            text = "Thought: " + text

        text = self._cut_after_first_action(text)
        return text

    def _cut_after_first_action(self, text: str) -> str:
        action_match = re.search(
            r"Action\s*:\s*(search|lookup|finish)\[.*?\]",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if action_match is None:
            return text.strip()

        return text[:action_match.end()].strip()
