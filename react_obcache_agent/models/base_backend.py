from abc import ABC, abstractmethod


class BaseModelBackend(ABC):
    """Common interface for ReAct model backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError
