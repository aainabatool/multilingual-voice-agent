import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import ollama


@dataclass
class LLMResponse:
    text: str
    model_name: str
    inference_time_s: float


class LLMAdapter(ABC):
    """Common interface so the LLM backend can be swapped/benchmarked (spec 5.6)."""

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.2) -> LLMResponse:
        raise NotImplementedError


class OllamaAdapter(LLMAdapter):
    """Local Ollama-backed LLM adapter."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def chat(self, messages: list[dict], temperature: float = 0.2) -> LLMResponse:
        start = time.perf_counter()
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": temperature},
        )
        elapsed = time.perf_counter() - start
        return LLMResponse(
            text=response["message"]["content"].strip(),
            model_name=self.model_name,
            inference_time_s=elapsed,
        )
