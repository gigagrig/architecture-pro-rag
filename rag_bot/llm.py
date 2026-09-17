"""HTTP client for a configured Chat Completions-compatible LLM."""

import requests
from pydantic import ValidationError

from rag_bot.config import Settings
from rag_bot.schemas import GeneratedAnswer


class GenerationError(Exception):
    """Sanitized upstream failure; never includes URLs, headers or response bodies."""


class ChatGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, messages: list[dict[str, str]]) -> GeneratedAnswer:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        try:
            response = requests.post(
                f"{self.settings.llm_base_url}/chat/completions",
                headers=headers,
                json={"model": self.settings.llm_model, "messages": messages,
                      "response_format": {"type": "json_schema", "json_schema": {
                          "name": "rag_answer", "strict": True,
                          "schema": GeneratedAnswer.model_json_schema(),
                      }},
                      "temperature": 0.3, "max_tokens": 1500},
                timeout=(10, self.settings.llm_timeout),
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise GenerationError("LLM did not finish its answer")
            return GeneratedAnswer.model_validate_json(choice["message"]["content"])
        except requests.Timeout:
            raise GenerationError("LLM request timed out") from None
        except requests.RequestException:
            raise GenerationError("LLM is unavailable or rejected the request") from None
        except (ValueError, KeyError, IndexError, TypeError, ValidationError):
            raise GenerationError("LLM returned an invalid answer format") from None
