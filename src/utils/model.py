from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class LLMError(RuntimeError):
    """Raised when the LLM service returns an error or malformed response."""


@dataclass
class LLMConfig:
    api_base: str
    model: str
    api_key: Optional[str] = None
    timeout: int = 120
    verify_ssl: bool = True
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9


@dataclass
class LLMResult:
    content: str
    raw: Dict[str, Any]


class LLMClient:
    """Generic LLM client (OpenAI-compatible API format)."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_env_config()

    def _chat_url(self) -> str:
        return f"{self.config.api_base.rstrip('/')}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        if "content" in data and isinstance(data["content"], str):
            return data["content"]

        choices = data.get("choices") or []
        if choices:
            first = choices[0] or {}
            message = first.get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

        raise LLMError(f"Unexpected LLM response format: {data}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        payload: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
        }
        if extra_payload:
            payload.update(extra_payload)
        try:
            response = requests.post(
                self._chat_url(),
                json=payload,
                headers=self._headers(),
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Failed to connect to LLM service: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"LLM API error {response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"LLM API did not return valid JSON: {response.text}") from exc

        content = self._extract_content(data)
        return LLMResult(content=content, raw=data)

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> LLMResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages=messages, **kwargs)


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_env_config() -> LLMConfig:
    api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
    model = os.getenv("LLM_MODEL", os.getenv("MODEL_NAME", "gpt-4o-mini"))
    api_key = os.getenv("LLM_API_KEY", "")
    timeout = int(os.getenv("LLM_TIMEOUT", "120"))
    verify_ssl = _to_bool(os.getenv("LLM_VERIFY_SSL", "true"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    top_p = float(os.getenv("LLM_TOP_P", "0.9"))

    return LLMConfig(
        api_base=api_base,
        model=model,
        api_key=api_key,
        timeout=timeout,
        verify_ssl=verify_ssl,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def ask_messages(
    messages: List[Dict[str, str]],
    **kwargs: Any,
) -> LLMResult:
    """Send arbitrary chat messages to the default LLM client."""
    return get_llm_client().chat(messages=messages, **kwargs)


def ask(
    system_prompt: str,
    user_prompt: str = "",
    **kwargs: Any,
) -> str:
    """
    Backward-compatible convenience wrapper.
    Returns only text content.
    """
    result = get_llm_client().ask(system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
    return result.content