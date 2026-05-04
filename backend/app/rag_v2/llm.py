"""LLM clients for grounded answer generation."""
from __future__ import annotations

import logging
from typing import Protocol

from app.rag_v2.config import RagV2Config

logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1024,
    ) -> str: ...


class GroqClient:
    def __init__(self, api_key: str, model: str) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError("Run: pip install groq") from exc
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        self._client = Groq(api_key=api_key)
        self._model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1024,
    ) -> str:
        full: list[dict[str, str]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=full,
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content or ""


class NvidiaClient:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        if not api_key:
            raise ValueError("NVIDIA_KEY is not set in .env")
        self._api_key = api_key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1024,
    ) -> str:
        import requests  # type: ignore[import-untyped]

        full: list[dict[str, str]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        resp = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": full,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"].get("content") or ""


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        self._api_key = api_key.strip().strip('"')
        self._model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1024,
    ) -> str:
        import requests  # type: ignore[import-untyped]

        full: list[dict[str, str]] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        resp = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "PowerMind RAG",
            },
            json={
                "model": self._model,
                "messages": full,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"].get("content") or ""


class GeminiClient:
    def __init__(self, api_keys: list[str], model: str) -> None:
        try:
            from google import genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Run: pip install google-genai") from exc
        keys = []
        for key in api_keys:
            clean = key.strip().strip('"')
            if clean and clean not in keys:
                keys.append(clean)
        if not keys:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        self._clients = [genai.Client(api_key=key) for key in keys]
        self._model = model
        self._next = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int = 1024,
    ) -> str:
        from google.genai import types  # type: ignore[import-untyped]

        conversation = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            conversation.append(f"{role.upper()}:\n{content}")
        prompt = "\n\n".join(conversation)

        last_error: Exception | None = None
        for _ in range(len(self._clients)):
            client = self._clients[self._next]
            self._next = (self._next + 1) % len(self._clients)
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system or None,
                        temperature=0,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Gemini generation key slot failed; trying next key: %s", exc)
        if last_error is not None:
            raise last_error
        return ""



def build_llm(config: RagV2Config) -> ChatClient:
    if config.generation_provider == "openrouter":
        logger.info("Using OpenRouter generation model=%s", config.openrouter_chat_model)
        return OpenRouterClient(
            api_key=config.openrouter_api_key,
            model=config.openrouter_chat_model,
            base_url=config.openrouter_base_url,
        )
    if config.generation_provider == "gemini":
        logger.info("Using Gemini generation model=%s", config.gemini_chat_model)
        api_keys = [config.gemini_api_key, *config.gemini_api_keys]
        return GeminiClient(api_keys=api_keys, model=config.gemini_chat_model)
    if config.generation_provider == "nvidia":
        logger.info("Using NVIDIA generation model=%s", config.nvidia_chat_model)
        return NvidiaClient(
            api_key=config.nvidia_api_key,
            model=config.nvidia_chat_model,
            base_url=config.nvidia_base_url,
        )
    if config.generation_provider == "groq":
        logger.info("Using Groq generation model=%s", config.groq_chat_model)
        return GroqClient(api_key=config.groq_api_key, model=config.groq_chat_model)
    raise ValueError(
        "Unsupported POWERMIND_GENERATION_PROVIDER="
        f"{config.generation_provider!r}; expected 'groq', 'gemini', 'nvidia', or 'openrouter'"
    )
