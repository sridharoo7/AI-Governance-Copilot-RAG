"""Provider adapters isolate OmniRoute and direct-provider behavior from the workflow."""

from __future__ import annotations

import asyncio
from typing import Protocol

import httpx
from pydantic import ValidationError

from .schemas import GroundedAnswer
from .settings import Settings


class AnswerProvider(Protocol):
    """Contract used by the graph to obtain schema-conformant grounded answers."""

    async def generate(self, prompt: str) -> GroundedAnswer: ...


class ProviderResponseError(httpx.HTTPError):
    """Signals a successful HTTP response that violates the provider's chat-completion contract."""


class OpenAICompatibleProvider:
    """Calls OmniRoute first and can be re-instantiated for a direct provider fallback."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 180,
        supports_json_mode: bool = True,
        supports_reasoning_control: bool = True,
    ) -> None:
        """Stores only the connection settings needed by the OpenAI-compatible endpoint."""

        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.timeout_seconds = timeout_seconds
        self.supports_json_mode = supports_json_mode
        self.supports_reasoning_control = supports_reasoning_control

    async def generate(self, prompt: str) -> GroundedAnswer:
        """Requests schema-valid JSON with bounded transient-error and format-repair retries."""

        # A concrete contract prevents small-model field-name drift such as "claim" in place
        # of the required AtomicClaim field "text". Quotes must still pass deterministic checks.
        schema_reminder = (
            "Return only this exact JSON shape: "
            '{"answer":"...","claims":[{"text":"...","citations":['
            '{"chunk_id":"retrieved-chunk-id","quote":"exact evidence substring"}]}],'
            '"abstained":false,"abstention_reason":null}. '
            "Use text, not claim. Use citations, not sources."
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for attempt in range(3):
            # Later calls are bounded repair retries, not opportunities to add facts. The same
            # evidence-only prompt is retained so an outage retry cannot change answer scope.
            request_prompt = prompt if attempt == 0 else f"{prompt}\n\nFORMAT REPAIR REQUIRED:\n{schema_reminder}"
            payload = {
                "model": self.model,
                "temperature": 0,
                "stream": False,
                "messages": [{"role": "system", "content": request_prompt}],
            }
            if self.supports_reasoning_control:
                # qwen3 is a thinking model; disable hidden reasoning so the response content is
                # limited to the validated JSON contract instead of mixing analysis with citations.
                payload["reasoning_effort"] = "none"
            if self.supports_json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                    response.raise_for_status()
            except httpx.HTTPStatusError as error:
                if _is_transient_status(error) and attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except httpx.RequestError:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            try:
                content = _chat_completion_content(response.json())
            except (KeyError, TypeError, ValueError) as error:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                # OpenRouter can return a 2xx body containing an upstream provider error. It
                # must be treated as a transport/provider failure, never as a Python KeyError.
                raise ProviderResponseError("Provider response did not contain chat-completion choices.") from error
            try:
                return GroundedAnswer.model_validate_json(content)
            except ValidationError:
                # Invalid JSON cannot reach citation validation or the caller as an answer.
                continue
        return GroundedAnswer(
            answer="I cannot answer from the approved corpus because the model did not return a valid grounded citation format.",
            claims=[],
            abstained=True,
            abstention_reason="The generation schema could not be validated after one repair attempt.",
        )


def _is_transient_status(error: httpx.HTTPStatusError) -> bool:
    """Returns whether an HTTP status is worth a bounded availability retry."""

    return error.response.status_code == 429 or 500 <= error.response.status_code <= 599


def _chat_completion_content(payload: object) -> str:
    """Extracts non-empty assistant content from an OpenAI-compatible completion payload."""

    if not isinstance(payload, dict):
        raise TypeError("Provider payload is not a JSON object.")
    choices = payload["choices"]
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Provider payload has no completion choices.")
    message = choices[0]["message"]
    if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
        raise ValueError("Provider completion has no assistant content.")
    return message["content"]


class FallbackAnswerProvider:
    """Uses a direct OpenAI-compatible provider only when the OmniRoute call fails."""

    def __init__(self, primary: AnswerProvider, fallback: AnswerProvider | None) -> None:
        """Keeps gateway preference explicit while allowing an approved outage fallback."""

        self.primary, self.fallback = primary, fallback

    async def generate(self, prompt: str) -> GroundedAnswer:
        """Attempts OmniRoute first and never retries a failed fallback indefinitely."""

        try:
            return await self.primary.generate(prompt)
        except httpx.HTTPError:
            if self.fallback is None:
                raise
            # The fallback receives exactly the same grounded prompt and cannot expand scope.
            return await self.fallback.generate(prompt)


def build_provider(config: Settings) -> AnswerProvider:
    """Builds the explicitly selected generation route with an approved local fallback."""

    direct = None
    if config.direct_llm_base_url:
        # Ollama ignores the key but OpenAI-compatible clients conventionally require a value.
        direct = OpenAICompatibleProvider(config.direct_llm_base_url, config.direct_llm_api_key or "ollama", config.rag_model, config.llm_timeout_seconds)
    if config.omniroute_base_url and config.omniroute_api_key:
        primary = OpenAICompatibleProvider(config.omniroute_base_url, config.omniroute_api_key, config.rag_model, config.llm_timeout_seconds)
        return FallbackAnswerProvider(primary, direct)
    if config.rag_generation_provider == "gemini":
        if not config.google_api_key:
            raise ValueError("Set GOOGLE_API_KEY before selecting RAG_GENERATION_PROVIDER=gemini.")
        # Gemini's official OpenAI-compatible endpoint keeps the runtime adapter identical
        # to OmniRoute/local adapters while providing stronger JSON citation generation.
        primary = OpenAICompatibleProvider(
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            config.google_api_key,
            config.gemini_generation_model,
            config.llm_timeout_seconds,
        )
        return FallbackAnswerProvider(primary, direct)
    if config.rag_generation_provider == "openrouter":
        if not config.openrouter_api_key or not config.openrouter_model:
            raise ValueError(
                "Set OPENROUTER_API_KEY and OPENROUTER_MODEL before selecting "
                "RAG_GENERATION_PROVIDER=openrouter."
            )
        # OpenRouter is already a multi-provider routing layer. Do not fall back to a
        # local LLM here: benchmark/release behavior must remain attributable to its pinned route.
        return OpenAICompatibleProvider(
            config.openrouter_base_url,
            config.openrouter_api_key,
            config.openrouter_model,
            config.llm_timeout_seconds,
            supports_json_mode=config.openrouter_supports_structured_outputs,
            supports_reasoning_control=False,
        )
    if direct:
        return direct
    raise ValueError("Configure OmniRoute credentials or DIRECT_LLM_BASE_URL before starting the API.")
