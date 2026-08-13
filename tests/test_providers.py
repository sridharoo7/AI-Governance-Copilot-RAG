"""Tests provider-side schema handling without calling Ollama or external services."""

import pytest

from rag_copilot.providers import OpenAICompatibleProvider, ProviderResponseError


@pytest.mark.asyncio
async def test_malformed_model_output_becomes_safe_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures a model schema failure cannot become an API 500 or unsupported answer."""

    provider = OpenAICompatibleProvider("http://example.invalid/v1", "", "test-model")

    async def invalid_completion(*_args, **_kwargs):
        """Returns the legacy malformed claim field on both bounded attempts."""

        class Response:
            """Mimics only the HTTP response surface used by the provider adapter."""

            def raise_for_status(self) -> None:
                """Leaves the synthetic response successful."""

            def json(self) -> dict:
                """Supplies JSON that fails the required AtomicClaim schema."""

                return {"choices": [{"message": {"content": '{"answer":"x","claims":[{"claim":"x"}],"abstained":false}'}}]}

        return Response()

    monkeypatch.setattr("httpx.AsyncClient.post", invalid_completion)
    answer = await provider.generate("grounded prompt")
    assert answer.abstained is True
    assert answer.claims == []


@pytest.mark.asyncio
async def test_missing_choices_becomes_a_controlled_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevents 2xx provider error bodies from escaping as an uncaught KeyError."""

    provider = OpenAICompatibleProvider("http://example.invalid/v1", "", "test-model")

    async def error_body(*_args, **_kwargs):
        """Returns a syntactically valid JSON body without the completion contract."""

        class Response:
            """Mimics a 2xx response returned by an upstream routing provider."""

            def raise_for_status(self) -> None:
                """Leaves the synthetic response successful at the HTTP layer."""

            def json(self) -> dict:
                """Represents an upstream error body that lacks choices."""

                return {"error": {"message": "temporary upstream failure"}}

        return Response()

    async def immediate_sleep(*_args, **_kwargs) -> None:
        """Removes retry delay from the deterministic unit test."""

    monkeypatch.setattr("httpx.AsyncClient.post", error_body)
    monkeypatch.setattr("asyncio.sleep", immediate_sleep)
    with pytest.raises(ProviderResponseError):
        await provider.generate("grounded prompt")
