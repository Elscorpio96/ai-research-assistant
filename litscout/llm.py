"""Thin LLM clients (Anthropic Messages API, or any OpenAI-compatible chat endpoint) over httpx."""

from __future__ import annotations

import httpx

from .config import Settings

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}


class LLMError(Exception):
    pass


def make_completer(settings: Settings, client: httpx.Client | None = None):
    """Return complete(system, prompt) -> (text, model)."""
    provider = settings.llm_provider
    model = settings.llm_model or DEFAULT_MODELS.get(provider, "")
    http = client or httpx.Client(timeout=120)

    def complete(system: str, prompt: str) -> tuple[str, str]:
        try:
            if provider == "anthropic":
                r = http.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": settings.llm_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 4000,
                        "temperature": 0.2,
                        "system": system,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                r.raise_for_status()
                data = r.json()
                return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"), data.get("model", model)
            if provider == "openai":
                headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
                r = http.post(
                    settings.llm_base_url.rstrip("/") + "/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "temperature": 0.2,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    },
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"], data.get("model", model)
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:200].replace(settings.llm_api_key or "\0", "***")
            raise LLMError(f"The {provider} API returned HTTP {e.response.status_code}: {detail}") from None
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach the {provider} API ({type(e).__name__}).") from None
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected response from the {provider} API: {e}") from None
        raise LLMError("No LLM provider is configured.")

    return complete
