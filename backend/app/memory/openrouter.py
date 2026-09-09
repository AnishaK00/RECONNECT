"""OpenRouter client for the memory-consolidation LLM call."""
from __future__ import annotations
import os
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv at runtime."""
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()

DEFAULT_MODELS = (
    "~openai/gpt-latest",
    "~anthropic/claude-sonnet-latest",
    "google/gemini-3-flash-preview",
)

class OpenRouterError(RuntimeError):
    pass

class OpenRouterMemoryClient:
    """Synchronous client; OpenRouter tries fallback models in order on failure."""
    def __init__(self, api_key: str | None = None, models: tuple[str, ...] | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured")
        configured = os.getenv("OPENROUTER_MEMORY_MODELS")
        self.models = models or (tuple(item.strip() for item in configured.split(",") if item.strip()) if configured else DEFAULT_MODELS)
        if not self.models:
            raise OpenRouterError("At least one OpenRouter model must be configured")

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.models[0],
            "models": list(self.models[1:]),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
            "provider": {"allow_fallbacks": True, "data_collection": "deny"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            request = Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode(), headers=headers, method="POST")
            with urlopen(request, timeout=90.0) as response:
                content = json.load(response)["choices"][0]["message"]["content"]
        except (HTTPError, URLError, KeyError, IndexError, TypeError, ValueError) as error:
            raise OpenRouterError(f"OpenRouter consolidation request failed: {error}") from error
        if not isinstance(content, str) or not content.strip():
            raise OpenRouterError("OpenRouter returned an empty consolidation response")
        return content
