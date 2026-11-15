"""Runtime settings, read from environment variables (or a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so the project needs no third-party packages."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _list(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.environ.get(name, default).split(",") if m.strip()]


@dataclass
class Settings:
    # "live" calls Bifrost + Tavily. "replay" uses recorded responses in fixtures/ (offline tests only).
    mode: str = field(default_factory=lambda: os.environ.get("FRESHPIN_MODE", "live"))

    # Bifrost gateway (OpenAI-compatible endpoint). Start it with: npx -y @maximhq/bifrost
    bifrost_url: str = field(default_factory=lambda: os.environ.get("BIFROST_URL", "http://localhost:8080").rstrip("/"))
    bifrost_virtual_key: str = field(default_factory=lambda: os.environ.get("BIFROST_VIRTUAL_KEY", ""))

    # Models use Bifrost's "provider/model" naming. The first is primary; the rest are fallbacks.
    extractor_models: list[str] = field(default_factory=lambda: _list(
        "FRESHPIN_EXTRACTOR_MODELS", "gemini/gemini-2.5-flash,groq/openai/gpt-oss-120b"))
    verifier_models: list[str] = field(default_factory=lambda: _list(
        "FRESHPIN_VERIFIER_MODELS", "groq/openai/gpt-oss-120b,gemini/gemini-2.5-flash"))
    # The second, independent judge in the recheck layer. Its primary should be a different provider
    # from the verifier's primary, so one model's blind spot can't confirm itself.
    checker_models: list[str] = field(default_factory=lambda: _list(
        "FRESHPIN_CHECKER_MODELS", "gemini/gemini-2.5-flash,groq/openai/gpt-oss-120b"))
    # Recheck every claim with a second search and a second judge (costs one extra Tavily search per claim).
    recheck: bool = field(default_factory=lambda: os.environ.get("FRESHPIN_RECHECK", "1") not in ("0", "false", "no"))
    # Used by the "simulate outage" switch: a model that will fail, so Bifrost must fall back.
    outage_model: str = field(default_factory=lambda: os.environ.get("FRESHPIN_OUTAGE_MODEL", "groq/model-that-does-not-exist"))

    # Spend guard for a single check run (tokens across all LLM calls).
    run_token_budget: int = field(default_factory=lambda: int(os.environ.get("FRESHPIN_RUN_TOKEN_BUDGET", "60000")))

    # Tavily
    tavily_api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))
    tavily_url: str = field(default_factory=lambda: os.environ.get("TAVILY_URL", "https://api.tavily.com/search"))
    max_sources_per_claim: int = field(default_factory=lambda: int(os.environ.get("FRESHPIN_MAX_SOURCES", "5")))

    db_path: Path = field(default_factory=lambda: Path(os.environ.get("FRESHPIN_DB", str(ROOT / "freshpin.db"))))
    host: str = field(default_factory=lambda: os.environ.get("FRESHPIN_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("FRESHPIN_PORT", "8787")))
    timeout_s: int = field(default_factory=lambda: int(os.environ.get("FRESHPIN_TIMEOUT", "60")))


def settings() -> Settings:
    return Settings()
