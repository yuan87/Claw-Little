"""
LLM Provider Registry & Factory

All supported providers are configured here. Each provider maps to either the
OpenAI-compatible or Anthropic-compatible adapter, with a base_url override
and default model.

To add a new provider, simply add an entry to the PROVIDERS dict below.
"""

import os
from dotenv import load_dotenv
from .openai_compatible_adapter import OpenAICompatibleAdapter
from .anthropic_compatible_adapter import AnthropicCompatibleAdapter

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Provider Registry
# ──────────────────────────────────────────────────────────────────────────────
# api_format      : "openai" | "anthropic"  → which adapter class to instantiate
# base_url        : API endpoint override (None = SDK default)
# api_key_env     : env-var name for the API key
# auth_token_env  : (optional) env-var name for bearer-token auth (Anthropic format)
# default_model   : model used when none is explicitly specified
# ──────────────────────────────────────────────────────────────────────────────

PROVIDERS = {
    # ── OpenAI-compatible providers ─────────────────────────────────────────
    "openai": {
        "api_format":     "openai",
        "base_url":       None,
        "api_key_env":    "OPENAI_API_KEY",
        "default_model":  "gpt-4o-mini",
    },
    "deepseek": {
        "api_format":     "openai",
        "base_url":       "https://api.deepseek.com",
        "api_key_env":    "DEEPSEEK_API_KEY",
        "default_model":  "deepseek-chat",
    },
    "openrouter": {
        "api_format":     "openai",
        "base_url":       "https://openrouter.ai/api/v1",
        "api_key_env":    "OPENROUTER_API_KEY",
        "default_model":  "openai/gpt-4o-mini",
    },
    "anyrouter": {
        "api_format":     "openai",
        "base_url":       "https://anyrouter.com/v1",
        "api_key_env":    "ANYROUTER_API_KEY",
        "default_model":  "gpt-4o-mini",
    },
    "gemini": {
        "api_format":     "openai",
        "base_url":       "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env":    "GEMINI_API_KEY",
        "default_model":  "gemini-2.5-flash",
    },
    # ── Anthropic-compatible providers ──────────────────────────────────────
    "anthropic": {
        "api_format":     "anthropic",
        "base_url":       None,
        "api_key_env":    "ANTHROPIC_API_KEY",
        "default_model":  "claude-sonnet-4-20250514",
    },
    "agentrouter": {
        "api_format":       "anthropic",
        "base_url":         "https://agentrouter.org/",
        "api_key_env":      "AGENTROUTER_API_KEY",
        "auth_token_env":   "AGENTROUTER_AUTH_TOKEN",
        "default_model":    "claude-sonnet-4-20250514",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_provider_config(provider: str) -> dict:
    """Return the config dict for *provider*, or raise ValueError."""
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Available: {', '.join(sorted(PROVIDERS.keys()))}"
        )
    return PROVIDERS[provider]


def get_default_model(provider: str) -> str:
    """Return the default model string for *provider*."""
    return get_provider_config(provider)["default_model"]


def get_api_format(provider: str) -> str:
    """Return 'openai' or 'anthropic' for *provider*."""
    return get_provider_config(provider)["api_format"]


def list_providers() -> list[str]:
    """Return a sorted list of all registered provider names."""
    return sorted(PROVIDERS.keys())


def get_llm_adapter(provider: str, api_key: str = None):
    """
    Factory: create the correct adapter for *provider*.

    Args:
        provider: registered provider name (case-insensitive)
        api_key:  optional override; falls back to the provider's env-var

    Returns:
        OpenAICompatibleAdapter | AnthropicCompatibleAdapter
    """
    config = get_provider_config(provider)
    resolved_key = api_key or os.getenv(config["api_key_env"])
    base_url = config.get("base_url")

    if config["api_format"] == "openai":
        return OpenAICompatibleAdapter(api_key=resolved_key, base_url=base_url)

    elif config["api_format"] == "anthropic":
        auth_token_env = config.get("auth_token_env")
        auth_token = os.getenv(auth_token_env) if auth_token_env else None
        return AnthropicCompatibleAdapter(
            api_key=resolved_key,
            base_url=base_url,
            auth_token=auth_token,
        )

    else:
        raise ValueError(f"Unknown api_format '{config['api_format']}' for provider '{provider}'")
