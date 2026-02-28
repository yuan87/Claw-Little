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
    # ═══════════════════════════════════════════════════════════════════════
    #  OpenAI-compatible providers (Chat Completions API)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Tier 1: Major providers ─────────────────────────────────────────
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
    "gemini": {
        "api_format":     "openai",
        "base_url":       "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env":    "GEMINI_API_KEY",
        "default_model":  "gemini-2.5-flash",
    },
    "mistral": {
        "api_format":     "openai",
        "base_url":       "https://api.mistral.ai/v1",
        "api_key_env":    "MISTRAL_API_KEY",
        "default_model":  "mistral-large-latest",
    },
    "xai": {
        "api_format":     "openai",
        "base_url":       "https://api.x.ai/v1",
        "api_key_env":    "XAI_API_KEY",
        "default_model":  "grok-3-mini",
    },

    # ── Tier 2: Inference platforms ─────────────────────────────────────
    "groq": {
        "api_format":     "openai",
        "base_url":       "https://api.groq.com/openai/v1",
        "api_key_env":    "GROQ_API_KEY",
        "default_model":  "llama-3.3-70b-versatile",
    },
    "cerebras": {
        "api_format":     "openai",
        "base_url":       "https://api.cerebras.ai/v1",
        "api_key_env":    "CEREBRAS_API_KEY",
        "default_model":  "llama-3.3-70b",
    },
    "together": {
        "api_format":     "openai",
        "base_url":       "https://api.together.xyz/v1",
        "api_key_env":    "TOGETHER_API_KEY",
        "default_model":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "nvidia": {
        "api_format":     "openai",
        "base_url":       "https://integrate.api.nvidia.com/v1",
        "api_key_env":    "NVIDIA_API_KEY",
        "default_model":  "nvidia/llama-3.1-nemotron-70b-instruct",
    },
    "huggingface": {
        "api_format":     "openai",
        "base_url":       "https://router.huggingface.co/v1",
        "api_key_env":    "HF_TOKEN",
        "default_model":  "deepseek-ai/DeepSeek-R1",
    },
    "venice": {
        "api_format":     "openai",
        "base_url":       "https://api.venice.ai/api/v1",
        "api_key_env":    "VENICE_API_KEY",
        "default_model":  "deepseek-r1-671b",
    },

    # ── Tier 3: Routers / Gateways ──────────────────────────────────────
    "openrouter": {
        "api_format":     "openai",
        "base_url":       "https://openrouter.ai/api/v1",
        "api_key_env":    "OPENROUTER_API_KEY",
        "default_model":  "anthropic/claude-sonnet-4",
    },
    "anyrouter": {
        "api_format":     "openai",
        "base_url":       "https://anyrouter.com/v1",
        "api_key_env":    "ANYROUTER_API_KEY",
        "default_model":  "gpt-4o-mini",
    },
    "litellm": {
        "api_format":     "openai",
        "base_url":       "http://localhost:4000",
        "api_key_env":    "LITELLM_API_KEY",
        "default_model":  "gpt-4o-mini",
    },

    # ── Tier 4: China-region providers ──────────────────────────────────
    "moonshot": {
        "api_format":     "openai",
        "base_url":       "https://api.moonshot.ai/v1",
        "api_key_env":    "MOONSHOT_API_KEY",
        "default_model":  "kimi-k2.5",
    },
    "zai": {
        "api_format":     "openai",
        "base_url":       "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env":    "ZAI_API_KEY",
        "default_model":  "glm-4.7",
    },
    "qianfan": {
        "api_format":     "openai",
        "base_url":       "https://qianfan.baidubce.com/v2",
        "api_key_env":    "QIANFAN_API_KEY",
        "default_model":  "deepseek-v3.2",
    },
    "qwen": {
        "api_format":     "openai",
        "base_url":       "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env":    "DASHSCOPE_API_KEY",
        "default_model":  "qwen-plus",
    },

    # ── Tier 5: Local / Self-hosted ─────────────────────────────────────
    "ollama": {
        "api_format":     "openai",
        "base_url":       "http://127.0.0.1:11434/v1",
        "api_key_env":    "OLLAMA_API_KEY",
        "default_model":  "llama3.3",
    },
    "vllm": {
        "api_format":     "openai",
        "base_url":       "http://127.0.0.1:8000/v1",
        "api_key_env":    "VLLM_API_KEY",
        "default_model":  "default",
    },
    "lmstudio": {
        "api_format":     "openai",
        "base_url":       "http://localhost:1234/v1",
        "api_key_env":    "LMSTUDIO_API_KEY",
        "default_model":  "default",
    },

    # ═══════════════════════════════════════════════════════════════════════
    #  Anthropic-compatible providers (Messages API)
    # ═══════════════════════════════════════════════════════════════════════
    "anthropic": {
        "api_format":     "anthropic",
        "base_url":       None,
        "api_key_env":    "ANTHROPIC_API_KEY",
        "default_model":  "claude-sonnet-4-20250514",
    },
    "minimax": {
        "api_format":       "anthropic",
        "base_url":         "https://api.minimax.io/anthropic",
        "api_key_env":      "MINIMAX_API_KEY",
        "default_model":    "MiniMax-M2.5",
    },
    "xiaomi": {
        "api_format":       "anthropic",
        "base_url":         "https://api.xiaomimimo.com/anthropic",
        "api_key_env":      "XIAOMI_API_KEY",
        "default_model":    "mimo-v2-flash",
    },
    "kimi-coding": {
        "api_format":       "anthropic",
        "base_url":         "https://api.moonshot.ai/anthropic",
        "api_key_env":      "KIMI_API_KEY",
        "default_model":    "k2p5",
    },
    "synthetic": {
        "api_format":       "anthropic",
        "base_url":         "https://api.synthetic.new/anthropic",
        "api_key_env":      "SYNTHETIC_API_KEY",
        "default_model":    "hf:MiniMaxAI/MiniMax-M2.1",
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
