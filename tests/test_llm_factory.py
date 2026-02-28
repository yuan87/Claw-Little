"""Tests for LLM Factory — provider registry + factory function."""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_adapters.llm_factory import (
    PROVIDERS,
    get_provider_config,
    get_default_model,
    get_api_format,
    list_providers,
    get_llm_adapter,
)
from llm_adapters.openai_compatible_adapter import OpenAICompatibleAdapter
from llm_adapters.anthropic_compatible_adapter import AnthropicCompatibleAdapter


class TestGetProviderConfig(unittest.TestCase):
    # ── Test 1: Valid providers ─────────────────────────────────────────
    def test_get_provider_config_valid(self):
        required_keys = {"api_format", "base_url", "api_key_env", "default_model"}
        for name in PROVIDERS:
            config = get_provider_config(name)
            self.assertTrue(required_keys.issubset(config.keys()), f"{name} missing keys")

    # ── Test 2: Invalid provider ───────────────────────────────────────
    def test_get_provider_config_invalid(self):
        with self.assertRaises(ValueError) as ctx:
            get_provider_config("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    # ── Test 3: Case insensitive ───────────────────────────────────────
    def test_get_provider_config_case_insensitive(self):
        config = get_provider_config("OpenAI")
        self.assertEqual(config["api_format"], "openai")
        config2 = get_provider_config("DEEPSEEK")
        self.assertEqual(config2["default_model"], "deepseek-chat")


class TestGetDefaultModel(unittest.TestCase):
    # ── Test 4: Default models ─────────────────────────────────────────
    def test_get_default_model(self):
        self.assertEqual(get_default_model("openai"), "gpt-4o-mini")
        self.assertEqual(get_default_model("deepseek"), "deepseek-chat")
        self.assertEqual(get_default_model("gemini"), "gemini-2.5-flash")
        self.assertEqual(get_default_model("anthropic"), "claude-sonnet-4-20250514")


class TestGetApiFormat(unittest.TestCase):
    # ── Test 5: API formats ────────────────────────────────────────────
    def test_get_api_format(self):
        self.assertEqual(get_api_format("openai"), "openai")
        self.assertEqual(get_api_format("deepseek"), "openai")
        self.assertEqual(get_api_format("anthropic"), "anthropic")
        self.assertEqual(get_api_format("agentrouter"), "anthropic")


class TestListProviders(unittest.TestCase):
    # ── Test 6: List all providers ─────────────────────────────────────
    def test_list_providers(self):
        providers = list_providers()
        self.assertEqual(providers, sorted(providers))  # sorted
        # Must include at least these core providers
        core = {"openai", "deepseek", "gemini", "anthropic", "groq", "mistral", "ollama"}
        self.assertTrue(core.issubset(set(providers)))
        self.assertGreaterEqual(len(providers), 20)


class TestGetLlmAdapter(unittest.TestCase):
    # ── Test 7: OpenAI returns correct type ────────────────────────────
    @patch("llm_adapters.openai_compatible_adapter.openai.OpenAI")
    def test_get_llm_adapter_openai_type(self, mock_openai):
        adapter = get_llm_adapter("openai", api_key="test-key")
        self.assertIsInstance(adapter, OpenAICompatibleAdapter)

    # ── Test 8: Anthropic returns correct type ─────────────────────────
    @patch("llm_adapters.anthropic_compatible_adapter.anthropic.Anthropic")
    def test_get_llm_adapter_anthropic_type(self, mock_anthropic):
        adapter = get_llm_adapter("anthropic", api_key="test-key")
        self.assertIsInstance(adapter, AnthropicCompatibleAdapter)

    # ── Test 9: Explicit api_key is used ───────────────────────────────
    @patch("llm_adapters.openai_compatible_adapter.openai.OpenAI")
    def test_get_llm_adapter_with_api_key(self, mock_openai):
        get_llm_adapter("openai", api_key="my-explicit-key")
        mock_openai.assert_called_once_with(api_key="my-explicit-key")

    # ── Test 10: All OpenAI-format providers ───────────────────────────
    @patch("llm_adapters.openai_compatible_adapter.openai.OpenAI")
    def test_all_openai_format_providers(self, mock_openai):
        openai_providers = [n for n, c in PROVIDERS.items() if c["api_format"] == "openai"]
        for name in openai_providers:
            adapter = get_llm_adapter(name, api_key="k")
            self.assertIsInstance(adapter, OpenAICompatibleAdapter, f"{name} should be OpenAICompatible")

    # ── Test 11: All Anthropic-format providers ────────────────────────
    @patch("llm_adapters.anthropic_compatible_adapter.anthropic.Anthropic")
    def test_all_anthropic_format_providers(self, mock_anthropic):
        anthropic_providers = [n for n, c in PROVIDERS.items() if c["api_format"] == "anthropic"]
        for name in anthropic_providers:
            adapter = get_llm_adapter(name, api_key="k")
            self.assertIsInstance(adapter, AnthropicCompatibleAdapter, f"{name} should be AnthropicCompatible")


if __name__ == "__main__":
    unittest.main()
