"""Tests for AnthropicCompatibleAdapter."""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_adapters.anthropic_compatible_adapter import AnthropicCompatibleAdapter


class TestNormalizeMessages(unittest.TestCase):
    """Tests for _normalize_messages (no API call needed — pure logic)."""

    def setUp(self):
        with patch("llm_adapters.anthropic_compatible_adapter.anthropic.Anthropic"):
            self.adapter = AnthropicCompatibleAdapter(api_key="fake")

    # ── Test 1: System extracted ───────────────────────────────────────
    def test_normalize_extracts_system(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        sys_prompt, anthropic_msgs = self.adapter._normalize_messages(msgs)
        self.assertEqual(sys_prompt, "Be helpful")
        self.assertEqual(len(anthropic_msgs), 1)
        self.assertEqual(anthropic_msgs[0]["role"], "user")

    # ── Test 2: Multiple system prompts concatenated ───────────────────
    def test_normalize_multiple_system_concatenated(self):
        msgs = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Go"},
        ]
        sys_prompt, _ = self.adapter._normalize_messages(msgs)
        self.assertIn("Part 1", sys_prompt)
        self.assertIn("Part 2", sys_prompt)
        self.assertIn("\n\n", sys_prompt)

    # ── Test 3: tool_output → user with prefix ─────────────────────────
    def test_normalize_tool_output_role(self):
        msgs = [
            {"role": "assistant", "content": "TOOL_CALL: ..."},
            {"role": "tool_output", "content": '{"output": "ok"}'},
        ]
        _, anthropic_msgs = self.adapter._normalize_messages(msgs)
        tool_msg = anthropic_msgs[1]
        self.assertEqual(tool_msg["role"], "user")
        self.assertIn("[Tool Output]:", tool_msg["content"])

    # ── Test 4: Unknown role → user ────────────────────────────────────
    def test_normalize_unknown_role(self):
        msgs = [{"role": "custom", "content": "data"}]
        _, anthropic_msgs = self.adapter._normalize_messages(msgs)
        self.assertEqual(anthropic_msgs[0]["role"], "user")

    # ── Test 5: Merge consecutive same-role ─────────────────────────────
    def test_normalize_merge_consecutive_same_role(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        _, anthropic_msgs = self.adapter._normalize_messages(msgs)
        self.assertEqual(len(anthropic_msgs), 1)
        self.assertIn("first", anthropic_msgs[0]["content"])
        self.assertIn("second", anthropic_msgs[0]["content"])

    # ── Test 6: Alternating preserved ──────────────────────────────────
    def test_normalize_alternating_preserved(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        _, anthropic_msgs = self.adapter._normalize_messages(msgs)
        self.assertEqual(len(anthropic_msgs), 3)


class TestGenerateResponse(unittest.TestCase):
    """Tests for generate_response (API is mocked)."""

    # ── Test 7: Successful response ────────────────────────────────────
    @patch("llm_adapters.anthropic_compatible_adapter.anthropic.Anthropic")
    def test_generate_response_success(self, MockAnthropic):
        mock_client = MockAnthropic.return_value
        mock_block = MagicMock()
        mock_block.text = "Hello from Claude"
        mock_client.messages.create.return_value.content = [mock_block]

        adapter = AnthropicCompatibleAdapter(api_key="fake")
        result = adapter.generate_response(
            [{"role": "user", "content": "Hi"}],
            model="claude-sonnet-4-20250514",
        )
        self.assertEqual(result, "Hello from Claude")

    # ── Test 8: System prompt passed as kwarg ──────────────────────────
    @patch("llm_adapters.anthropic_compatible_adapter.anthropic.Anthropic")
    def test_generate_response_with_system(self, MockAnthropic):
        mock_client = MockAnthropic.return_value
        mock_block = MagicMock()
        mock_block.text = "ok"
        mock_client.messages.create.return_value.content = [mock_block]

        adapter = AnthropicCompatibleAdapter(api_key="fake")
        adapter.generate_response([
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ])
        call_kwargs = mock_client.messages.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("system") or call_kwargs[1].get("system"), "Be concise")

    # ── Test 9: No user messages → error ───────────────────────────────
    @patch("llm_adapters.anthropic_compatible_adapter.anthropic.Anthropic")
    def test_generate_response_no_messages(self, MockAnthropic):
        adapter = AnthropicCompatibleAdapter(api_key="fake")
        result = adapter.generate_response([])
        self.assertEqual(result, "Error: No user messages found.")

    # ── Test 10: Exception → error string ──────────────────────────────
    @patch("llm_adapters.anthropic_compatible_adapter.anthropic.Anthropic")
    def test_generate_response_error(self, MockAnthropic):
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = Exception("timeout")

        adapter = AnthropicCompatibleAdapter(api_key="fake")
        result = adapter.generate_response([{"role": "user", "content": "Hi"}])
        self.assertIn("Error communicating with Anthropic-compatible API", result)
        self.assertIn("timeout", result)


if __name__ == "__main__":
    unittest.main()
