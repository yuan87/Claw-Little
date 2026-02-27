"""Tests for OpenAICompatibleAdapter."""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_adapters.openai_compatible_adapter import OpenAICompatibleAdapter


class TestNormalizeMessages(unittest.TestCase):
    """Tests for _normalize_messages (no API call needed — pure logic)."""

    def setUp(self):
        with patch("llm_adapters.openai_compatible_adapter.openai.OpenAI"):
            self.adapter = OpenAICompatibleAdapter(api_key="fake")

    # ── Test 1: Standard roles pass through ────────────────────────────
    def test_normalize_standard_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = self.adapter._normalize_messages(msgs)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[2]["role"], "assistant")

    # ── Test 2: tool_output → user with prefix ─────────────────────────
    def test_normalize_tool_output_role(self):
        msgs = [
            {"role": "assistant", "content": "TOOL_CALL: ..."},
            {"role": "tool_output", "content": '{"output": "file.txt"}'},
        ]
        result = self.adapter._normalize_messages(msgs)
        tool_msg = result[1]
        self.assertEqual(tool_msg["role"], "user")
        self.assertIn("[Tool Output]:", tool_msg["content"])
        self.assertIn('{"output": "file.txt"}', tool_msg["content"])

    # ── Test 3: Unknown role → user ────────────────────────────────────
    def test_normalize_unknown_role(self):
        msgs = [{"role": "custom_role", "content": "data"}]
        result = self.adapter._normalize_messages(msgs)
        self.assertEqual(result[0]["role"], "user")

    # ── Test 4: Merge consecutive user messages ────────────────────────
    def test_normalize_merge_consecutive_user(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = self.adapter._normalize_messages(msgs)
        self.assertEqual(len(result), 1)
        self.assertIn("first", result[0]["content"])
        self.assertIn("second", result[0]["content"])

    # ── Test 5: System messages NOT merged ─────────────────────────────
    def test_normalize_no_merge_system(self):
        msgs = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
        ]
        result = self.adapter._normalize_messages(msgs)
        self.assertEqual(len(result), 2)

    # ── Test 6: Alternating preserved ──────────────────────────────────
    def test_normalize_alternating_preserved(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = self.adapter._normalize_messages(msgs)
        self.assertEqual(len(result), 3)


class TestGenerateResponse(unittest.TestCase):
    """Tests for generate_response (API is mocked)."""

    # ── Test 7: Successful response ────────────────────────────────────
    @patch("llm_adapters.openai_compatible_adapter.openai.OpenAI")
    def test_generate_response_success(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from LLM"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        adapter = OpenAICompatibleAdapter(api_key="fake")
        result = adapter.generate_response(
            [{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
        )
        self.assertEqual(result, "Hello from LLM")
        mock_client.chat.completions.create.assert_called_once()

    # ── Test 8: Error response ─────────────────────────────────────────
    @patch("llm_adapters.openai_compatible_adapter.openai.OpenAI")
    def test_generate_response_error(self, MockOpenAI):
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = Exception("connection refused")

        adapter = OpenAICompatibleAdapter(api_key="fake")
        result = adapter.generate_response([{"role": "user", "content": "Hi"}])
        self.assertIn("Error communicating with OpenAI-compatible API", result)
        self.assertIn("connection refused", result)


if __name__ == "__main__":
    unittest.main()
