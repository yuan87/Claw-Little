"""Tests for AgenticLoopExecutor."""
import sys
import os
import unittest
from unittest.mock import MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agentic_loop.agentic_loop_executor import AgenticLoopExecutor


class TestAgenticLoopExecutor(unittest.TestCase):

    def _make_executor(self, llm_responses):
        """
        Create an AgenticLoopExecutor with a mock adapter that returns
        *llm_responses* in order on successive calls.
        """
        mock_adapter = MagicMock()
        mock_adapter.generate_response.side_effect = llm_responses

        mock_tool_executor = MagicMock()
        # parse_tool_call: delegate to real logic for realism, but we
        # control it via side_effect below when needed.
        mock_tool_executor.parse_tool_call.side_effect = self._default_parse

        return AgenticLoopExecutor(mock_adapter, mock_tool_executor)

    @staticmethod
    def _default_parse(response: str):
        """Minimal parse matching the real implementation."""
        import json
        if response.startswith("TOOL_CALL:"):
            try:
                return json.loads(response.replace("TOOL_CALL:", "").strip())
            except Exception:
                return None
        return None

    # ── Test 1: Plain text response ────────────────────────────────────
    def test_plain_response(self):
        executor = self._make_executor(["Here is your answer."])
        result = executor.run_agentic_loop([{"role": "user", "content": "Hi"}])
        self.assertEqual(result, "Here is your answer.")
        executor.tool_executor.execute_tool.assert_not_called()

    # ── Test 2: Tool call → then plain response ────────────────────────
    def test_tool_call_then_response(self):
        tool_call = 'TOOL_CALL: {"tool_name": "execute_bash", "args": "ls"}'
        executor = self._make_executor([tool_call, "Found 3 files."])
        executor.tool_executor.execute_tool.return_value = {
            "output": "a.txt\nb.txt\nc.txt", "returncode": 0
        }

        result = executor.run_agentic_loop([{"role": "user", "content": "list files"}])
        self.assertEqual(result, "Found 3 files.")
        executor.tool_executor.execute_tool.assert_called_once_with("execute_bash", "ls")

    # ── Test 3: System prompt injected ─────────────────────────────────
    def test_system_prompt_injected(self):
        executor = self._make_executor(["ok"])
        msgs = [{"role": "user", "content": "test"}]
        executor.run_agentic_loop(msgs)
        # First message should be system prompt
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("AI assistant", msgs[0]["content"])

    # ── Test 4: System prompt NOT duplicated ───────────────────────────
    def test_system_prompt_not_duplicated(self):
        executor = self._make_executor(["ok"])
        msgs = [
            {"role": "system", "content": "Custom system prompt"},
            {"role": "user", "content": "test"},
        ]
        executor.run_agentic_loop(msgs)
        # Should still be exactly 2 messages (system + user), not 3
        system_msgs = [m for m in msgs if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0]["content"], "Custom system prompt")

    # ── Test 5: Model passed to adapter ────────────────────────────────
    def test_model_passed_to_adapter(self):
        executor = self._make_executor(["done"])
        executor.run_agentic_loop(
            [{"role": "user", "content": "hi"}],
            model="deepseek-chat",
        )
        call_kwargs = executor.llm_adapter.generate_response.call_args
        self.assertEqual(call_kwargs.kwargs.get("model") or call_kwargs[1].get("model"), "deepseek-chat")

    # ── Test 6: Unknown tool → error string ────────────────────────────
    def test_unknown_tool_error(self):
        bad_call = 'TOOL_CALL: {"tool_name": "write_file", "args": "data"}'
        executor = self._make_executor([bad_call])
        result = executor.run_agentic_loop([{"role": "user", "content": "go"}])
        self.assertIn("Error", result)
        self.assertIn("Unknown tool", result)


if __name__ == "__main__":
    unittest.main()
