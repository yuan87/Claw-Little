"""Tests for ToolExecutor (PersistentShell is mocked — no /bin/bash needed)."""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_guardrail.safety_guardrail import SafetyGuardrail


class TestParseToolCall(unittest.TestCase):
    """Tests for ToolExecutor.parse_tool_call (no shell needed)."""

    def setUp(self):
        # Patch PersistentShell so ToolExecutor.__init__ doesn't spawn /bin/bash
        patcher = patch("tool_executor.tool_executor.PersistentShell")
        self.MockShell = patcher.start()
        self.addCleanup(patcher.stop)

        from tool_executor.tool_executor import ToolExecutor
        self.te = ToolExecutor(SafetyGuardrail())

    # ── Test 1: Valid tool call ────────────────────────────────────────
    def test_parse_valid_tool_call(self):
        response = 'TOOL_CALL: {"tool_name": "execute_bash", "args": "ls -l"}'
        result = self.te.parse_tool_call(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool_name"], "execute_bash")
        self.assertEqual(result["args"], "ls -l")

    # ── Test 2: Non-tool-call response ─────────────────────────────────
    def test_parse_non_tool_call(self):
        result = self.te.parse_tool_call("Hello there!")
        self.assertIsNone(result)

    # ── Test 3: Invalid JSON ──────────────────────────────────────────
    def test_parse_invalid_json(self):
        result = self.te.parse_tool_call("TOOL_CALL: {bad json}")
        self.assertIsNone(result)

    # ── Test 4: TOOL_CALL prefix with no body ──────────────────────────
    def test_parse_tool_call_prefix_only(self):
        result = self.te.parse_tool_call("TOOL_CALL:")
        self.assertIsNone(result)


class TestExecuteTool(unittest.TestCase):
    """Tests for ToolExecutor.execute_tool (shell is mocked)."""

    def setUp(self):
        patcher = patch("tool_executor.tool_executor.PersistentShell")
        self.MockShell = patcher.start()
        self.addCleanup(patcher.stop)

        from tool_executor.tool_executor import ToolExecutor
        self.te = ToolExecutor(SafetyGuardrail())

    # ── Test 5: Safe command executes ──────────────────────────────────
    def test_execute_tool_safe_command(self):
        self.te.shell.execute.return_value = "file1.txt\nfile2.txt"
        result = self.te.execute_tool("execute_bash", "ls")
        self.assertEqual(result["returncode"], 0)
        self.assertIn("file1.txt", result["output"])
        self.te.shell.execute.assert_called_once_with("ls")

    # ── Test 6: Blocked command ────────────────────────────────────────
    def test_execute_tool_blocked_command(self):
        result = self.te.execute_tool("execute_bash", "rm -rf /")
        self.assertEqual(result["returncode"], 1)
        self.assertIn("Guardrail blocked command", result["output"])
        self.te.shell.execute.assert_not_called()

    # ── Test 7: Unknown tool ──────────────────────────────────────────
    def test_execute_tool_unknown_tool(self):
        result = self.te.execute_tool("write_file", "data")
        self.assertEqual(result["returncode"], 1)
        self.assertIn("Unknown tool", result["output"])


if __name__ == "__main__":
    unittest.main()
