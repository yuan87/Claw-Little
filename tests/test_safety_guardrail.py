"""Tests for SafetyGuardrail."""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_guardrail.safety_guardrail import SafetyGuardrail


class TestSafetyGuardrail(unittest.TestCase):
    def setUp(self):
        self.guardrail = SafetyGuardrail()

    # ── Test 1: Safe command ────────────────────────────────────────────
    def test_safe_command(self):
        is_safe, message = self.guardrail.is_safe("ls -l")
        self.assertTrue(is_safe)
        self.assertEqual(message, "Command is safe.")

    # ── Test 2: Every blocked command ───────────────────────────────────
    def test_blocked_commands(self):
        for cmd in self.guardrail.blocklist:
            is_safe, message = self.guardrail.is_safe(cmd)
            self.assertFalse(is_safe, f"'{cmd}' should be blocked")
            self.assertIn(cmd, message)

    # ── Test 3: Empty command ───────────────────────────────────────────
    def test_empty_command(self):
        is_safe, message = self.guardrail.is_safe("")
        self.assertTrue(is_safe)
        self.assertEqual(message, "Command is empty.")

    # ── Test 4: Piped command (first token safe) ────────────────────────
    def test_piped_command_safe(self):
        is_safe, _ = self.guardrail.is_safe("echo hello | grep hello")
        self.assertTrue(is_safe)

    # ── Test 5: Command with arguments ──────────────────────────────────
    def test_command_with_args(self):
        is_safe, _ = self.guardrail.is_safe("cat file.txt")
        self.assertTrue(is_safe)

    # ── Test 6: Unparseable command ─────────────────────────────────────
    def test_unparseable_command(self):
        is_safe, message = self.guardrail.is_safe("echo 'unterminated")
        self.assertFalse(is_safe)
        self.assertIn("Error parsing command", message)


if __name__ == "__main__":
    unittest.main()
