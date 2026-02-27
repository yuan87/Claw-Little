"""Tests for Orchestrator (all dependencies mocked)."""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestOrchestrator(unittest.TestCase):
    """
    Orchestrator tests.  Every external dependency is patched so the tests
    run without API keys, /bin/bash, or filesystem side-effects.
    """

    def _create_orchestrator(self):
        """Build an Orchestrator with all heavy deps mocked out."""
        patches = [
            patch("orchestrator.orchestrator.SessionManager"),
            patch("orchestrator.orchestrator.SafetyGuardrail"),
            patch("orchestrator.orchestrator.ToolExecutor"),
            patch("orchestrator.orchestrator.AgenticLoopExecutor"),
            patch("orchestrator.orchestrator.get_llm_adapter"),
        ]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

        MockSessionMgr = mocks[0]
        mock_sm = MockSessionMgr.return_value
        mock_sm.session_dir = "/fake/sessions"
        mock_sm.get_current_session_id.return_value = "test_session"
        mock_sm.list_sessions.return_value = []

        from orchestrator.orchestrator import Orchestrator
        orch = Orchestrator()
        return orch, mocks

    # ── Test 1: Default provider and model ─────────────────────────────
    @patch.dict(os.environ, {"DEFAULT_LLM_PROVIDER": "openai", "DEFAULT_LLM_MODEL": "gpt-4o-mini"}, clear=False)
    def test_init_default_provider(self):
        orch, _ = self._create_orchestrator()
        self.assertEqual(orch.current_llm_provider, "openai")
        self.assertEqual(orch.current_llm_model, "gpt-4o-mini")

    # ── Test 2: Loads most recent session ──────────────────────────────
    def test_init_loads_most_recent_session(self):
        patches = [
            patch("orchestrator.orchestrator.SessionManager"),
            patch("orchestrator.orchestrator.SafetyGuardrail"),
            patch("orchestrator.orchestrator.ToolExecutor"),
            patch("orchestrator.orchestrator.AgenticLoopExecutor"),
            patch("orchestrator.orchestrator.get_llm_adapter"),
            patch("os.path.exists", return_value=True),
        ]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

        MockSM = mocks[0]
        mock_sm = MockSM.return_value
        mock_sm.session_dir = "/fake/sessions"
        mock_sm.list_sessions.return_value = ["s1", "s2", "s3"]
        mock_sm.get_current_session_id.return_value = "s3"

        from orchestrator.orchestrator import Orchestrator
        orch = Orchestrator()
        mock_sm.load_session.assert_called_with("s3")

    # ── Test 3: Help output ────────────────────────────────────────────
    def test_help_output(self):
        orch, _ = self._create_orchestrator()
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            orch.print_help()
            output = mock_out.getvalue()
        self.assertIn("/llm", output)
        self.assertIn("/session", output)
        self.assertIn("/exit", output)
        self.assertIn("/help", output)
        self.assertIn("/providers", output)

    # ── Test 4: Print providers output ─────────────────────────────────
    def test_print_providers_output(self):
        orch, _ = self._create_orchestrator()
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            orch._print_providers()
            output = mock_out.getvalue()
        for provider in ["openai", "deepseek", "anthropic", "gemini", "openrouter", "anyrouter", "agentrouter"]:
            self.assertIn(provider, output)


if __name__ == "__main__":
    unittest.main()
