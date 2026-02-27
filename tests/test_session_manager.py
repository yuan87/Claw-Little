"""Tests for SessionManager."""
import sys
import os
import json
import unittest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from session_manager.session_manager import SessionManager


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sm = SessionManager(session_dir=self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # ── Test 1: Auto-generated ID ──────────────────────────────────────
    def test_create_new_session_auto_id(self):
        sid = self.sm.create_new_session()
        self.assertIsNotNone(sid)
        self.assertTrue(os.path.exists(self.sm._get_session_file_path(sid)))

    # ── Test 2: Custom ID ──────────────────────────────────────────────
    def test_create_new_session_custom_id(self):
        sid = self.sm.create_new_session("my_session")
        self.assertEqual(sid, "my_session")
        self.assertTrue(os.path.exists(self.sm._get_session_file_path(sid)))

    # ── Test 3: Duplicate raises ───────────────────────────────────────
    def test_create_duplicate_session_raises(self):
        self.sm.create_new_session("dup")
        with self.assertRaises(ValueError):
            self.sm.create_new_session("dup")

    # ── Test 4: Save and load ──────────────────────────────────────────
    def test_save_and_load_session(self):
        self.sm.create_new_session("persist")
        self.sm.add_message("user", "hello")
        self.sm.add_message("assistant", "hi")
        self.sm.save_session()

        sm2 = SessionManager(session_dir=self.tmp_dir.name)
        result = sm2.load_session("persist")
        self.assertTrue(result)
        self.assertEqual(len(sm2.get_history()), 2)
        self.assertEqual(sm2.get_history()[0], {"role": "user", "content": "hello"})
        self.assertEqual(sm2.get_history()[1], {"role": "assistant", "content": "hi"})

    # ── Test 5: Load nonexistent ───────────────────────────────────────
    def test_load_nonexistent_session(self):
        result = self.sm.load_session("nope")
        self.assertFalse(result)

    # ── Test 6: add_message persists ───────────────────────────────────
    def test_add_message_persists(self):
        self.sm.create_new_session("auto_save")
        self.sm.add_message("user", "test msg")

        sm2 = SessionManager(session_dir=self.tmp_dir.name)
        sm2.load_session("auto_save")
        self.assertEqual(len(sm2.get_history()), 1)
        self.assertEqual(sm2.get_history()[0]["content"], "test msg")

    # ── Test 7: Empty history ──────────────────────────────────────────
    def test_get_history_empty(self):
        self.sm.create_new_session("empty")
        self.assertEqual(self.sm.get_history(), [])

    # ── Test 8: List sessions ──────────────────────────────────────────
    def test_list_sessions(self):
        self.sm.create_new_session("alpha")
        self.sm.create_new_session("beta")
        self.sm.create_new_session("gamma")
        sessions = self.sm.list_sessions()
        self.assertEqual(sessions, ["alpha", "beta", "gamma"])

    # ── Test 9: Current session ID ─────────────────────────────────────
    def test_get_current_session_id(self):
        self.sm.create_new_session("current_test")
        self.assertEqual(self.sm.get_current_session_id(), "current_test")

    # ── Test 10: Save with no active session ───────────────────────────
    def test_save_no_active_session(self):
        sm_fresh = SessionManager(session_dir=self.tmp_dir.name)
        # Should not raise, just print warning
        sm_fresh.save_session()


if __name__ == "__main__":
    unittest.main()
