"""
Smoke tests for the FinAuditEnv structured environment with rewards.

Run with:
    python test_finaudit.py -v

No external dependencies required — uses the built-in unittest module.
"""

import os
import sys
import unittest
import sqlite3

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FinAuditEnv, connect_db, fetch_all_transactions, fetch_limited_transactions
from rewards import (
    check_categorization, check_anomalies, check_reconciliation,
    progress_reward, categorization_step_reward, flag_step_reward,
    efficiency_penalty, invalid_action_penalty,
    consecutive_repeat_penalty, no_state_change_penalty,
    check_action_dependencies, cheating_penalty,
    compute_final_reward,
)


# ===========================================================================
# DB Helper Tests
# ===========================================================================

class TestDBHelpers(unittest.TestCase):

    DB_PATH = "test_helpers.db"

    def setUp(self):
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS transactions "
                  "(id INTEGER PRIMARY KEY, account_id INTEGER, type TEXT, amount REAL)")
        c.execute("DELETE FROM transactions")
        for i in range(10):
            c.execute("INSERT INTO transactions (account_id, type, amount) VALUES (?, ?, ?)",
                      (100 + i % 3, "credit" if i % 2 == 0 else "debit", float(i * 100)))
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

    def test_connect_db(self):
        conn = connect_db(self.DB_PATH)
        self.assertIsNotNone(conn)
        conn.close()

    def test_fetch_all_transactions(self):
        conn = connect_db(self.DB_PATH)
        rows = fetch_all_transactions(conn)
        conn.close()
        self.assertEqual(len(rows), 10)

    def test_fetch_limited_default(self):
        conn = connect_db(self.DB_PATH)
        rows = fetch_limited_transactions(conn)
        conn.close()
        self.assertLessEqual(len(rows), 5)


# ===========================================================================
# Verifier Tests
# ===========================================================================

class TestVerifiers(unittest.TestCase):

    DB_PATH = "test_verifiers.db"

    def setUp(self):
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        c.execute("CREATE TABLE transactions (tx_id INTEGER PRIMARY KEY, amount REAL)")
        c.execute("CREATE TABLE ground_truth_categories (tx_id INTEGER PRIMARY KEY, category TEXT)")
        c.execute("CREATE TABLE ground_truth_anomalies (tx_id INTEGER PRIMARY KEY, is_anomaly INTEGER)")
        for i in range(1, 6):
            c.execute("INSERT INTO transactions VALUES (?, ?)", (i, float(i * 100)))
            c.execute("INSERT INTO ground_truth_categories VALUES (?, ?)", (i, "medium"))
            c.execute("INSERT INTO ground_truth_anomalies VALUES (?, ?)", (i, 1 if i == 5 else 0))
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

    def test_categorization_perfect(self):
        cats = {1: "medium", 2: "medium", 3: "medium", 4: "medium", 5: "medium"}
        self.assertAlmostEqual(check_categorization(self.DB_PATH, cats), 1.0)

    def test_categorization_partial(self):
        cats = {1: "medium", 2: "WRONG", 3: "medium"}
        self.assertAlmostEqual(check_categorization(self.DB_PATH, cats), 2.0 / 3.0, places=4)

    def test_categorization_empty(self):
        self.assertEqual(check_categorization(self.DB_PATH, {}), 0.0)

    def test_anomalies_perfect(self):
        self.assertAlmostEqual(check_anomalies(self.DB_PATH, [5]), 1.0)

    def test_anomalies_partial(self):
        score = check_anomalies(self.DB_PATH, [4, 5])
        self.assertAlmostEqual(score, 2 / 3, places=3)

    def test_anomalies_none_flagged(self):
        self.assertEqual(check_anomalies(self.DB_PATH, []), 0.0)

    def test_anomalies_all_wrong(self):
        self.assertEqual(check_anomalies(self.DB_PATH, [1, 2, 3]), 0.0)

    def test_reconciliation_exact(self):
        self.assertEqual(check_reconciliation(self.DB_PATH, 1500.0), 1.0)

    def test_reconciliation_wrong(self):
        self.assertEqual(check_reconciliation(self.DB_PATH, 9999.0), 0.0)

    def test_reconciliation_none(self):
        self.assertEqual(check_reconciliation(self.DB_PATH, None), 0.0)


# ===========================================================================
# Step Reward & Penalty Tests
# ===========================================================================

class TestStepRewards(unittest.TestCase):

    def test_progress_reward_first_time(self):
        # First time inspect_data → positive reward
        self.assertGreater(progress_reward("inspect_data", set()), 0)

    def test_progress_reward_repeated(self):
        # Already done inspect_data → 0
        self.assertEqual(progress_reward("inspect_data", {"inspect_data"}), 0.0)

    def test_progress_reward_non_meaningful(self):
        # assign_category is not a "meaningful" progress action
        self.assertEqual(progress_reward("assign_category", set()), 0.0)

    def test_efficiency_penalty(self):
        self.assertLess(efficiency_penalty(), 0)

    def test_invalid_action_escalating(self):
        p1 = invalid_action_penalty(1)
        p2 = invalid_action_penalty(2)
        p3 = invalid_action_penalty(3)
        self.assertLess(p1, 0)
        self.assertLess(p2, p1)  # more negative
        self.assertLess(p3, p2)

    def test_consecutive_repeat_none(self):
        self.assertEqual(consecutive_repeat_penalty(1), 0.0)

    def test_consecutive_repeat_penalty(self):
        self.assertLess(consecutive_repeat_penalty(2), 0)
        self.assertLess(consecutive_repeat_penalty(3), consecutive_repeat_penalty(2))

    def test_consecutive_repeat_blocked(self):
        self.assertIsNone(consecutive_repeat_penalty(4))

    def test_no_state_change_penalty(self):
        self.assertLess(no_state_change_penalty(), 0)

    def test_categorization_step_reward_correct(self):
        # This test uses the verifier DB
        DB = "test_steprew.db"
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("CREATE TABLE ground_truth_categories (tx_id INTEGER PRIMARY KEY, category TEXT)")
        c.execute("INSERT INTO ground_truth_categories VALUES (1, 'medium')")
        conn.commit()
        conn.close()
        try:
            self.assertGreater(categorization_step_reward(DB, 1, "medium"), 0)
            self.assertLess(categorization_step_reward(DB, 1, "wrong"), 0)
        finally:
            os.remove(DB)

    def test_flag_step_reward_correct(self):
        DB = "test_flagrew.db"
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("CREATE TABLE ground_truth_anomalies (tx_id INTEGER PRIMARY KEY, is_anomaly INTEGER)")
        c.execute("INSERT INTO ground_truth_anomalies VALUES (1, 1)")
        c.execute("INSERT INTO ground_truth_anomalies VALUES (2, 0)")
        conn.commit()
        conn.close()
        try:
            self.assertGreater(flag_step_reward(DB, 1), 0)  # true anomaly
            self.assertLess(flag_step_reward(DB, 2), 0)     # not anomaly
        finally:
            os.remove(DB)


# ===========================================================================
# Action Dependency Tests
# ===========================================================================

class TestDependencies(unittest.TestCase):

    def test_all_satisfied(self):
        deps = check_action_dependencies({"inspect_data", "calculate_total", "assign_category"})
        self.assertTrue(deps["satisfied"])
        self.assertEqual(deps["missing"], [])

    def test_missing_inspect(self):
        deps = check_action_dependencies({"calculate_total", "flag_anomaly"})
        self.assertFalse(deps["satisfied"])
        self.assertIn("inspect_data", deps["missing"])

    def test_missing_analysis(self):
        deps = check_action_dependencies({"inspect_data", "calculate_total"})
        self.assertFalse(deps["satisfied"])

    def test_cheating_penalty_triggered(self):
        self.assertEqual(cheating_penalty(set()), -0.5)

    def test_cheating_penalty_not_triggered(self):
        self.assertEqual(
            cheating_penalty({"inspect_data", "calculate_total", "flag_anomaly"}),
            0.0
        )


# ===========================================================================
# Aggregator Tests
# ===========================================================================

class TestAggregator(unittest.TestCase):

    DB_PATH = "test_agg.db"

    def setUp(self):
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        c.execute("CREATE TABLE transactions (tx_id INTEGER PRIMARY KEY, amount REAL)")
        c.execute("CREATE TABLE ground_truth_categories (tx_id INTEGER PRIMARY KEY, category TEXT)")
        c.execute("CREATE TABLE ground_truth_anomalies (tx_id INTEGER PRIMARY KEY, is_anomaly INTEGER)")
        c.execute("INSERT INTO transactions VALUES (1, 100.0)")
        c.execute("INSERT INTO ground_truth_categories VALUES (1, 'medium')")
        c.execute("INSERT INTO ground_truth_anomalies VALUES (1, 1)")
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

    def test_perfect_score(self):
        result = compute_final_reward(
            self.DB_PATH,
            categories={1: "medium"},
            flagged=[1],
            submitted_total=100.0,
            completed_actions={"inspect_data", "calculate_total", "assign_category"},
            cumulative_step_reward=0.0,
        )
        self.assertGreater(result["final_reward"], 0.8)

    def test_zero_score(self):
        result = compute_final_reward(
            self.DB_PATH,
            categories={}, flagged=[], submitted_total=None,
            completed_actions=set(), cumulative_step_reward=-0.5,
        )
        self.assertEqual(result["final_reward"], 0.0)

    def test_clamped_to_unit(self):
        result = compute_final_reward(
            self.DB_PATH,
            categories={1: "medium"}, flagged=[1], submitted_total=100.0,
            completed_actions={"inspect_data", "calculate_total", "assign_category"},
            cumulative_step_reward=5.0,
        )
        self.assertLessEqual(result["final_reward"], 1.0)


# ===========================================================================
# FinAuditEnv Integration Tests
# ===========================================================================

class TestFinAuditEnv(unittest.TestCase):

    def setUp(self):
        self.env = FinAuditEnv(db_path="test_finaudit.db")

    def tearDown(self):
        if os.path.exists("test_finaudit.db"):
            os.remove("test_finaudit.db")

    def _reset(self):
        return self.env.reset("finaudit_complex")

    def test_reset_returns_dict(self):
        obs = self._reset()
        self.assertIsInstance(obs, dict)

    def test_reset_clears_state(self):
        self._reset()
        self.env.step({"action_type": "flag_anomaly", "params": {"transaction_id": 1}})
        self._reset()
        self.assertEqual(self.env.steps_taken, 0)
        self.assertEqual(self.env.flagged, [])
        self.assertEqual(self.env.completed_action_types, set())
        self.assertEqual(self.env.consecutive_repeat_count, 0)

    def test_step_returns_3_tuple(self):
        self._reset()
        result = self.env.step({"action_type": "inspect_data", "params": {}})
        self.assertEqual(len(result), 3)
        obs, reward, done = result
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_no_flat_reward_for_neutral_actions(self):
        """After the first inspect, a second inspect should NOT get +0.01."""
        self._reset()
        _, r1, _ = self.env.step({"action_type": "inspect_data", "params": {}})
        _, r2, _ = self.env.step({"action_type": "inspect_data", "params": {}})
        # r1 includes progress reward, r2 should NOT (already done)
        # r2 should be lower due to repeat penalty and no progress
        self.assertGreater(r1, r2)

    def test_invalid_action_negative(self):
        self._reset()
        _, reward, _ = self.env.step({"action_type": "bad", "params": {}})
        self.assertLess(reward, 0)

    def test_invalid_action_escalates(self):
        self._reset()
        _, r1, _ = self.env.step({"action_type": "bad1", "params": {}})
        _, r2, _ = self.env.step({"action_type": "bad2", "params": {}})
        _, r3, _ = self.env.step({"action_type": "bad3", "params": {}})
        self.assertLess(r3, r1)  # escalating

    def test_consecutive_repeat_blocked(self):
        """Action blocked after >3 consecutive repeats."""
        self._reset()
        for i in range(5):
            obs, reward, done = self.env.step({"action_type": "inspect_data", "params": {}})
        # The 4th+ should be blocked with error
        result = obs.get("last_action_result", {})
        self.assertIn("blocked", str(result.get("error", "")).lower())

    def test_ground_truth_hidden_from_inspect(self):
        self._reset()
        obs, _, _ = self.env.step({"action_type": "inspect_data", "params": {}})
        tables = obs["last_action_result"].get("tables", [])
        for t in tables:
            self.assertFalse(t.startswith("ground_truth"))

    def test_submit_rejected_without_dependencies(self):
        """Submit should be REJECTED (not end episode) if deps not met."""
        self._reset()
        obs, reward, done = self.env.step({
            "action_type": "submit_answer",
            "params": {"total": 100, "flagged": [], "categories": {}},
        })
        # Episode should NOT be done — submit was rejected
        self.assertFalse(done)
        self.assertLess(reward, 0)  # cheating penalty
        self.assertIn("Missing", obs["last_action_result"]["error"])

    def test_submit_accepted_with_dependencies(self):
        """Submit works after completing required actions."""
        self._reset()
        self.env.step({"action_type": "inspect_data", "params": {}})
        self.env.step({"action_type": "calculate_total", "params": {}})
        self.env.step({"action_type": "assign_category",
                       "params": {"transaction_id": 1, "category": "large"}})

        obs, reward, done = self.env.step({
            "action_type": "submit_answer",
            "params": {"total": 17760737.10, "flagged": [], "categories": {1: "large"}},
        })
        self.assertTrue(done)
        self.assertGreater(reward, 0)
        self.assertIn("reward_breakdown", obs)

    def test_max_steps_enforcement(self):
        self._reset()
        self.env.MAX_STEPS = 5
        for _ in range(10):
            _, _, done = self.env.step({"action_type": "inspect_data", "params": {}})
            if done:
                break
        self.assertTrue(done)

    def test_early_termination_on_invalid_streak(self):
        self._reset()
        self.env.MAX_INVALID_STREAK = 3
        for _ in range(5):
            _, _, done = self.env.step({"action_type": f"bad", "params": {}})
            if done:
                break
        self.assertTrue(done)

    def test_state_change_penalty_on_reflag(self):
        """Re-flagging the same tx should incur no-state-change penalty."""
        self._reset()
        _, r1, _ = self.env.step({"action_type": "flag_anomaly", "params": {"transaction_id": 1}})
        _, r2, _ = self.env.step({"action_type": "flag_anomaly", "params": {"transaction_id": 1}})
        self.assertLess(r2, r1)  # second flag is penalized

    def test_observation_has_steps_remaining(self):
        self._reset()
        obs, _, _ = self.env.step({"action_type": "inspect_data", "params": {}})
        self.assertIn("steps_remaining", obs)
        self.assertEqual(obs["steps_remaining"], self.env.MAX_STEPS - 1)

    def test_reward_variability(self):
        """Incorrect answers should produce low reward, correct = high."""
        self._reset()
        # Do required actions
        self.env.step({"action_type": "inspect_data", "params": {}})
        self.env.step({"action_type": "calculate_total", "params": {}})
        self.env.step({"action_type": "assign_category",
                       "params": {"transaction_id": 1, "category": "wrong"}})
        # Submit with wrong values
        _, bad_reward, _ = self.env.step({
            "action_type": "submit_answer",
            "params": {"total": 0, "flagged": [999], "categories": {1: "wrong"}},
        })

        # Now run a "good" episode
        self._reset()
        self.env.step({"action_type": "inspect_data", "params": {}})
        self.env.step({"action_type": "calculate_total", "params": {}})
        self.env.step({"action_type": "assign_category",
                       "params": {"transaction_id": 1, "category": "large"}})
        _, good_reward, _ = self.env.step({
            "action_type": "submit_answer",
            "params": {"total": 17760737.10, "flagged": [], "categories": {1: "large"}},
        })

        self.assertGreater(good_reward, bad_reward)


if __name__ == "__main__":
    unittest.main()
