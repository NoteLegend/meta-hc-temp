"""
baseline_runner.py — Run heuristic agent episodes, log trajectories,
                     and print diagnostics for the FinAudit environment.

Usage:
    python baseline_runner.py                        # 3 episodes, verbose
    python baseline_runner.py --episodes 10          # 10 episodes
    python baseline_runner.py --episodes 5 --output logs/   # save to logs/
    python baseline_runner.py --quiet                # suppress per-step output

No external dependencies — uses only the project modules.
"""

import os
import sys
import argparse
import sqlite3

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FinAuditEnv, connect_db, fetch_all_transactions
from logger import (
    EpisodeLogger,
    print_episode_trace,
    print_episode_summary,
    analyze_failures,
)


# ===========================================================================
# System prompt for LLM / TRL GRPO training
# ===========================================================================

SYSTEM_PROMPT = """You are a financial auditing agent operating inside the FinAuditEnv reinforcement learning environment.

Your goal is to inspect the transactions table, calculate totals, categorize transactions by amount, flag anomalous transactions, and submit a final structured answer. You must act only by emitting one valid JSON action object at a time. Do not include prose, Markdown, explanations, or extra keys outside the action JSON.

Categorization rules:
- amount <= 100: category is "small"
- amount <= 1000: category is "medium"
- amount <= 5000: category is "large"
- amount > 5000: category is "anomaly"

Valid action schemas:

1. Inspect visible transaction data:
{"action_type":"inspect_data","params":{}}

2. Filter transactions by an exact column value:
{"action_type":"filter_transactions","params":{"column":"tx_type","value":"credit"}}
{"action_type":"filter_transactions","params":{"column":"account_id","value":100}}

3. Calculate the total amount, optionally with an exact filter:
{"action_type":"calculate_total","params":{}}
{"action_type":"calculate_total","params":{"column":"tx_type","value":"debit"}}

4. Assign one category to one transaction:
{"action_type":"assign_category","params":{"transaction_id":1,"category":"small"}}

5. Flag one transaction as anomalous:
{"action_type":"flag_anomaly","params":{"transaction_id":1}}

6. Submit the final answer:
{"action_type":"submit_answer","params":{"total":12345.67,"flagged":[1,7,12],"categories":{"1":"small","2":"medium","3":"large"},"summary":"Inspected transactions, calculated total, categorized records, and flagged anomalies."}}

Operational rules:
- Use only these six action_type values: inspect_data, filter_transactions, calculate_total, assign_category, flag_anomaly, submit_answer.
- Use transaction IDs from observations or action results.
- Flag every transaction with amount > 5000 as an anomaly.
- Assign categories using the exact threshold rules above.
- Calculate the total before submitting.
- Submit only after inspecting data, calculating a total, assigning categories, and flagging anomalies found.
- Keep each response as exactly one JSON object matching one of the schemas above."""


# ===========================================================================
# Heuristic Agent — deterministic, no LLM calls
# ===========================================================================

class HeuristicAgent:
    """A simple rule-based agent for the FinAudit environment.

    Strategy (avoids consecutive repeats):
      1. inspect_data       — learn the schema
      2. calculate_total    — compute overall total
      3. filter credits     — explore data
      4. categorize visible transactions by amount thresholds
      5. flag any transactions with amount > $5,000
      6. submit structured answer

    Interleaves action types to avoid consecutive repeat penalties.
    """

    def __init__(self):
        self.plan = []
        self.step_idx = 0

    def reset(self, initial_obs: dict) -> None:
        """Build an action plan based on the initial observation."""
        self.step_idx = 0

        txns = initial_obs.get("transactions", [])

        # Phase 1: Explore & calculate (no consecutive repeats)
        self.plan = [
            {"action_type": "inspect_data", "params": {}},
            {"action_type": "calculate_total", "params": {}},
            {"action_type": "filter_transactions",
             "params": {"column": "tx_type", "value": "credit"}},
        ]

        # Phase 2: Interleave categorize and flag to avoid repeats
        categories = {}
        flagged = []

        for txn in txns:
            tx_id = txn.get("tx_id")
            amount = txn.get("amount", 0)
            if tx_id is None:
                continue

            cat = self._classify(amount)
            categories[tx_id] = cat

            # Add category action
            self.plan.append({
                "action_type": "assign_category",
                "params": {"transaction_id": tx_id, "category": cat},
            })

            # If anomaly, interleave a flag right after to break repeats
            if amount > 5000:
                flagged.append(tx_id)
                self.plan.append({
                    "action_type": "flag_anomaly",
                    "params": {"transaction_id": tx_id},
                })

        # Phase 3: Flag any remaining anomalies not yet flagged
        for txn in txns:
            tx_id = txn.get("tx_id")
            amount = txn.get("amount", 0)
            if amount > 5000 and tx_id not in flagged:
                flagged.append(tx_id)
                self.plan.append({
                    "action_type": "flag_anomaly",
                    "params": {"transaction_id": tx_id},
                })

        self._pending_categories = categories
        self._pending_flagged = flagged
        self._computed_total = None

    def act(self, observation: dict) -> dict:
        """Return the next action from the plan."""
        result = observation.get("last_action_result")
        if isinstance(result, dict) and "total" in result:
            self._computed_total = result["total"]

        if self.step_idx >= len(self.plan):
            return {
                "action_type": "submit_answer",
                "params": {
                    "total": self._computed_total or 0,
                    "flagged": self._pending_flagged,
                    "categories": self._pending_categories,
                    "summary": "Heuristic agent: inspected, calculated, categorized, flagged, submitted.",
                },
            }

        action = self.plan[self.step_idx]
        self.step_idx += 1
        return action

    @staticmethod
    def _classify(amount: float) -> str:
        """Classify by amount (mirrors ground truth rules)."""
        if amount <= 100:
            return "small"
        elif amount <= 1000:
            return "medium"
        elif amount <= 5000:
            return "large"
        else:
            return "anomaly"


# ===========================================================================
# Rollout loop
# ===========================================================================

def run_episodes(n_episodes: int = 3,
                 verbose: bool = True,
                 task_name: str = "finaudit_complex") -> EpisodeLogger:
    """Run n_episodes with the heuristic agent and log everything.

    Args:
        n_episodes: how many episodes to run.
        verbose: if True, print step-by-step traces.
        task_name: which task template to use.

    Returns:
        A filled EpisodeLogger with all trajectories.
    """
    env = FinAuditEnv()
    agent = HeuristicAgent()
    logger = EpisodeLogger()

    for ep in range(n_episodes):
        obs = env.reset(task_name)
        agent.reset(obs)
        done = False
        step = 0

        while not done:
            action = agent.act(obs)
            obs, reward, done = env.step(action)
            step += 1

            logger.log_step(
                episode_id=ep,
                step=step,
                action=action,
                observation=obs,
                reward=reward,
                done=done,
            )

        if verbose:
            episode_data = logger.get_episode(ep)
            print_episode_trace(episode_data)

    return logger


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run heuristic baseline episodes for FinAudit environment."
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save trajectory logs (JSON + CSV)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step trace output",
    )
    args = parser.parse_args()

    verbose = not args.quiet
    logger = run_episodes(n_episodes=args.episodes, verbose=verbose)

    # --- Print summary ---
    print("\n")
    stats = logger.summary()
    print("=" * 60)
    print("  Run Summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<25}: {v}")
    print("=" * 60)

    # --- Print one full episode sample ---
    if logger.get_episode_ids():
        print("\n--- Sample Episode Summary ---")
        print_episode_summary(logger.get_episode(0))

    # --- Failure analysis ---
    analyze_failures(logger)

    # --- Save logs ---
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        logger.save_json(os.path.join(args.output, "trajectories.json"))
        logger.save_csv(os.path.join(args.output, "trajectories.csv"))

        # Also save summary
        import json
        summary_path = os.path.join(args.output, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[LOG] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
