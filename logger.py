"""
logger.py — Trajectory logging, debug utilities, and failure analysis
for the FinAudit environment.

No external dependencies beyond the standard library.

Usage:
    from logger import EpisodeLogger, print_episode_trace, analyze_failures

    log = EpisodeLogger()
    log.log_step(episode_id=0, step=1, action={...}, observation={...}, reward=0.01, done=False)
    ...
    log.save_json("logs/trajectories.json")
    log.save_csv("logs/trajectories.csv")
    analyze_failures(log)
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional


# ===========================================================================
# EpisodeLogger — structured trajectory storage
# ===========================================================================

class EpisodeLogger:
    """Collects step-level data across multiple episodes."""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []  # flat list of all step records

    def log_step(self,
                 episode_id: int,
                 step: int,
                 action: dict,
                 observation: dict,
                 reward: float,
                 done: bool) -> None:
        """Record a single step."""
        self.steps.append({
            "episode_id": episode_id,
            "step": step,
            "action_type": action.get("action_type", ""),
            "action_params": json.dumps(action.get("params", {}), default=str),
            "reward": round(reward, 4),
            "done": done,
            # Store a compact version of the observation (skip full txn rows)
            "last_action_result": json.dumps(
                observation.get("last_action_result"), default=str
            ),
            "steps_remaining": observation.get("steps_remaining"),
            "error": observation.get("error"),
        })

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_episode(self, episode_id: int) -> List[Dict[str, Any]]:
        """Return all steps for a given episode."""
        return [s for s in self.steps if s["episode_id"] == episode_id]

    def get_episode_ids(self) -> List[int]:
        """Return sorted list of unique episode IDs."""
        return sorted(set(s["episode_id"] for s in self.steps))

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all episodes."""
        episode_ids = self.get_episode_ids()
        if not episode_ids:
            return {"episodes": 0, "total_steps": 0}

        episode_rewards = {}
        episode_steps = {}
        for s in self.steps:
            eid = s["episode_id"]
            episode_steps[eid] = episode_steps.get(eid, 0) + 1
            # Final reward is the reward on the last (done=True) step
            if s["done"]:
                episode_rewards[eid] = s["reward"]

        rewards = list(episode_rewards.values())
        steps = list(episode_steps.values())

        return {
            "episodes": len(episode_ids),
            "total_steps": len(self.steps),
            "avg_steps_per_episode": round(sum(steps) / len(steps), 1),
            "total_reward": round(sum(rewards), 4),
            "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "min_reward": round(min(rewards), 4) if rewards else 0.0,
            "max_reward": round(max(rewards), 4) if rewards else 0.0,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_json(self, path: str) -> None:
        """Export all step records to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=2, default=str)
        print(f"[LOG] Saved {len(self.steps)} steps to {path}")

    def save_csv(self, path: str) -> None:
        """Export all step records to a CSV file."""
        if not self.steps:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = list(self.steps[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.steps)
        print(f"[LOG] Saved {len(self.steps)} steps to {path}")


# ===========================================================================
# Debug utilities
# ===========================================================================

def print_episode_trace(episode: List[Dict[str, Any]]) -> None:
    """Print a detailed step-by-step execution trace for one episode.

    Highlights:
      ⚠  invalid actions (negative reward on non-submit steps)
      🔁 repeated actions (same action_type back-to-back)
      📈 reward spikes (reward > 0.1 on a single step)
    """
    if not episode:
        print("  (empty episode)")
        return

    eid = episode[0]["episode_id"]
    print(f"\n{'='*60}")
    print(f"  Episode {eid} — {len(episode)} steps")
    print(f"{'='*60}")

    prev_action = None
    for s in episode:
        flags = []

        # Highlight invalid actions
        if s["reward"] < 0 and not s["done"]:
            flags.append("⚠ INVALID")

        # Highlight repetition
        if s["action_type"] == prev_action:
            flags.append("🔁 REPEAT")

        # Highlight reward spikes
        if s["reward"] > 0.1:
            flags.append("📈 SPIKE")

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  step {s['step']:>2} | "
            f"{s['action_type']:<22} | "
            f"reward={s['reward']:>+7.4f} | "
            f"done={str(s['done']):<5}"
            f"{flag_str}"
        )
        prev_action = s["action_type"]

    # Final reward
    final = episode[-1]
    print(f"  {'─'*56}")
    print(f"  Final reward: {final['reward']:.4f}")


def print_episode_summary(episode: List[Dict[str, Any]]) -> None:
    """Print a compact summary of one episode: actions taken and rewards."""
    if not episode:
        print("  (empty episode)")
        return

    eid = episode[0]["episode_id"]
    actions = [s["action_type"] for s in episode]
    rewards = [s["reward"] for s in episode]
    final_reward = episode[-1]["reward"] if episode[-1]["done"] else 0.0

    print(f"  Episode {eid}: {len(episode)} steps, final_reward={final_reward:.4f}")
    print(f"    Actions: {' → '.join(actions)}")
    print(f"    Step rewards: {', '.join(f'{r:+.4f}' for r in rewards)}")


# ===========================================================================
# Failure analysis
# ===========================================================================

def analyze_failures(logger: EpisodeLogger) -> Dict[str, Any]:
    """Detect problematic episodes and return a diagnostic report.

    Checks for:
      - Zero-reward episodes
      - Early termination (< 3 steps)
      - Repeated-action loops (same action > 3 times consecutively)

    Returns a dict with lists of flagged episode IDs and prints a report.
    """
    zero_reward = []
    early_termination = []
    action_loops = []

    for eid in logger.get_episode_ids():
        ep = logger.get_episode(eid)
        if not ep:
            continue

        # Check zero reward
        final = ep[-1]
        if final["done"] and final["reward"] <= 0.0:
            zero_reward.append(eid)

        # Check early termination
        if len(ep) < 3:
            early_termination.append(eid)

        # Check repeated-action loops
        max_consecutive = 1
        current_run = 1
        for i in range(1, len(ep)):
            if ep[i]["action_type"] == ep[i - 1]["action_type"]:
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 1
        if max_consecutive > 3:
            action_loops.append(eid)

    # Print report
    print(f"\n{'='*60}")
    print("  Failure Analysis Report")
    print(f"{'='*60}")
    total = len(logger.get_episode_ids())
    print(f"  Total episodes analyzed: {total}")
    print(f"  Zero-reward episodes   : {len(zero_reward)}  {zero_reward or ''}")
    print(f"  Early termination (<3) : {len(early_termination)}  {early_termination or ''}")
    print(f"  Repeated-action loops  : {len(action_loops)}  {action_loops or ''}")
    healthy = total - len(set(zero_reward + early_termination + action_loops))
    print(f"  Healthy episodes       : {healthy}/{total}")
    print(f"{'='*60}")

    return {
        "zero_reward": zero_reward,
        "early_termination": early_termination,
        "action_loops": action_loops,
    }
