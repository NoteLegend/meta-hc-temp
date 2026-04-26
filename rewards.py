"""
rewards.py — Multi-component reward system for the FinAudit environment.

All reward, penalty, and verification logic lives here.
No external dependencies beyond sqlite3.

ARCHITECTURE:
  Step rewards  = behavioral shaping only (small signals per action)
  Final reward  = correctness evaluation (computed only at submit_answer)

STEP REWARDS:
  - progress_reward     : small positive for meaningful first-time actions
  - no_state_change_penalty : penalty when action produces no new information
  - invalid_action_penalty  : escalating penalty for bad actions
  - consecutive_repeat_penalty : escalating penalty for same action in a row
  - efficiency_penalty  : small per-step cost to encourage shorter episodes

FINAL REWARD (at submit_answer):
  - categorization accuracy   (weight 0.3)
  - anomaly detection F1      (weight 0.3)
  - reconciliation exact match(weight 0.4)
  - minus penalty deductions

ANTI-HACKING:
  - action dependency check before submit
  - cheating penalty for skipping reasoning steps
"""

import sqlite3
from typing import Dict, List, Optional, Set


# ===========================================================================
# Verifier functions — each returns a float in [0.0, 1.0]
# ===========================================================================

def check_categorization(db_path: str, agent_categories: Dict[int, str]) -> float:
    """Compare agent's category assignments against ground truth.

    Returns proportion of correctly labeled transactions (0.0–1.0).
    Returns 0.0 if agent submitted no categories.
    """
    if not agent_categories:
        return 0.0

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT tx_id, category FROM ground_truth_categories")
        truth = dict(c.fetchall())
        conn.close()
    except sqlite3.Error:
        return 0.0

    correct = 0
    total = len(agent_categories)
    for tx_id, label in agent_categories.items():
        agent_label = str(label).strip().lower()
        true_label = str(truth.get(int(tx_id), "")).strip().lower()
        if agent_label == true_label:
            correct += 1

    return correct / total if total > 0 else 0.0


def check_anomalies(db_path: str, agent_flagged: List[int]) -> float:
    """Compare agent's flagged anomalies against ground truth using F1 score.

    Returns F1 (harmonic mean of precision and recall), 0.0–1.0.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT tx_id FROM ground_truth_anomalies WHERE is_anomaly = 1")
        true_anomalies = set(row[0] for row in c.fetchall())
        conn.close()
    except sqlite3.Error:
        return 0.0

    if not true_anomalies or not agent_flagged:
        return 0.0

    agent_set = set(int(x) for x in agent_flagged)
    true_positives = len(agent_set & true_anomalies)
    precision = true_positives / len(agent_set) if agent_set else 0.0
    recall = true_positives / len(true_anomalies) if true_anomalies else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def check_reconciliation(db_path: str, agent_total: Optional[float], target_tx_type: str) -> float:
    """Check if agent's total matches DB total for the specific transaction type.

    Returns 1.0 if exact match (within $0.01), else 0.0.
    """
    if agent_total is None:
        return 0.0

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        # CHANGED: Added WHERE clause to filter by target_tx_type
        c.execute("SELECT SUM(amount) FROM transactions WHERE tx_type = ?", (target_tx_type,))
        true_total = c.fetchone()[0]
        conn.close()
    except sqlite3.Error:
        return 0.0

    if true_total is None:
        # Handle cases where there are 0 transactions of that type
        true_total = 0.0 

    return 1.0 if abs(float(agent_total) - float(true_total)) < 0.01 else 0.0


# ===========================================================================
# Step-level rewards — behavioral shaping signals
# ===========================================================================

def progress_reward(action_type: str, completed_actions: Set[str]) -> float:
    """Positive reward ONLY for first-time meaningful actions.

    Meaningful actions: inspect_data, calculate_total
    These earn a small reward the FIRST time they are performed.
    Repeated calls earn 0.

    Args:
        action_type: the action being taken.
        completed_actions: set of action_types already performed this episode.

    Returns:
        +0.02 for first-time meaningful action, 0.0 otherwise.
    """
    MEANINGFUL = {"inspect_data", "calculate_total"}
    if action_type in MEANINGFUL and action_type not in completed_actions:
        return 0.02
    return 0.0


def categorization_step_reward(db_path: str, tx_id: int, category: str) -> float:
    """Small reward if the agent assigns the CORRECT category to a transaction.

    Returns +0.01 if correct, -0.005 if incorrect.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT category FROM ground_truth_categories WHERE tx_id = ?", (int(tx_id),))
        row = c.fetchone()
        conn.close()
    except sqlite3.Error:
        return 0.0

    if row is None:
        return 0.0

    true_label = str(row[0]).strip().lower()
    agent_label = str(category).strip().lower()
    return 0.01 if agent_label == true_label else -0.005


def flag_step_reward(db_path: str, tx_id: int) -> float:
    """Small reward if the agent correctly flags an anomalous transaction.
    MASSIVE reward (+1.5) if it catches the Smurfing pattern.
    Returns +0.01 if standard anomaly, -0.005 if false positive.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        # CHANGED: Select anomaly_type alongside is_anomaly
        c.execute("SELECT is_anomaly, anomaly_type FROM ground_truth_anomalies WHERE tx_id = ?", (int(tx_id),))
        row = c.fetchone()
        conn.close()
    except sqlite3.Error:
        return 0.0

    if row is None:
        return -0.005  # unknown tx_id → wrong flag

    is_anomaly = row[0]
    anomaly_type = row[1]

    if is_anomaly == 1:
        # THE SMURF CATCHER
        if anomaly_type == "smurfing_pattern":
            return 1.5  
        else:
            return 0.01 # Standard >$5000 anomaly
    else:
        return -0.005   # Wrong flag penalty


# ===========================================================================
# Penalties
# ===========================================================================

def efficiency_penalty() -> float:
    """Per-step penalty to encourage shorter trajectories.

    Applied every step. -0.01 per step.
    """
    return -0.01


def invalid_action_penalty(invalid_count: int) -> float:
    """Escalating penalty for invalid/malformed actions.

    Args:
        invalid_count: how many invalid actions have been taken so far.

    Returns:
        -0.02 for first, -0.05 for second, -0.1 for third+.
    """
    if invalid_count <= 1:
        return -0.02
    elif invalid_count <= 2:
        return -0.05
    else:
        return -0.1


def consecutive_repeat_penalty(consecutive_count: int) -> float:
    """Escalating penalty for the SAME action type repeated consecutively.

    Args:
        consecutive_count: how many times in a row this action has been taken.

    Returns:
        0.0 for first occurrence, -0.02 at 2, -0.05 at 3, blocked at >3.
        Returns None if the action should be BLOCKED (>3 consecutive repeats).
    """
    if consecutive_count <= 1:
        return 0.0
    elif consecutive_count == 2:
        return -0.02
    elif consecutive_count == 3:
        return -0.05
    else:
        return None  # signal to block execution


def no_state_change_penalty() -> float:
    """Penalty when an action produces no meaningful state change.

    E.g., flagging an already-flagged transaction, or re-categorizing
    with the same label.
    """
    return -0.01


# ===========================================================================
# Action dependency enforcement
# ===========================================================================

# Required actions before submit_answer is allowed
REQUIRED_BEFORE_SUBMIT = {"inspect_data", "calculate_total"}

def check_action_dependencies(completed_actions: set) -> dict:
    """Verify agent has performed required actions before submitting.

    Requirements:
      - get_audit_policy must have been called
      - inspect_data must have been called
      - calculate_total must have been called
      - at least one of: assign_category OR flag_anomaly

    Returns:
        {"satisfied": True/False, "missing": [list of missing requirements]}
    """
    missing = []

    # NEW: Mandatory Policy Check
    if "get_audit_policy" not in completed_actions:
        missing.append("get_audit_policy")

    # Your existing loop for inspect_data and calculate_total
    for req in REQUIRED_BEFORE_SUBMIT:
        if req not in completed_actions:
            missing.append(req)

    has_analysis = ("assign_category" in completed_actions or
                    "flag_anomaly" in completed_actions)
    if not has_analysis:
        missing.append("assign_category OR flag_anomaly")

    return {"satisfied": len(missing) == 0, "missing": missing}


def cheating_penalty(completed_actions: Set[str]) -> float:
    """Strong penalty if agent submits without required prior reasoning.

    Returns -0.5 if dependencies not met, else 0.0.
    """
    deps = check_action_dependencies(completed_actions)
    if not deps["satisfied"]:
        return -0.5
    return 0.0


# ===========================================================================
# Final reward aggregator — correctness evaluation at submit_answer
# ===========================================================================

# Clean 3-component weights (sum = 1.0)
REWARD_WEIGHTS = {
    "reconciliation":  0.4,
    "anomaly":         0.3,
    "categorization":  0.3,
}


def compute_final_reward(
    db_path: str,
    categories: dict,
    flagged: list,
    submitted_total: Optional[float],
    completed_actions: Set[str],
    cumulative_step_reward: float,
    target_tx_type: str, # <-- 1. ADDED TARGET TYPE HERE
) -> dict:
    """Compute the final reward at submit_answer.

    Final reward = weighted correctness score + penalty deductions.
    Clamped to [0.0, 1.0].
    """
    # --- Verifier scores (each 0.0–1.0) ---
    cat_score = check_categorization(db_path, categories)
    anom_score = check_anomalies(db_path, flagged)
    
    # <-- 2. PASSED TARGET TYPE TO THE MATH VERIFIER
    recon_score = check_reconciliation(db_path, submitted_total, target_tx_type) 

    # --- Weighted correctness score ---
    correctness = (
        REWARD_WEIGHTS["reconciliation"] * recon_score +
        REWARD_WEIGHTS["anomaly"] * anom_score +
        REWARD_WEIGHTS["categorization"] * cat_score
    )

    # --- Penalty deductions ---
    cheat = cheating_penalty(completed_actions)

    # Clamp cumulative step penalties so they don't collapse the reward
    # entirely (cap penalty deduction at -0.3)
    step_penalty = max(-0.3, cumulative_step_reward)

    penalty_total = cheat + step_penalty

    # --- Combine and clamp ---
    raw_total = correctness + penalty_total
    final = max(0.0, min(1.0, raw_total))

    return {
        "breakdown": {
            "categorization": round(REWARD_WEIGHTS["categorization"] * cat_score, 4),
            "anomaly": round(REWARD_WEIGHTS["anomaly"] * anom_score, 4),
            "reconciliation": round(REWARD_WEIGHTS["reconciliation"] * recon_score, 4),
        },
        "scores": {
            "categorization_raw": round(cat_score, 4),
            "anomaly_raw": round(anom_score, 4),
            "reconciliation_raw": round(recon_score, 4),
        },
        "correctness": round(correctness, 4),
        "cheating_penalty": round(cheat, 4),
        "step_penalty": round(step_penalty, 4),
        "penalty_total": round(penalty_total, 4),
        "raw_total": round(raw_total, 4),
        "final_reward": round(final, 4),
    }
