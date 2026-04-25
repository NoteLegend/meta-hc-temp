import sqlite3
import shutil
import os
import json
from typing import Tuple, Dict, Any

from models import RescueAction, RescueObservation, RescueState
from graders import grade_easy_task, grade_medium_task, grade_hard_task

class DatabaseRescueEnv:
    def __init__(self):
        self.task_name = None
        self.steps_taken = 0
        self.working_db = "working.db"
        self.template_dir = "templates"

    def _get_schema(self, cursor: sqlite3.Cursor) -> str:
        """Retrieves the current database schema as a string."""
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return "\n".join([t[0] for t in tables if t[0] is not None])

    def reset(self, task_name: str) -> RescueObservation:
        """Resets the environment by copying the task's template DB."""
        self.task_name = task_name
        self.steps_taken = 0
        
        # Copy the messy template DB to our working path
        template_path = os.path.join(self.template_dir, f"{task_name}.db")
        if not os.path.exists(template_path):
            raise ValueError(f"Template DB for task '{task_name}' not found.")
            
        shutil.copyfile(template_path, self.working_db)
        
        # Return initial observation
        with sqlite3.connect(self.working_db) as conn:
            schema = self._get_schema(conn.cursor())
            
        return RescueObservation(
            schema_info=schema,
            query_result=None,
            rows_affected=0,
            error=None
        )

    def step(self, action: RescueAction) -> Tuple[RescueObservation, float, bool, Dict[str, Any]]:
        self.steps_taken += 1
        reward = 0.0
        done = False
        info = {}

        # If the agent submits, trigger the grader and end the episode
        if action.submit:
            reward = self._grade_task()
            done = True
            with sqlite3.connect(self.working_db) as conn:
                schema = self._get_schema(conn.cursor())
            return RescueObservation(schema_info=schema), reward, done, info

        # Otherwise, execute the query
        obs = RescueObservation(schema_info="", rows_affected=0)
        
        try:
            with sqlite3.connect(self.working_db) as conn:
                conn.row_factory = sqlite3.Row 
                cursor = conn.cursor()
                
                cursor.execute(action.query)
                conn.commit()
                
                obs.schema_info = self._get_schema(cursor)
                obs.rows_affected = cursor.rowcount
                
                # If it was a SELECT query, fetch results
                if action.query.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchmany(50) # Limit to avoid context bloat
                    obs.query_result = [dict(row) for row in rows]
                    
        except sqlite3.Error as e:
            obs.error = str(e)
            
        return obs, reward, done, info

    def state(self) -> RescueState:
        return RescueState(
            task_name=self.task_name,
            steps_taken=self.steps_taken,
            db_path=self.working_db
        )
        
    def _grade_task(self) -> float:
        """Routes the current database to the correct scoring logic."""
        if self.task_name == "easy_data_cleaning":
            return grade_easy_task(self.working_db)
        elif self.task_name == "medium_schema_normalization":
            return grade_medium_task(self.working_db)
        elif self.task_name == "hard_complex_reconciliation":
            return grade_hard_task(self.working_db)
        return 0.0


# ===========================================================================
# NEW: Database helper functions for FinAudit environment
# ===========================================================================

def connect_db(db_path: str = "working.db") -> sqlite3.Connection:
    """Open a connection to the given SQLite database.

    Returns a Connection with row_factory set to sqlite3.Row so that rows
    can be accessed like dicts.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all_transactions(conn: sqlite3.Connection):
    """Return every row from the 'transactions' table as a list of dicts."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM transactions")
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error:
        return []


def fetch_limited_transactions(conn: sqlite3.Connection, limit: int = 5):
    """Return up to *limit* rows from the 'transactions' table as dicts."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM transactions LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error:
        return []


# ===========================================================================
# NEW: FinAuditEnv — structured OpenEnv-style environment for financial
#      auditing tasks.  Sits alongside the original DatabaseRescueEnv.
# ===========================================================================

class FinAuditEnv:
    """A structured RL-style environment for financial auditing.

    Actions are dicts with an 'action_type' key routed to modular handlers.
    Observations are dicts capped at 5 transaction rows.

    Reward architecture:
      - Step rewards = behavioral shaping (progress, penalties)
      - Final reward = correctness evaluation (computed only at submit_answer)
    """

    # Supported action types mapped to their handler methods
    VALID_ACTIONS = [
        "inspect_data",
        "filter_transactions",
        "calculate_total",
        "assign_category",
        "flag_anomaly",
        "submit_answer",
    ]

    TASK_DESCRIPTION = (
        "You are a financial auditing agent. Inspect the transactions table, "
        "filter and categorise records, flag anomalies, and submit your findings."
    )

    # Anti-hacking: maximum steps before forced episode end
    MAX_STEPS = 50
    # Maximum consecutive invalid actions before early termination
    MAX_INVALID_STREAK = 5

    def __init__(self, db_path: str = "working.db", template_dir: str = "templates"):
        self.db_path = db_path
        self.template_dir = template_dir

        # Episode state (all reset in reset())
        self.task_name: str = ""
        self.steps_taken: int = 0
        self.flagged: list = []              # transaction IDs flagged as anomalies
        self.categories: dict = {}           # {transaction_id: category_label}
        self.last_action_result = None
        self.done: bool = False

        # Reward tracking
        self.cumulative_step_reward: float = 0.0  # accumulated step-level signals
        self.reward_breakdown: dict = {}          # final reward details

        # Action tracking (for dependency enforcement & repeat detection)
        self.action_history: list = []            # full history of action keys
        self.completed_action_types: set = set()  # unique action_types performed
        self.last_action_type: str = ""           # for consecutive repeat detection
        self.consecutive_repeat_count: int = 0    # consecutive same-action counter
        self.invalid_action_count: int = 0        # escalating invalid penalty
        self.invalid_streak: int = 0              # consecutive invalid for early term
        self.submitted_total: float = None        # agent's reconciliation answer

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "finaudit_complex") -> dict:
        """Reset the environment: reload DB, clear state, return observation.

        Args:
            task_name: which template DB to copy into the working path.

        Returns:
            Structured observation dict.
        """
        self.task_name = task_name
        self.steps_taken = 0
        self.flagged = []
        self.categories = {}
        self.last_action_result = None
        self.done = False

        # Reset reward tracking
        self.cumulative_step_reward = 0.0
        self.reward_breakdown = {}

        # Reset action tracking
        self.action_history = []
        self.completed_action_types = set()
        self.last_action_type = ""
        self.consecutive_repeat_count = 0
        self.invalid_action_count = 0
        self.invalid_streak = 0
        self.submitted_total = None

        # Copy the template DB to the working path
        template_path = os.path.join(self.template_dir, f"{task_name}.db")
        if not os.path.exists(template_path):
            raise ValueError(f"Template DB for task '{task_name}' not found.")
        shutil.copyfile(template_path, self.db_path)

        return self._build_observation()

    def step(self, action: dict):
        """Execute one action and return (observation, reward, done).

        Reward Logic:
          - Per-step: behavioral shaping signals (progress, penalties)
          - At submit: full correctness evaluation via verifiers
          - Penalties applied immediately

        Args:
            action: dict with 'action_type' (str) and optional 'params' (dict).

        Returns:
            (observation_dict, reward_float, done_bool)
        """
        from rewards import (
            progress_reward, categorization_step_reward, flag_step_reward,
            efficiency_penalty, invalid_action_penalty,
            consecutive_repeat_penalty, no_state_change_penalty,
            check_action_dependencies, compute_final_reward,
        )

        self.steps_taken += 1
        step_reward = 0.0

        # --- Per-step efficiency penalty (always applied) ---
        step_reward += efficiency_penalty()

        # --- Anti-hacking: enforce max step limit ---
        if self.steps_taken >= self.MAX_STEPS:
            self.done = True
            self.last_action_result = {
                "error": f"Max steps ({self.MAX_STEPS}) reached. Episode ended."
            }
            self.cumulative_step_reward += step_reward
            # Compute final reward even on timeout
            self.reward_breakdown = compute_final_reward(
                self.db_path, self.categories, self.flagged,
                self.submitted_total, self.completed_action_types,
                self.cumulative_step_reward,
            )
            final_reward = self.reward_breakdown["final_reward"]
            return self._build_observation(), final_reward, self.done

        action_type = action.get("action_type", "")
        params = action.get("params", {})

        # --- Track action history ---
        action_key = (action_type, json.dumps(params, sort_keys=True, default=str))
        self.action_history.append(action_key)

        # --- Consecutive repeat detection ---
        if action_type == self.last_action_type:
            self.consecutive_repeat_count += 1
        else:
            self.consecutive_repeat_count = 1
        self.last_action_type = action_type

        # Check for consecutive repeat penalty or block
        repeat_pen = consecutive_repeat_penalty(self.consecutive_repeat_count)
        if repeat_pen is None:
            # BLOCKED: >3 consecutive repeats → reject action, apply penalty
            step_reward += -0.1
            self.cumulative_step_reward += step_reward
            self.last_action_result = {
                "error": f"Action '{action_type}' blocked: repeated "
                         f"{self.consecutive_repeat_count} times consecutively. "
                         f"Try a different action."
            }
            return self._build_observation(), step_reward, self.done
        else:
            step_reward += repeat_pen

        # --- Dispatch to handler ---
        handler_map = {
            "inspect_data":        self._handle_inspect_data,
            "filter_transactions": self._handle_filter_transactions,
            "calculate_total":     self._handle_calculate_total,
            "assign_category":     self._handle_assign_category,
            "flag_anomaly":        self._handle_flag_anomaly,
            "submit_answer":       self._handle_submit_answer,
        }

        handler = handler_map.get(action_type)
        if handler is None:
            # --- Invalid action: escalating penalty ---
            self.invalid_action_count += 1
            self.invalid_streak += 1
            step_reward += invalid_action_penalty(self.invalid_action_count)
            self.cumulative_step_reward += step_reward
            self.last_action_result = {
                "error": f"Unknown action_type '{action_type}'. "
                         f"Valid actions: {self.VALID_ACTIONS}"
            }
            # Early termination if too many consecutive invalid actions
            if self.invalid_streak >= self.MAX_INVALID_STREAK:
                self.done = True
                self.last_action_result["error"] += (
                    f" Episode terminated: {self.MAX_INVALID_STREAK} "
                    f"consecutive invalid actions."
                )
            return self._build_observation(), step_reward, self.done

        # Reset invalid streak on valid action
        self.invalid_streak = 0

        # --- Submit: enforce action dependencies BEFORE executing ---
        if action_type == "submit_answer":
            deps = check_action_dependencies(self.completed_action_types)
            if not deps["satisfied"]:
                # REJECT submit — don't end episode, apply cheating penalty
                step_reward += -0.5
                self.cumulative_step_reward += step_reward
                self.last_action_result = {
                    "error": f"Cannot submit yet. Missing required actions: "
                             f"{deps['missing']}. Complete them first."
                }
                return self._build_observation(), step_reward, self.done

        # --- Capture state BEFORE action (for state-change validation) ---
        prev_categories = dict(self.categories)
        prev_flagged = list(self.flagged)

        # --- Execute handler ---
        try:
            self.last_action_result = handler(params)
        except Exception as exc:
            self.last_action_result = {"error": str(exc)}
            self.invalid_action_count += 1
            step_reward += invalid_action_penalty(self.invalid_action_count)
            self.cumulative_step_reward += step_reward
            return self._build_observation(), step_reward, self.done

        # --- Track completed action types ---
        self.completed_action_types.add(action_type)

        # --- Step-level reward signals (progress-based, NOT flat) ---

        # 1. Progress reward: first-time meaningful action
        step_reward += progress_reward(action_type, self.completed_action_types - {action_type})

        # 2. Categorization step reward: correct label = +0.01, wrong = -0.005
        if action_type == "assign_category":
            tx_id = params.get("transaction_id")
            category = params.get("category")
            if tx_id is not None and category:
                step_reward += categorization_step_reward(self.db_path, tx_id, category)
            # State-change validation: did category actually change?
            if self.categories == prev_categories:
                step_reward += no_state_change_penalty()

        # 3. Flag step reward: correct flag = +0.01, wrong = -0.005
        if action_type == "flag_anomaly":
            tx_id = params.get("transaction_id")
            if tx_id is not None:
                # Only reward if this is a NEW flag (not already flagged)
                if tx_id not in prev_flagged:
                    step_reward += flag_step_reward(self.db_path, tx_id)
                else:
                    # Re-flagging same tx → no state change
                    step_reward += no_state_change_penalty()

        # --- Final reward on submit ---
        if self.done:
            self.cumulative_step_reward += step_reward
            self.reward_breakdown = compute_final_reward(
                self.db_path, self.categories, self.flagged,
                self.submitted_total, self.completed_action_types,
                self.cumulative_step_reward,
            )
            final_reward = self.reward_breakdown["final_reward"]
            return self._build_observation(), final_reward, self.done

        self.cumulative_step_reward += step_reward
        return self._build_observation(), step_reward, self.done

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict:
        """Construct the observation dict (max 5 transaction rows)."""
        conn = connect_db(self.db_path)
        try:
            transactions = fetch_limited_transactions(conn, limit=5)
        finally:
            conn.close()

        obs = {
            "task_description": self.TASK_DESCRIPTION,
            "last_action_result": self.last_action_result,
            "transactions": transactions,
            "done": self.done,
            "steps_remaining": max(0, self.MAX_STEPS - self.steps_taken),
            "error": None,
        }

        # Include reward breakdown on final observation
        if self.done and self.reward_breakdown:
            obs["reward_breakdown"] = self.reward_breakdown

        return obs

    # ------------------------------------------------------------------
    # Action handlers — each is a separate, modular function
    # ------------------------------------------------------------------

    def _handle_inspect_data(self, params: dict) -> dict:
        """Return column names and the first 5 rows of the transactions table.

        Anti-hacking: ground_truth tables are hidden from the agent.
        """
        conn = connect_db(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions LIMIT 5")
            rows = [dict(r) for r in cursor.fetchall()]
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Show only agent-visible tables (hide ground_truth_*)
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'ground_truth%' "
                "AND name NOT LIKE 'sqlite_%'"
            )
            visible_tables = [r[0] for r in cursor.fetchall()]
        finally:
            conn.close()

        return {
            "columns": columns,
            "sample_rows": rows,
            "row_count": len(rows),
            "tables": visible_tables,
        }

    def _handle_filter_transactions(self, params: dict) -> dict:
        """Filter transactions by a given column and value.

        Required params: column (str), value (str/number).
        """
        column = params.get("column")
        value = params.get("value")
        if not column:
            return {"error": "Missing required param 'column'."}
        if value is None:
            return {"error": "Missing required param 'value'."}

        conn = connect_db(self.db_path)
        try:
            cursor = conn.cursor()
            # Use parameterised query for safety
            query = f"SELECT * FROM transactions WHERE {column} = ? LIMIT 5"
            cursor.execute(query, (value,))
            rows = [dict(r) for r in cursor.fetchall()]
        except sqlite3.Error as exc:
            return {"error": str(exc)}
        finally:
            conn.close()

        return {"filtered_rows": rows, "match_count": len(rows)}

    def _handle_calculate_total(self, params: dict) -> dict:
        """Calculate the SUM of the 'amount' column, optionally filtered.

        Optional params: column (str), value (str/number) — to filter before summing.
        """
        conn = connect_db(self.db_path)
        try:
            cursor = conn.cursor()
            column = params.get("column")
            value = params.get("value")
            if column and value is not None:
                query = f"SELECT SUM(amount) AS total FROM transactions WHERE {column} = ?"
                cursor.execute(query, (value,))
            else:
                cursor.execute("SELECT SUM(amount) AS total FROM transactions")
            result = cursor.fetchone()
            total = dict(result)["total"] if result else 0
        except sqlite3.Error as exc:
            return {"error": str(exc)}
        finally:
            conn.close()

        return {"total": total}

    def _handle_assign_category(self, params: dict) -> dict:
        """Tag a transaction with a category label (in-memory tracking).

        Required params: transaction_id (int), category (str).
        """
        txn_id = params.get("transaction_id")
        category = params.get("category")
        if txn_id is None:
            return {"error": "Missing required param 'transaction_id'."}
        if not category:
            return {"error": "Missing required param 'category'."}

        self.categories[txn_id] = category
        return {
            "assigned": {txn_id: category},
            "all_categories": dict(self.categories),
        }

    def _handle_flag_anomaly(self, params: dict) -> dict:
        """Flag a transaction ID as anomalous (in-memory tracking).

        Required params: transaction_id (int).
        """
        txn_id = params.get("transaction_id")
        if txn_id is None:
            return {"error": "Missing required param 'transaction_id'."}

        if txn_id not in self.flagged:
            self.flagged.append(txn_id)
        return {"flagged_id": txn_id, "all_flagged": list(self.flagged)}

    def _handle_submit_answer(self, params: dict) -> dict:
        """Mark the episode as done and return a summary of findings.

        Expected params for full format reward:
            total (float)      — agent's computed total spend
            flagged (list)     — tx_ids flagged as anomalies
            categories (dict)  — {tx_id: category} assignments
            summary (str)      — free-form agent summary (optional)
        """
        self.done = True

        # Extract structured answer for verifiers
        if "total" in params:
            self.submitted_total = float(params["total"])

        # Allow agent to pass additional flags/categories in submit
        if "flagged" in params:
            for tx_id in params["flagged"]:
                if tx_id not in self.flagged:
                    self.flagged.append(tx_id)

        if "categories" in params:
            self.categories.update(params["categories"])

        return {
            "status": "submitted",
            "steps_taken": self.steps_taken,
            "flagged_transactions": list(self.flagged),
            "assigned_categories": dict(self.categories),
            "submitted_total": self.submitted_total,
            "agent_summary": params.get("summary", ""),
        }