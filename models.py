from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class RescueAction(BaseModel):
    query: str = Field(description="The SQL query to execute against the SQLite database.")
    submit: bool = Field(default=False, description="Set to True ONLY when you have completed the task and are ready for grading.")

class RescueObservation(BaseModel):
    schema_info: str = Field(description="The current schema of the database.")
    query_result: Optional[List[Dict[str, Any]]] = Field(default=None, description="Results from a SELECT query, limited to 50 rows.")
    rows_affected: int = Field(default=0, description="Number of rows modified by INSERT/UPDATE/DELETE.")
    error: Optional[str] = Field(default=None, description="SQL execution error message, if any.")

class RescueState(BaseModel):
    task_name: str
    steps_taken: int
    db_path: str


# ---------------------------------------------------------------------------
# NEW: FinAudit models for the structured OpenEnv-style environment
# ---------------------------------------------------------------------------

class FinAuditAction(BaseModel):
    """Structured action for the FinAudit environment.

    action_type must be one of:
        inspect_data, filter_transactions, calculate_total,
        assign_category, flag_anomaly, submit_answer
    params is a free-form dict whose keys depend on the action_type.
    """
    action_type: str = Field(
        description="The type of action to perform (e.g. 'inspect_data', 'filter_transactions')."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action. Keys vary by action_type."
    )


class FinAuditObservation(BaseModel):
    """Structured observation returned by the FinAudit environment."""
    task_description: str = Field(
        default="",
        description="High-level description of the current auditing task."
    )
    last_action_result: Optional[Any] = Field(
        default=None,
        description="Result produced by the most recent action."
    )
    transactions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="A small subset of transactions (max 5 rows)."
    )
    done: bool = Field(
        default=False,
        description="Whether the episode is finished."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message, if the last action failed."
    )