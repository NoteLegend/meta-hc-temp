"""
Generate the FinAudit SQLite template databases.

Run with:
    python generate_finaudit_db.py

Creates:
    templates/finaudit_easy.db     - 100 rows, 1 fixed anomaly over $50,000
    templates/finaudit_medium.db   - 500 rows, 5 fixed anomalies over $50,000
    templates/finaudit_complex.db  - 1000 rows, roughly 3% random anomalies

Each database contains:
    transactions                 - visible to the agent
    ground_truth_categories      - hidden grading table
    ground_truth_anomalies       - hidden grading table
"""

import datetime
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional


TEMPLATE_DIR = "templates"
ACCOUNTS = list(range(100, 106))
TX_TYPES = ["credit", "debit"]
BASE_DATE = datetime.date(2023, 1, 1)
RANDOM_SEED = 42


@dataclass(frozen=True)
class TaskSpec:
    name: str
    rows: int
    fixed_anomalies: Optional[int] = None
    random_anomaly_rate: Optional[float] = None


TASKS = [
    TaskSpec(name="finaudit_easy", rows=100, fixed_anomalies=1),
    TaskSpec(name="finaudit_medium", rows=500, fixed_anomalies=5),
    TaskSpec(name="finaudit_complex", rows=1000, random_anomaly_rate=0.03),
]


def classify_category(amount: float) -> str:
    """Assign a category label based on amount thresholds."""
    if amount <= 100:
        return "small"
    if amount <= 1000:
        return "medium"
    if amount <= 5000:
        return "large"
    return "anomaly"


def is_anomaly(amount: float) -> bool:
    """Flag as anomalous if amount exceeds $5,000."""
    return amount > 5000


def create_schema(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        CREATE TABLE transactions (
            tx_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id  INTEGER NOT NULL,
            amount      REAL    NOT NULL,
            tx_type     TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            description TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE ground_truth_categories (
            tx_id    INTEGER PRIMARY KEY,
            category TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE ground_truth_anomalies (
            tx_id      INTEGER PRIMARY KEY,
            is_anomaly INTEGER NOT NULL
        )
    """)


def fixed_anomaly_ids(row_count: int, anomaly_count: int) -> set[int]:
    """Return deterministic anomaly IDs spread across the dataset."""
    if anomaly_count <= 0:
        return set()
    stride = row_count // (anomaly_count + 1)
    return {stride * (idx + 1) for idx in range(anomaly_count)}


def generate_amount(rng: random.Random, tx_id: int, anomaly_ids: set[int], random_rate: Optional[float]) -> float:
    """Generate a transaction amount following the task anomaly policy."""
    if tx_id in anomaly_ids:
        return round(rng.uniform(50_000.01, 250_000.0), 2)

    if random_rate is not None and rng.random() < random_rate:
        return round(rng.uniform(5_000.01, 250_000.0), 2)

    return round(rng.uniform(10.0, 5000.0), 2)


def build_rows(spec: TaskSpec) -> tuple[list[tuple], list[tuple], list[tuple]]:
    rng = random.Random(f"{RANDOM_SEED}-{spec.name}")
    anomaly_ids = fixed_anomaly_ids(spec.rows, spec.fixed_anomalies or 0)

    transactions = []
    categories = []
    anomalies = []

    for tx_id in range(1, spec.rows + 1):
        account_id = rng.choice(ACCOUNTS)
        amount = generate_amount(rng, tx_id, anomaly_ids, spec.random_anomaly_rate)
        tx_type = rng.choice(TX_TYPES)
        date = (BASE_DATE + datetime.timedelta(days=rng.randint(0, 365))).isoformat()
        description = f"{spec.name.upper()}-TXN-{tx_id:04d}"

        transactions.append((account_id, amount, tx_type, date, description))
        categories.append((tx_id, classify_category(amount)))
        anomalies.append((tx_id, 1 if is_anomaly(amount) else 0))

    return transactions, categories, anomalies


def write_database(spec: TaskSpec) -> None:
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    db_path = os.path.join(TEMPLATE_DIR, f"{spec.name}.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_schema(cursor)
    transactions, categories, anomalies = build_rows(spec)

    cursor.executemany(
        "INSERT INTO transactions (account_id, amount, tx_type, date, description) "
        "VALUES (?, ?, ?, ?, ?)",
        transactions,
    )
    cursor.executemany(
        "INSERT INTO ground_truth_categories (tx_id, category) VALUES (?, ?)",
        categories,
    )
    cursor.executemany(
        "INSERT INTO ground_truth_anomalies (tx_id, is_anomaly) VALUES (?, ?)",
        anomalies,
    )
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_rows = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM ground_truth_anomalies WHERE is_anomaly = 1")
    anomaly_count = cursor.fetchone()[0]
    cursor.execute("SELECT category, COUNT(*) FROM ground_truth_categories GROUP BY category")
    category_counts = dict(cursor.fetchall())
    cursor.execute("SELECT SUM(amount) FROM transactions")
    grand_total = cursor.fetchone()[0]

    conn.close()

    print(f"Generated {db_path}")
    print(f"  Transactions : {total_rows}")
    print(f"  Anomalies    : {anomaly_count}")
    print(f"  Categories   : {category_counts}")
    print(f"  Grand total  : ${grand_total:,.2f}")


def generate(specs: Iterable[TaskSpec] = TASKS) -> None:
    for spec in specs:
        write_database(spec)


if __name__ == "__main__":
    generate()
