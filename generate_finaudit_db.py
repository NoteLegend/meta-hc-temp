"""
Generate the finaudit_complex.db template with ground truth tables.

Run with:
    python generate_finaudit_db.py

Creates templates/finaudit_complex.db with:
  - transactions        (1000 rows — visible to agent)
  - ground_truth_categories  (1000 rows — hidden from agent)
  - ground_truth_anomalies   (1000 rows — hidden from agent)
"""

import sqlite3
import random
import datetime
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.path.join("templates", "finaudit_complex.db")
NUM_TRANSACTIONS = 1000
ACCOUNTS = list(range(100, 106))  # 6 accounts: 100-105
RANDOM_SEED = 42                  # deterministic for reproducibility


# ---------------------------------------------------------------------------
# Category / anomaly rules  (deterministic, based on amount)
# ---------------------------------------------------------------------------

def classify_category(amount: float) -> str:
    """Assign a category label based on amount thresholds."""
    if amount <= 100:
        return "small"
    elif amount <= 1000:
        return "medium"
    elif amount <= 5000:
        return "large"
    else:
        return "anomaly"


def is_anomaly(amount: float) -> bool:
    """Flag as anomalous if amount exceeds $5,000."""
    return amount > 5000


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate():
    os.makedirs("templates", exist_ok=True)

    # Remove old DB if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    random.seed(RANDOM_SEED)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # --- 1. Transactions table (agent-visible) ---
    c.execute("""
        CREATE TABLE transactions (
            tx_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id  INTEGER NOT NULL,
            amount      REAL    NOT NULL,
            tx_type     TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            description TEXT
        )
    """)

    # --- 2. Ground truth tables (hidden from agent) ---
    c.execute("""
        CREATE TABLE ground_truth_categories (
            tx_id    INTEGER PRIMARY KEY,
            category TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE ground_truth_anomalies (
            tx_id      INTEGER PRIMARY KEY,
            is_anomaly INTEGER NOT NULL   -- 1 = anomalous, 0 = normal
        )
    """)

    # --- 3. Generate data ---
    base_date = datetime.date(2023, 1, 1)
    tx_types = ["credit", "debit"]

    txn_data = []
    cat_data = []
    anom_data = []

    for i in range(NUM_TRANSACTIONS):
        acct = random.choice(ACCOUNTS)
        tx_type = random.choice(tx_types)
        date = (base_date + datetime.timedelta(days=random.randint(0, 365))).isoformat()
        desc = f"TXN-{i}"

        # Most transactions are normal (10–5000).
        # ~3 % are planted anomalies (> $5,000).
        if random.random() < 0.03:
            amount = round(random.uniform(10_000, 999_999.99), 2)
        else:
            amount = round(random.uniform(10.0, 5000.0), 2)

        tx_id = i + 1  # AUTOINCREMENT starts at 1
        txn_data.append((acct, amount, tx_type, date, desc))
        cat_data.append((tx_id, classify_category(amount)))
        anom_data.append((tx_id, 1 if is_anomaly(amount) else 0))

    c.executemany(
        "INSERT INTO transactions (account_id, amount, tx_type, date, description) "
        "VALUES (?, ?, ?, ?, ?)",
        txn_data,
    )
    c.executemany(
        "INSERT INTO ground_truth_categories (tx_id, category) VALUES (?, ?)",
        cat_data,
    )
    c.executemany(
        "INSERT INTO ground_truth_anomalies (tx_id, is_anomaly) VALUES (?, ?)",
        anom_data,
    )

    conn.commit()

    # --- 4. Print summary ---
    c.execute("SELECT COUNT(*) FROM transactions")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM ground_truth_anomalies WHERE is_anomaly = 1")
    anomalies = c.fetchone()[0]
    c.execute("SELECT category, COUNT(*) FROM ground_truth_categories GROUP BY category")
    cats = dict(c.fetchall())
    c.execute("SELECT SUM(amount) FROM transactions")
    grand_total = c.fetchone()[0]

    print(f"Generated {DB_PATH}")
    print(f"  Transactions : {total}")
    print(f"  Anomalies    : {anomalies}")
    print(f"  Categories   : {cats}")
    print(f"  Grand total  : ${grand_total:,.2f}")

    conn.close()


if __name__ == "__main__":
    generate()
