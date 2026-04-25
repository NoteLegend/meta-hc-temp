import sqlite3
import random
import datetime
import os

DB_CONFIGS = {
    "finaudit_easy": {"rows": 100, "anomaly_mode": "fixed", "anom_count": 1, "min_anom": 50000.0, "max_anom": 99999.0},
    "finaudit_medium": {"rows": 500, "anomaly_mode": "fixed", "anom_count": 5, "min_anom": 10000.0, "max_anom": 49999.0},
    "finaudit_complex": {"rows": 1000, "anomaly_mode": "rate", "anom_rate": 0.03, "min_anom": 10000.0, "max_anom": 999999.99}
}

ACCOUNTS = list(range(100, 106))
RANDOM_SEED = 42

def classify_category(amount: float) -> str:
    if amount <= 100: return "small"
    elif amount <= 1000: return "medium"
    elif amount <= 5000: return "large"
    else: return "anomaly"

def is_anomaly(amount: float) -> bool:
    return amount > 5000

def generate_db(task_name: str, config: dict):
    db_path = os.path.join("templates", f"{task_name}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("CREATE TABLE transactions (tx_id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER NOT NULL, amount REAL NOT NULL, tx_type TEXT NOT NULL, date TEXT NOT NULL, description TEXT)")
    c.execute("CREATE TABLE ground_truth_categories (tx_id INTEGER PRIMARY KEY, category TEXT NOT NULL)")
    c.execute("CREATE TABLE ground_truth_anomalies (tx_id INTEGER PRIMARY KEY, is_anomaly INTEGER NOT NULL)")

    base_date = datetime.date(2023, 1, 1)
    txn_data, cat_data, anom_data = [], [], []
    
    anom_indices = set()
    if config["anomaly_mode"] == "fixed":
        anom_indices = set(random.sample(range(config["rows"]), config["anom_count"]))

    for i in range(config["rows"]):
        acct = random.choice(ACCOUNTS)
        tx_type = random.choice(["credit", "debit"])
        date = (base_date + datetime.timedelta(days=random.randint(0, 365))).isoformat()
        
        is_anom = (i in anom_indices) if config["anomaly_mode"] == "fixed" else (random.random() < config["anom_rate"])

        if is_anom:
            amount = round(random.uniform(config["min_anom"], config["max_anom"]), 2)
        else:
            amount = round(random.uniform(10.0, 5000.0), 2)

        tx_id = i + 1
        txn_data.append((acct, amount, tx_type, date, f"TXN-{i}"))
        cat_data.append((tx_id, classify_category(amount)))
        anom_data.append((tx_id, 1 if is_anomaly(amount) else 0))

    c.executemany("INSERT INTO transactions (account_id, amount, tx_type, date, description) VALUES (?, ?, ?, ?, ?)", txn_data)
    c.executemany("INSERT INTO ground_truth_categories (tx_id, category) VALUES (?, ?)", cat_data)
    c.executemany("INSERT INTO ground_truth_anomalies (tx_id, is_anomaly) VALUES (?, ?)", anom_data)
    
    conn.commit()
    print(f"Generated {db_path} | Rows: {config['rows']}")
    conn.close()

def generate_all():
    os.makedirs("templates", exist_ok=True)
    random.seed(RANDOM_SEED)
    for task_name, config in DB_CONFIGS.items():
        generate_db(task_name, config)

if __name__ == "__main__":
    generate_all()