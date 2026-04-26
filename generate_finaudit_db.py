import sqlite3
import random
import datetime
import os

DB_CONFIGS = {
    "finaudit_easy": {"rows": 100, "anomaly_mode": "fixed", "anom_count": 1, "min_anom": 50000.0, "max_anom": 99999.0, "smurf_traps": 1},
    "finaudit_medium": {"rows": 500, "anomaly_mode": "fixed", "anom_count": 5, "min_anom": 10000.0, "max_anom": 49999.0, "smurf_traps": 2},
    "finaudit_complex": {"rows": 1000, "anomaly_mode": "rate", "anom_rate": 0.03, "min_anom": 10000.0, "max_anom": 999999.99, "smurf_traps": 3}
}

ACCOUNTS = list(range(100, 150))
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

    # CHANGED: 'date' TEXT is now 'timestamp' TEXT
    c.execute("CREATE TABLE transactions (tx_id INTEGER PRIMARY KEY AUTOINCREMENT, account_id INTEGER NOT NULL, amount REAL NOT NULL, tx_type TEXT NOT NULL, timestamp TEXT NOT NULL, description TEXT)")
    c.execute("CREATE TABLE ground_truth_categories (tx_id INTEGER PRIMARY KEY, category TEXT NOT NULL)")
    
    # CHANGED: Added anomaly_type so your grading scripts know *why* it's an anomaly
    c.execute("CREATE TABLE ground_truth_anomalies (tx_id INTEGER PRIMARY KEY, is_anomaly INTEGER NOT NULL, anomaly_type TEXT NOT NULL)")

    base_date = datetime.datetime(2023, 1, 1, 8, 0, 0)
    txn_data, cat_data, anom_data = [], [], []
    
    anom_indices = set()
    if config["anomaly_mode"] == "fixed":
        anom_indices = set(random.sample(range(config["rows"]), config["anom_count"]))

    current_tx_id = 1

    # --- 1. GENERATE NORMAL NOISE AND STANDARD ANOMALIES ---
    for i in range(config["rows"]):
        acct = random.choice(ACCOUNTS)
        tx_type = random.choice(["credit", "debit"])
        
        # Exact timestamps instead of just dates
        random_seconds = random.randint(0, 365 * 24 * 60 * 60)
        tx_timestamp = (base_date + datetime.timedelta(seconds=random_seconds)).isoformat()
        
        is_anom = (i in anom_indices) if config["anomaly_mode"] == "fixed" else (random.random() < config["anom_rate"])

        if is_anom:
            amount = round(random.uniform(config["min_anom"], config["max_anom"]), 2)
        else:
            # Cap normal transactions at 4900 so they don't accidentally become > 5000 anomalies
            amount = round(random.uniform(10.0, 4900.0), 2)

        txn_data.append((acct, amount, tx_type, tx_timestamp, f"TXN-{current_tx_id}"))
        cat_data.append((current_tx_id, classify_category(amount)))
        anom_data.append((current_tx_id, 1 if is_anomaly(amount) else 0, "standard_high_value" if is_anomaly(amount) else "none"))
        
        current_tx_id += 1

    # --- 2. INJECT THE "TEMPORAL SMURFING" TRAPS ---
    for trap_idx in range(config.get("smurf_traps", 0)):
        # Give smurfs unique 900-series account IDs so you can verify the agent found the right ones
        smurf_acct = 999 - trap_idx 
        
        smurf_start_time = base_date + datetime.timedelta(days=random.randint(0, 360))

        # 6 transactions of ~$950 spaced 30 mins apart = ~$5,700 total.
        for s in range(6):
            smurf_amount = round(random.uniform(900.0, 990.0), 2)
            smurf_timestamp = (smurf_start_time + datetime.timedelta(minutes=s * 30)).isoformat()
            
            txn_data.append((smurf_acct, smurf_amount, "credit", smurf_timestamp, f"SMURF-TRAP-{trap_idx}-{s}"))
            
            # The individual category is still "medium"
            cat_data.append((current_tx_id, classify_category(smurf_amount))) 
            
            # CRITICAL: Mark as anomaly in Ground Truth even though amount < 5000
            anom_data.append((current_tx_id, 1, "smurfing_pattern")) 
            current_tx_id += 1

    c.executemany("INSERT INTO transactions (account_id, amount, tx_type, timestamp, description) VALUES (?, ?, ?, ?, ?)", txn_data)
    c.executemany("INSERT INTO ground_truth_categories (tx_id, category) VALUES (?, ?)", cat_data)
    c.executemany("INSERT INTO ground_truth_anomalies (tx_id, is_anomaly, anomaly_type) VALUES (?, ?, ?)", anom_data)
    
    conn.commit()
    print(f"Generated {db_path} | Rows: {config['rows']} | Smurf Traps: {config.get('smurf_traps', 0)}")
    conn.close()

def generate_all():
    os.makedirs("templates", exist_ok=True)
    random.seed(RANDOM_SEED)
    for task_name, config in DB_CONFIGS.items():
        generate_db(task_name, config)

if __name__ == "__main__":
    generate_all()