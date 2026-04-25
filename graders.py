import sqlite3

def grade_easy_task(db_path: str) -> float:
    """Agent must trim whitespace from names and standardize dates to YYYY-MM-DD."""
    actual_score = 0.0
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            
            # Get total number of rows to calculate percentages
            c.execute("SELECT COUNT(*) FROM customers")
            total_rows = c.fetchone()[0]
            
            if total_rows > 0:
                # Count how many names are perfectly trimmed
                c.execute("SELECT COUNT(*) FROM customers WHERE name = TRIM(name)")
                trimmed_names = c.fetchone()[0]
                
                # Count how many dates follow the strict YYYY-MM-DD format 
                # (length 10, hyphen at pos 5 and 8)
                c.execute("""
                    SELECT COUNT(*) FROM customers 
                    WHERE length(signup_date) = 10 
                    AND substr(signup_date, 5, 1) = '-' 
                    AND substr(signup_date, 8, 1) = '-'
                """)
                formatted_dates = c.fetchone()[0]
                
                # Calculate partial credit: 50% for names, 50% for dates
                name_score = trimmed_names / total_rows
                date_score = formatted_dates / total_rows
                actual_score = (name_score + date_score) / 2.0
    except sqlite3.Error:
        actual_score = 0.0

    # THE VALIDATOR CLAMP
    if actual_score >= 1.0: return 0.99
    if actual_score <= 0.0: return 0.01
    return actual_score


def grade_medium_task(db_path: str) -> float:
    """Agent must create 'customers' and 'orders' tables with proper references."""
    score = 0.0
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            
            # 1. Check if target tables exist (0.4 points)
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('customers', 'orders')")
            tables = [row[0] for row in c.fetchall()]
            if 'customers' in tables and 'orders' in tables:
                score += 0.4
                
                # 2. Check for deduplication: No duplicate names in customers (0.3 points)
                c.execute("SELECT COUNT(*) FROM (SELECT name FROM customers GROUP BY name HAVING COUNT(*) > 1)")
                duplicate_groups = c.fetchone()[0]
                if duplicate_groups == 0:
                    score += 0.3
                    
                # 3. Check Referential Integrity: No orphaned orders (0.3 points)
                # Assumes 'orders' has a 'customer_id' column mapping to customers(id)
                # If your orders table uses 'customer_name', change 'customer_id' to 'customer_name' and 'id' to 'name'
                try:
                    c.execute("""
                        SELECT COUNT(*) FROM orders 
                        WHERE customer_id NOT IN (SELECT id FROM customers)
                    """)
                    orphaned_orders = c.fetchone()[0]
                    if orphaned_orders == 0:
                        score += 0.3
                except sqlite3.Error:
                    # Column might not exist or agent failed to create it properly
                    pass
    except sqlite3.Error:
        pass
        
    # THE VALIDATOR CLAMP
    if score >= 1.0: return 0.99
    if score <= 0.0: return 0.01
    return score


def grade_hard_task(db_path: str) -> float:
    """Agent must create a view 'account_balances' calculating net balance (credit - debit)."""
    actual_score = 0.0
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            
            # THE GOLDEN QUERY: We dynamically calculate the true answer
            # Assumes the raw data is in a table called 'transactions'
            golden_query = """
                SELECT 
                    account_id, 
                    SUM(CASE WHEN type = 'credit' THEN amount ELSE -amount END) as true_balance
                FROM transactions
                GROUP BY account_id
                ORDER BY account_id
            """
            c.execute(golden_query)
            golden_rows = c.fetchall()
            
            # Fetch the agent's view
            c.execute("SELECT account_id, net_balance FROM account_balances ORDER BY account_id")
            agent_rows = c.fetchall()
            
            # Compare the dynamic result to the agent's view
            if len(golden_rows) > 0 and agent_rows == golden_rows:
                actual_score = 1.0
            elif len(agent_rows) > 0:
                # Partial credit calculation: how many rows matched exactly?
                matches = len(set(agent_rows) & set(golden_rows))
                actual_score = matches / len(golden_rows)

    except sqlite3.Error:
        actual_score = 0.0

    # THE VALIDATOR CLAMP
    if actual_score >= 1.0: return 0.99
    if actual_score <= 0.0: return 0.01
    return actual_score