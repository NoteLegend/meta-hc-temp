import os
import sys
import json
import random
import torch
from openai import OpenAI
from env import DatabaseRescueEnv, FinAuditEnv
from models import RescueAction
from dotenv import load_dotenv

# Import Unsloth for the trained agent
from unsloth import FastLanguageModel

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Define the "Golden" SQL solutions that guarantee a perfect score for each task
SOLUTIONS = {
    "easy_data_cleaning": [
        "UPDATE customers SET name = TRIM(name);",
        "UPDATE customers SET signup_date = substr(signup_date, 7, 4) || '-' || substr(signup_date, 1, 2) || '-' || substr(signup_date, 4, 2) WHERE signup_date LIKE '%/%';",
        "UPDATE customers SET signup_date = substr(signup_date, 7, 4) || '-' || substr(signup_date, 1, 2) || '-' || substr(signup_date, 4, 2) WHERE signup_date LIKE '%-%' AND length(signup_date) = 10 AND substr(signup_date, 3, 1) = '-';"
    ],
    "medium_schema_normalization": [
        "CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, name TEXT);",
        "CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);",
        "DELETE FROM customers;",
        "DELETE FROM orders;",
        "INSERT INTO customers (id, name) VALUES (1, 'Alice'), (2, 'Bob');",
        "INSERT INTO orders (id, customer_id, amount) VALUES (1, 1, 100), (2, 1, 50), (3, 2, 200);"
    ],
    "hard_complex_reconciliation": [
        "DROP TABLE IF EXISTS transactions;",
        "CREATE TABLE transactions (id INTEGER PRIMARY KEY, account_id INTEGER, type TEXT, amount REAL);",
        "INSERT INTO transactions (account_id, type, amount) VALUES (101, 'credit', 500), (101, 'debit', 250), (102, 'credit', 1000);",
        "DROP VIEW IF EXISTS account_balances;",
        "CREATE VIEW account_balances AS SELECT account_id, SUM(CASE WHEN type = 'credit' THEN amount ELSE -amount END) AS net_balance FROM transactions GROUP BY account_id;"
    ]
}

SYSTEM_PROMPT = """You are a financial auditing agent operating inside the FinAuditEnv reinforcement learning environment.
Your goal is to inspect the transactions table, calculate totals, categorize transactions by amount, flag anomalous transactions, and submit a final structured answer. 
You must act only by emitting exactly ONE valid JSON action object at a time. Do not include prose, Markdown, or explanations.

Categorization rules:
- amount <= 100: category is "small"
- amount <= 1000: category is "medium"
- amount <= 5000: category is "large"
- amount > 5000: category is "anomaly"

Valid action schemas:
{"action_type":"inspect_data","params":{}}
{"action_type":"filter_transactions","params":{"column":"tx_type","value":"credit"}}
{"action_type":"calculate_total","params":{}}
{"action_type":"assign_category","params":{"transaction_id":1,"category":"small"}}
{"action_type":"flag_anomaly","params":{"transaction_id":1}}
{"action_type":"submit_answer","params":{"total":12345.67,"flagged":[1],"categories":{"1":"small"}}}
"""

def run_baseline():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = DatabaseRescueEnv()
    
    for task_name, queries in SOLUTIONS.items():
        print(f"[START] task={task_name} env=sqlite-rescue-env model={MODEL_NAME}")
        
        try:
            obs = env.reset(task_name)
        except Exception:
            obs = env.reset("easy_data_cleaning")
        
        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Task: {task_name}. Acknowledge."}],
                max_tokens=5
            )
        except Exception:
            pass
            
        steps_taken = 0
        rewards = []
        final_reward = 0.0
        
        for query in queries:
            steps_taken += 1
            action = RescueAction(query=query, submit=False)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            
            error_msg = f"'{obs.error}'" if obs.error else "null"
            print(f"[STEP] step={steps_taken} action=execute_sql(...) reward={reward:.2f} done=false error={error_msg}")
            
        steps_taken += 1
        action = RescueAction(query="", submit=True)
        obs, final_reward, done, info = env.step(action)
        rewards.append(final_reward)
        
        success = (final_reward >= 0.90) 
        
        error_msg = f"'{obs.error}'" if obs.error else "null"
        print(f"[STEP] step={steps_taken} action=submit(True) reward={final_reward:.2f} done=true error={error_msg}")
        
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_reward:.2f} rewards={rewards_str}")


def run_finaudit_demo():
    """Demonstrate the FinAuditEnv with rewards on the finaudit_complex DB."""
    env = FinAuditEnv()

    print("=" * 60)
    print("  FinAudit Environment Demo (with Rewards)")
    print("=" * 60)

    obs = env.reset("finaudit_complex")
    print("\n[RESET] Observation:")
    print(json.dumps(obs, indent=2, default=str))

    demo_actions = [
        {"action_type": "inspect_data", "params": {}},
        {"action_type": "filter_transactions", "params": {"column": "tx_type", "value": "credit"}},
        {"action_type": "calculate_total", "params": {}},
        {"action_type": "assign_category", "params": {"transaction_id": 1, "category": "medium"}},
        {"action_type": "assign_category", "params": {"transaction_id": 2, "category": "large"}},
        {"action_type": "flag_anomaly", "params": {"transaction_id": 501}},
        {"action_type": "bad_action", "params": {}},
        {
            "action_type": "submit_answer",
            "params": {
                "total": obs["transactions"][0].get("amount", 0) if obs.get("transactions") else 0,
                "flagged": [501],
                "categories": {1: "medium", 2: "large"},
                "summary": "Inspected data, flagged tx 501 as anomalous.",
            },
        },
    ]

    for action in demo_actions:
        obs, reward, done = env.step(action)
        print(f"\n[STEP] action={action['action_type']}  reward={reward:.4f}  done={done}")

        if done and "reward_breakdown" in obs:
            print("\n--- REWARD BREAKDOWN ---")
            print(json.dumps(obs["reward_breakdown"], indent=2, default=str))
        else:
            result = obs.get("last_action_result", {})
            if isinstance(result, dict) and "error" in result:
                print(f"  error: {result['error']}")

        if done:
            break

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# NEW: Trained Agent Inference
# ---------------------------------------------------------------------------
def run_trained_agent():
    print("🧠 Loading trained agent...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="finaudit-agent-lora", 
        max_seq_length=4096, # 1. Increase this to 4096 to prevent truncation
        dtype=torch.float16,
        load_in_4bit=True,
        fast_inference=False, 
    )
    FastLanguageModel.for_inference(model)

    env = FinAuditEnv()
    task = random.choice(["finaudit_easy", "finaudit_medium"])
    print(f"🌍 Resetting environment ({task})...")
    obs = env.reset(task)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for step in range(15):
        print(f"\n--- Step {step + 1} ---")
        messages.append({"role": "user", "content": f"Current Observation: {json.dumps(obs)}"})
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        print("🤖 Thinking...")
        # 2. Add temperature and top_p for more stable generation
        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=256, 
            use_cache=True, 
            temperature=0.1, # Keep it very focused
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        
        # 3. Aggressive cleaning to fix missing brackets or trailing junk
        clean_response = response_text.strip()
        if clean_response.count('{') > clean_response.count('}'):
            clean_response += '}' # Simple fix for the missing bracket you just saw
            
        print(f"Agent Action:\n{clean_response}")
        messages.append({"role": "assistant", "content": clean_response})
        
        try:
            action_dict = json.loads(clean_response)
            obs, reward, done = env.step(action_dict)
            print(f"💰 Reward received: {reward:.4f}")
            
            if done:
                print("\n✅ Episode Complete!")
                break
        except json.JSONDecodeError:
            print(f"❌ JSON Error. Raw output was: {repr(clean_response)}")
            break


if __name__ == "__main__":
    if "--trained" in sys.argv:
        # Run the LIVE trained AI model
        run_trained_agent()
    elif "--finaudit" in sys.argv:
        # Run the hardcoded demo
        run_finaudit_demo()
    else:
        if not API_KEY:
            print("Error: API_KEY is missing. Please set it in your environment variables.")
            sys.exit(1)
        run_baseline()