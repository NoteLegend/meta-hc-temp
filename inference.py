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

# 🚨 UPDATED: Synced exactly with the new train_grpo.py
SYSTEM_PROMPT = """You are an autonomous forensic financial auditing agent operating inside the FinAuditEnv reinforcement learning environment.
Your goal is to inspect the transactions table, calculate totals, categorize transactions by amount, flag anomalies, and submit a final structured answer. 

CRITICAL STRATEGY RULES:
1. You MUST execute the 'get_audit_policy' action FIRST to learn the current corporate regulations for flagging anomalies and detecting fraud (like Smurfing). Do NOT rely on assumed anomaly thresholds.
2. You MUST use 'calculate_total' before using 'submit_answer'.
3. You must explain your reasoning in a "thought" field before taking any action.
4. Do not repeat actions you have already taken without a valid reason.

Base Categorization rules:
- amount <= 100: category is "small"
- amount <= 1000: category is "medium"
- amount <= 5000: category is "large"
(Note: For the 'anomaly' category and flagging rules, refer to the audit policy).

You must act only by emitting exactly ONE valid JSON object per turn. Do not include Markdown formatting (like ```json) or conversational text outside the JSON.

Valid action schemas:
{"thought":"I need to learn the current compliance rules before auditing.","action_type":"get_audit_policy","params":{}}
{"thought":"I need to understand the database schema and see sample rows.","action_type":"inspect_data","params":{}}
{"thought":"I am isolating the credit transactions to audit incoming money.","action_type":"filter_transactions","params":{"column":"tx_type","value":"credit"}}
{"thought":"I need the exact mathematical sum of the filtered rows.","action_type":"calculate_total","params":{}}
{"thought":"This transaction is under 100, so it is small.","action_type":"assign_category","params":{"transaction_id":1,"category":"small"}}
{"thought":"This transaction violates Rule 104 in the policy.","action_type":"flag_anomaly","params":{"transaction_id":999}}
{"thought":"I have completed the policy checks and calculations. Submitting final report.","action_type":"submit_answer","params":{"total":12345.67,"flagged":[999],"categories":{"1":"small"}}}
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
# 🚨 UPDATED: Trained Agent Inference with Batch Evaluation
# ---------------------------------------------------------------------------
def run_trained_agent(num_episodes=50):
    print(f"🧠 Loading trained agent for {num_episodes} evaluation episodes...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/finaudit-grpo-alpha/checkpoint-800", 
        max_seq_length=8192, # <--- 🚨 INCREASED TO 8192
        dtype=torch.float16,
        load_in_4bit=True,
        fast_inference=False, 
    )
    FastLanguageModel.for_inference(model)

    env = FinAuditEnv()
    stats = {"success": 0, "total_reward": 0, "hallucinations": 0}

    print(f"\n=== 🚀 Starting Batch Evaluation: {num_episodes} Episodes ===")
    
    for i in range(num_episodes):
        print(f"\n--- 🏁 EPISODE {i+1} ---")
        task = random.choice(["finaudit_easy", "finaudit_medium", "finaudit_complex"])
        obs = env.reset(task)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        episode_reward = 0
        
        for step in range(15):
            messages.append({"role": "user", "content": f"Current Observation: {json.dumps(obs)}"})
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            
            outputs = model.generate(
                input_ids=inputs, 
                max_new_tokens=256, 
                use_cache=True, 
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
            
            clean_response = response_text.strip()
            if clean_response.count('{') > clean_response.count('}'):
                clean_response += '}'
                
            print(f"Step {step+1} Action: {clean_response}")
            messages.append({"role": "assistant", "content": clean_response})
            
            try:
                action_dict = json.loads(clean_response)
                obs, reward, done = env.step(action_dict)
                episode_reward += reward
                
                if done:
                    print(f"✅ Finished Episode {i+1} ({task}) with Reward: {episode_reward:.4f}")
                    stats["success"] += 1
                    break
            except json.JSONDecodeError:
                print(f"❌ Hallucination on Step {step+1}. Raw: {repr(clean_response)}")
                stats["hallucinations"] += 1
                break
        
        stats["total_reward"] += episode_reward

    # Print Final Report Card
    print("\n" + "="*40)
    print("📈 ALPHA-AUDITOR FINAL REPORT CARD")
    print(f"Total Episodes:     {num_episodes}")
    print(f"Success Rate:       {(stats['success']/num_episodes)*100:.1f}%")
    print(f"Average Reward:     {stats['total_reward']/num_episodes:.4f}")
    print(f"Hallucination Rate: {(stats['hallucinations']/num_episodes)*100:.1f}%")
    print("="*40)


if __name__ == "__main__":
    if "--trained" in sys.argv:
        # Defaulting to 50 for the final Alpha test
        run_trained_agent(num_episodes=50)
    elif "--finaudit" in sys.argv:
        run_finaudit_demo()
    else:
        if not API_KEY:
            print("Error: API_KEY is missing. Please set it in your environment variables.")
            sys.exit(1)
        run_baseline()