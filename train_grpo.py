"""
train_grpo.py — Train an LLM to act in the FinAudit environment using TRL GRPO and Unsloth.
"""

import os
import json
import random
import requests
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ===========================================================================
# 1. Configuration & Server Connection
# ===========================================================================
API_URL = "http://localhost:8000"

# UPDATED: Added strict strategic rules to the system prompt
SYSTEM_PROMPT = """You are a financial auditing agent operating inside the FinAuditEnv reinforcement learning environment.
Your goal is to inspect the transactions table, calculate totals, categorize transactions by amount, flag anomalous transactions, and submit a final structured answer. 
You must act only by emitting exactly ONE valid JSON action object at a time. Do not include prose, Markdown, or explanations.

CRITICAL STRATEGY RULES:
1. Do not repeat actions you have already taken.
2. You MUST use 'calculate_total' before using 'submit_answer'.

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

# ===========================================================================
# 2. Reward Functions
# ===========================================================================

def format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards the model +0.5 if it outputs strictly valid JSON."""
    rewards = []
    for completion in completions:
        try:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            parsed = json.loads(text.strip())
            if "action_type" in parsed and "params" in parsed:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        except json.JSONDecodeError:
            rewards.append(-0.5) 
    return rewards

def environment_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Connects to the local FastAPI server to test the generated action."""
    rewards = []
    for completion in completions:
        try:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            action_dict = json.loads(text.strip())
            
            task = random.choice(["finaudit_easy", "finaudit_medium", "finaudit_complex"])
            requests.post(f"{API_URL}/reset?task_name={task}")
            
            response = requests.post(f"{API_URL}/step", json=action_dict)
            
            if response.status_code == 200:
                step_data = response.json()
                rewards.append(float(step_data.get("reward", 0.0)))
            else:
                rewards.append(-1.0) 
        except Exception:
            rewards.append(-1.0)
            
    return rewards

def strategy_reward_func(prompts, completions, **kwargs) -> list[float]:
    """NEW: Punishes bad strategic habits (repeating actions, submitting too early)."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            # Extract the prompt history to see what the agent already did
            prompt_text = prompt[-1]["content"] if isinstance(prompt, list) else prompt
            comp_text = completion[0]["content"] if isinstance(completion, list) else completion
            action_dict = json.loads(comp_text.strip())
            action_type = action_dict.get("action_type", "")

            reward = 0.0
            
            # Penalty for repeating an action (like inspect_data twice)
            if action_type in prompt_text and action_type not in ["assign_category", "flag_anomaly"]:
                reward -= 0.8
            
            # Massive penalty for submitting before calculating totals
            if action_type == "submit_answer" and "calculate_total" not in prompt_text:
                reward -= 1.5

            rewards.append(reward)
        except Exception:
            rewards.append(0.0) # Let the format_reward_func handle the JSON crashes
    return rewards

# ===========================================================================
# 3. Model & Dataset Preparation
# ===========================================================================

def prepare_dataset(num_samples=500):
    """NEW: Blends easy, medium, and complex databases to prevent overfitting."""
    print(f"Generating dynamic dataset of {num_samples} diverse scenarios...")
    prompts = []
    # Weighted distribution: 50% easy, 30% medium, 20% complex
    tasks = ["finaudit_easy", "finaudit_medium", "finaudit_complex"]
    weights = [0.5, 0.3, 0.2]
    
    for _ in range(num_samples):
        task = random.choices(tasks, weights=weights)[0]
        res = requests.post(f"{API_URL}/reset?task_name={task}")
        obs = res.json().get("observation", {})
        
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current Observation: {json.dumps(obs)}"}
        ])
    return Dataset.from_dict({"prompt": prompts})

def main():
    print("Loading Unsloth Model...")
    # UPDATED: Increased max_seq_length to 8192 to prevent context truncation
    max_seq_length = 8192 
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        dtype=torch.float16, 
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=16,
        gpu_memory_utilization=0.6,
    )
    
    print("Applying PEFT / LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # UPDATED: Increased num_samples to 500 for a rigorous curriculum
    dataset = prepare_dataset(num_samples=500)

    # ===========================================================================
    # 4. GRPO Training
    # ===========================================================================
    print("Initializing GRPOTrainer...")
    training_args = GRPOConfig(
        output_dir="outputs/finaudit-grpo-alpha",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=5, # Reduced logging noise
        max_steps=800,   # UPDATED: 800 steps for strategic mastery
        save_steps=200,  # Save checkpoints just in case
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4, 
        max_completion_length=256,
        fp16=True,  
        bf16=False, 
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        # UPDATED: Added the strategy_reward_func to the array
        reward_funcs=[format_reward_func, environment_reward_func, strategy_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting Phase 2 Training! Ensure FastAPI server is running on port 8000...")
    trainer.train()

    print("Saving Alpha-Auditor model...")
    model.save_pretrained("finaudit-agent-lora")
    tokenizer.save_pretrained("finaudit-agent-lora")
    print("Training Complete!")

if __name__ == "__main__":
    main()