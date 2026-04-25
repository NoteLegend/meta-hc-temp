"""
train_grpo.py — Train an LLM to act in the FinAudit environment using TRL GRPO and Unsloth.

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install trl datasets requests
"""

import os
import json
import random
import requests
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

# ===========================================================================
# 1. Configuration & Server Connection
# ===========================================================================
API_URL = "http://localhost:7860"

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

# ===========================================================================
# 2. Reward Functions
# ===========================================================================

def format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards the model +0.5 if it outputs strictly valid JSON."""
    rewards = []
    for completion in completions:
        try:
            # The completion is usually a list of dicts or a raw string depending on the tokenizer
            text = completion[0]["content"] if isinstance(completion, list) else completion
            parsed = json.loads(text.strip())
            if "action_type" in parsed and "params" in parsed:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        except json.JSONDecodeError:
            rewards.append(-0.5) # Penalty for hallucinating prose/markdown
    return rewards

def environment_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Connects to the local FastAPI server to test the generated action.
    Returns the step_reward computed by FinAuditEnv.
    """
    rewards = []
    for completion in completions:
        try:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            action_dict = json.loads(text.strip())
            
            # Step 1: Reset the environment to get a fresh state
            # Randomly select a task to prevent overfitting
            task = random.choice(["finaudit_easy", "finaudit_medium", "finaudit_complex"])
            requests.post(f"{API_URL}/reset?task_name={task}")
            
            # Step 2: Execute the generated action
            response = requests.post(f"{API_URL}/step", json=action_dict)
            
            if response.status_code == 200:
                step_data = response.json()
                rewards.append(float(step_data.get("reward", 0.0)))
            else:
                rewards.append(-1.0) # Server error penalty
                
        except Exception:
            # If JSON is invalid or server is down, penalty
            rewards.append(-1.0)
            
    return rewards

# ===========================================================================
# 3. Model & Dataset Preparation
# ===========================================================================

def prepare_dataset(num_samples=50):
    """Generates initial observation prompts to kick off the RL episodes."""
    prompts = []
    for _ in range(num_samples):
        # Fetch an initial state from the server
        task = random.choice(["finaudit_easy", "finaudit_medium", "finaudit_complex"])
        res = requests.post(f"{API_URL}/reset?task_name={task}")
        obs = res.json().get("observation", {})
        
        # Format for TRL Chat Template
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current Observation: {json.dumps(obs)}"}
        ])
    return Dataset.from_dict({"prompt": prompts})

def main():
    print("Loading Unsloth Model...")
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
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

    print("Fetching training states from Environment API...")
    dataset = prepare_dataset(num_samples=100)

    # ===========================================================================
    # 4. GRPO Training
    # ===========================================================================
    print("Initializing GRPOTrainer...")
    training_args = GRPOConfig(
        output_dir="outputs/finaudit-grpo",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=200, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4, # Number of actions sampled per prompt
        max_completion_length=256,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, environment_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting Training! Ensure FastAPI server is running on port 7860...")
    trainer.train()

    print("Saving trained model...")
    model.save_pretrained("finaudit-agent-lora")
    tokenizer.save_pretrained("finaudit-agent-lora")
    print("Training Complete!")

if __name__ == "__main__":
    main()