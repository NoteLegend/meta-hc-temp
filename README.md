# 🕵️‍♂️ Alpha-Auditor: Catching Temporal Fraud with Cognitive RL

**OpenEnv Hackathon Submission (India 2026)**

* 🚀 **[Play with the Environment on HF Spaces]**
* 📝 **[Read the 2-Min Story on our HF Blog]**
* 💻 **[Run the Inference in Google Colab]**

## 🛑 The Problem
Standard financial auditing agents are often just SQL scripts wrapped in an LLM shell. They rely on hardcoded thresholds (e.g., "Flag any amount > $5000"). While functional, they fail against sophisticated fraud like **Temporal Smurfing**—where bad actors structure multiple smaller transactions across a tight time window to evade threshold triggers.

## 🏗️ The Environment: FinAuditEnv
We built `FinAuditEnv` using OpenEnv to force an agent to perform **Multi-Modal Process Supervision**. 
* **Dynamic Goals:** The environment randomly demands "Credit" or "Debit" audits per episode, preventing memorization.
* **The Rulebook:** The agent is physically blocked from submitting an answer until it uses the `get_audit_policy` tool to read unstructured corporate law.
* **The Smurf Trap:** We injected temporal fraud rings (e.g., 6 transactions of $950 within 3 hours). The agent must use cross-referential reasoning to catch them.

## 🧠 The Training (GRPO on Llama 3.2-3B)
We trained a lightweight 3B model using Unsloth's GRPO. We designed a rigorous reward function:
* Micro-rewards for correct data categorization.
* Severe penalties for repeating actions or hallucinating JSON.
* **The Jackpot (+1.5):** A massive reward if the agent successfully explains its logic and flags the hidden Smurfing ID.

## 📊 Results & Evidence of Learning
The untrained baseline failed entirely (0% success), usually crashing due to bad JSON or guessing random math totals. After 800 steps, our Alpha-Auditor achieved a **58% success rate** across 50 dynamic episodes.

```text
========================================
📈 ALPHA-AUDITOR FINAL REPORT CARD
Total Episodes:     50
Success Rate:       58.0%
Average Reward:     -0.3711
Hallucination Rate: 38.0%
========================================
