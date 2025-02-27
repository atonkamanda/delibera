"""
rl.py

This module performs a reinforcement learning (RL) training loop using TRL's PPOTrainer.
The RL stage refines the SFT model using a judge-style reward computed with access to safety policies.

The RL stage follows the paper's approach:
  - For each prompt (from a safety-relevant dataset), the model generates a response using a
    policy-aware generation (i.e. including policy context in the system prompt).
  - A judge rates the response for correctness,
    helpfulness, and adherence to the safety style guidelines.
  - Multiple judge evaluations (k runs) are performed and the minimum score is taken.
  - The PPO objective uses these rewards to update the model.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from datasets import load_dataset
import yaml

# -----------------
# Helper functions (simulate the judge evaluation)
# -----------------

def detect_category(text):
    """
    A simple heuristic to detect the safety category in the query.
    For full implementation, replace with a proper categorization method.
    """
    patterns = {
        "illicit_behavior": r"\b(illegal|untraceable|shady|law enforcement|cops|hide)\b",
        "self_harm": r"\b(suicide|self-harm|kill myself|not continuing life)\b",
        "harassment": r"\b(hate|harass|threat|stalk|bully)\b",
        "sexual_content": r"\b(porn|sexual|nsfw|adult site)\b",
        "regulated_advice": r"\b(medical|legal|financial advice)\b"
    }
    for cat, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return cat
    return "safe"

def is_helpful(response):
    """
    Dummy heuristic for helpfulness. For example, check if the response is not empty and has a minimal length.
    """
    return len(response.split()) >= 5

def is_proper_refusal(response, policy):
    """
    Dummy heuristic for proper refusal.
    Check if the response contains a brief apology and explicit refusal language.
    """
    lower = response.lower()
    # Check for common tokens in proper refusals.
    return ("sorry" in lower or "can't" in lower or "cannot" in lower)

def compute_judge_score(response, query, policies, k=3):
    """
    Simulate the judge running k times over (query, response) pair using the safety policies.
    Here we use heuristic checks:
      - For safe queries, we reward helpful responses.
      - For non-safe queries where a refusal is expected (based on the policy's response style),
        we reward if the response is a proper refusal.
      - Otherwise, a neutral baseline is used.
    The final score is the minimum over k runs.
    """
    scores = []
    # Retrieve the available safety policies.
    safety_policies = policies.get("safety_policies", {})
    for _ in range(k):
        cat = detect_category(query)
        # Get the policy for the detected category (if available)
        policy = safety_policies.get(cat, {})
        
        if cat == "safe":
            score = 1.0 if is_helpful(response) else 0.2
        # If the policy (if provided) indicates a refusal style, check the response accordingly.
        elif policy.get("response_styles", {}).get("refuse", {}).get("template", "").lower():
            score = 1.0 if is_proper_refusal(response, policy) else -1.0
        else:
            score = 0.5  # Neutral baseline for other cases.
        scores.append(score)
    return min(scores)

def compute_rewards(responses, queries, policies, k=3):
    """
    Compute a reward for each (query, response) pair by running the judge simulation k times.
    Returns a torch tensor of rewards.
    """
    reward_list = []
    for response, query in zip(responses, queries):
        reward = compute_judge_score(response, query, policies, k)
        reward_list.append(reward)
    return torch.tensor(reward_list, dtype=torch.float32)

# -----------------
# RL Training function
# -----------------

def load_policies(policies_config_path):
    with open(policies_config_path, "r") as f:
        policies = yaml.safe_load(f)
    return policies

def train_rl(args):
    # Load safety policies from YAML file.
    policies = load_policies(args["policies_config"])
    print("Loaded policies:", policies)

    # Load a dataset of prompts for RL training.
    # For demonstration, we use the first 100 samples.
    dataset = load_dataset(args["dataset"], split="train")
    
    # Check dataset structure and extract prompts properly
    if "text" in dataset.column_names:
        # If dataset has a 'text' field, use it directly
        prompts = dataset["text"][:100]
    elif "prompt" in dataset.column_names:
        # If dataset has a 'prompt' field, use it directly
        prompts = dataset["prompt"][:100]
    elif isinstance(dataset[0], dict) and "prompt" in dataset[0]:
        # If dataset items are dictionaries with 'prompt' key
        prompts = [sample["prompt"] for sample in dataset[:100]]
    elif isinstance(dataset[0], str):
        # If dataset items are directly strings, use them as prompts
        prompts = dataset[:100]
    else:
        # Fall back to first 100 examples, try to convert to string
        prompts = [str(sample) for sample in dataset[:100]]
    
    print(f"Loaded {len(prompts)} prompts for RL training")
    
    # Load the SFT checkpoint model (starting point for RL)
    model = AutoModelForCausalLM.from_pretrained(args["rl_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(args["rl_model_path"])

    # Prepare PPO configuration using command-line arguments.
    ppo_config = PPOConfig(
        learning_rate=float(args["ppo_learning_rate"]),
        mini_batch_size=int(args["ppo_mini_batch_size"]),
        batch_size=int(args["ppo_batch_size"]),
        update_epochs=int(args["ppo_epochs"]),
        cliprange=float(args["ppo_cliprange"]),
        cliprange_value=float(args["ppo_cliprange_value"]),
        vf_coef=float(args["ppo_vf_coef"]),
        initial_kl_coef=float(args["ppo_initial_kl_coef"]),
    )

    # Initialize the PPO trainer.
    trainer = PPOTrainer(model=model, tokenizer=tokenizer, **ppo_config.__dict__)

    # Define a policy-aware generation function.
    def generate_with_policy(query):
        cat = detect_category(query)
        # Retrieve the safety policy for the detected category.
        safety_policies = policies.get("safety_policies", {})
        policy = safety_policies.get(cat, {})
        
        # Format the policy in a more readable way instead of dumping the raw object
        policy_text = f"Category: {cat}"
        if "description" in policy:
            policy_text += f"\nDescription: {policy['description']}"
        if "response_styles" in policy:
            styles = policy.get("response_styles", {})
            if "refuse" in styles:
                policy_text += f"\nRefusal style: {styles['refuse'].get('template', '')}"
        
        # Build a system prompt with policy context.
        prompt = f"[System: Follow these policies: {policy_text}]\n{query}"
        # Encode the prompt and generate.
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        # Generation parameters can be tuned (here we use default settings).
        output_ids = model.generate(input_ids, max_length=input_ids.shape[1] + 50)
        return output_ids

    # RL training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")
        # Process the data in batches.
        for i in range(0, len(prompts), ppo_config.batch_size):
            batch_prompts = prompts[i: i + ppo_config.batch_size]
            # Tokenize the batch prompts.
            query_tensors = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            # Generate responses using the policy-aware prompt.
            response_tensors = []
            decoded_responses = []
            for prompt_str in batch_prompts:
                out_ids = generate_with_policy(prompt_str)
                # Extract just the generated tokens, not including the prompt
                prompt_length = query_tensors["input_ids"].shape[1]
                response_tensor = out_ids[:, prompt_length:]
                response_tensors.append(response_tensor)
                decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                decoded_responses.append(decoded)

            # Stack response tensors for proper formatting
            stacked_responses = torch.cat(response_tensors, dim=0)

            # Compute rewards for each (query, response) pair.
            rewards = compute_rewards(decoded_responses, batch_prompts, policies, k=3)
            rewards_tensor = rewards.to(model.device)

            # Run a PPO step using the trainer.
            stats = trainer.step(query_tensors["input_ids"], stacked_responses, rewards_tensor)
            print(f"Processed batch {i // ppo_config.batch_size + 1}, PPO stats: {stats}")

    # Save the RL-refined model checkpoint.
    rl_model_path = "rl_model"
    model.save_pretrained(rl_model_path)
    tokenizer.save_pretrained(rl_model_path)
    print("Reinforcement learning training completed. RL model saved at:", rl_model_path) 