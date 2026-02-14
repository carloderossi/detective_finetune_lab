import torch
import json
import random
from typing import List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ----------------------------
# Config
# ----------------------------
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./detective-qwen-sft"          # from Stage 1
PROMPT_DATA = "deduction_prompts.jsonl"     # short prompts like "Holmes explains the solution..."

JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # separate judge model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 1. Load policy (Qwen + LoRA + value head)
# ----------------------------
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
)

print("Loading LoRA adapter from:", LORA_PATH)
policy_lora = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Wrapping with value head...")
policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_lora)
policy.to(DEVICE)

# ----------------------------
# 2. Reward model (LLM-as-judge)
# ----------------------------
print("Loading judge model:", JUDGE_MODEL)
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
judge = pipeline(
    "text-generation",
    model=JUDGE_MODEL,
    tokenizer=judge_tokenizer,
    device_map="auto",
)

def score_deduction(text: str) -> float:
    """
    Ask the judge model to rate deduction quality from 1 to 10.
    Returns a float in [0, 1].
    """
    prompt = (
        "You are an expert literary critic and logician. "
        "Rate the following detective deduction passage from 1 to 10, "
        "where 10 means:\n"
        "- the clues are used logically,\n"
        "- the reasoning is clear and stepwise,\n"
        "- the solution follows from earlier details,\n"
        "- there are no obvious logical gaps.\n\n"
        "Passage:\n"
        f"{text}\n\n"
        "Answer ONLY with a number from 1 to 10."
    )

    out = judge(
        prompt,
        max_new_tokens=8,
        temperature=0.0,
        do_sample=False,
    )[0]["generated_text"]

    for tok in out.split():
        tok_clean = tok.strip().replace(".", "")
        if tok_clean.isdigit():
            val = int(tok_clean)
            if 1 <= val <= 10:
                return float(val) / 10.0
    return 0.5  # fallback neutral

# ----------------------------
# 3. PPO config
# ----------------------------
config = PPOConfig(
    model_name=BASE_MODEL,
    learning_rate=1e-6,   # small RL step
    batch_size=4,
    ppo_epochs=4,
    log_with=None,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=policy,
    tokenizer=tokenizer,
)

# ----------------------------
# 4. Load prompts
# ----------------------------
def load_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts

print("Loading prompts from:", PROMPT_DATA)
prompts = load_prompts(PROMPT_DATA)
print(f"Loaded {len(prompts)} prompts.")

# ----------------------------
# 5. Generation helper
# ----------------------------
def generate_response(prompt: str, max_new_tokens: int = 256) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        out = policy.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ----------------------------
# 6. RL loop
# ----------------------------
if __name__ == "__main__":
    epochs = 50  # small RL run

    for epoch in range(epochs):
        batch_prompts = random.sample(prompts, config.batch_size)

        texts = []
        rewards = []

        for p in batch_prompts:
            response = generate_response(p)
            texts.append(response)
            r = score_deduction(response)
            rewards.append(r)

        # tokenize for PPO
        tokenized_prompts = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)

        tokenized_responses = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)

        stats = ppo_trainer.step(
            queries=tokenized_prompts["input_ids"],
            responses=tokenized_responses["input_ids"],
            rewards=torch.tensor(rewards).to(DEVICE),
        )

        print(f"Epoch {epoch}: avg reward = {sum(rewards)/len(rewards):.3f}")
        # inside RL loop, after computing avg reward
        import os, json
        os.makedirs("logs", exist_ok=True)
        with open("logs/ppo_rewards.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "avg_reward": float(sum(rewards)/len(rewards))}) + "\n")

    print("Saving RL‑tuned policy...")
    policy.save_pretrained("./detective-qwen-rl")