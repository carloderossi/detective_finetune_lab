import json
import random
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
FT_MODEL_PATH   = "./detective-qwen-sft"
RL_MODEL_PATH   = "./detective-qwen-rl"

EVAL_PROMPTS = "eval_prompts.jsonl"
HELDOUT_PATH = "heldout_eval.jsonl"


# ----------------------------
# Load models
# ----------------------------
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    return tok, model


print("Loading base model...")
base_tok, base_model = load_model(BASE_MODEL_NAME)

print("Loading SFT model...")
ft_tok, ft_model = load_model(FT_MODEL_PATH)

print("Loading RL model...")
rl_tok, rl_model = load_model(RL_MODEL_PATH)


# ----------------------------
# Generation helper
# ----------------------------
def generate(model, tokenizer, prompt, max_new_tokens=300):
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ----------------------------
# Judge prompt builder
# ----------------------------
def build_judge_prompt(prompt, text_a, text_b, text_c, reference):
    return f"""
You are an expert literary critic and logician.

You will be given:
- A detective prompt
- Three anonymous candidates (A, B, C)
- A reference passage from classic detective fiction

For each candidate, rate from 1 to 10 on:
1. Holmes/Poirot stylistic fidelity
2. Logical deduction quality
3. Narrative coherence
4. Consistency of clues

Then provide a final ranking (best to worst).

Respond in JSON with the following structure:

{{
  "A": {{"style": int, "logic": int, "coherence": int, "clues": int}},
  "B": {{"style": int, "logic": int, "coherence": int, "clues": int}},
  "C": {{"style": int, "logic": int, "coherence": int, "clues": int}},
  "ranking": ["A", "B", "C"]
}}

Detective prompt:
{prompt}

Candidate A:
{text_a}

Candidate B:
{text_b}

Candidate C:
{text_c}

Reference passage (for style calibration only, do NOT score this):
{reference}
"""


# ----------------------------
# Judge wrapper
# ----------------------------
def judge_three(judge_client, prompt, text_a, text_b, text_c, reference):
    jp = build_judge_prompt(prompt, text_a, text_b, text_c, reference)
    raw = judge_client.complete(jp)
    return json.loads(raw)


# ----------------------------
# Load data
# ----------------------------
def load_jsonl(path, field):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj[field])
    return out


# ----------------------------
# Main
# ----------------------------
def main(judge_client):
    prompts = load_jsonl(EVAL_PROMPTS, "prompt")
    references = load_jsonl(HELDOUT_PATH, "text")

    for i, prompt in enumerate(prompts):
        print(f"\n=== Prompt {i} ===")
        print(prompt)

        # generate from all models
        base_out = generate(base_model, base_tok, prompt)
        ft_out   = generate(ft_model,   ft_tok,   prompt)
        rl_out   = generate(rl_model,   rl_tok,   prompt)

        # pick a random reference passage
        ref = random.choice(references)

        # judge
        result = judge_three(
            judge_client,
            prompt,
            base_out,
            ft_out,
            rl_out,
            ref,
        )

        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import subprocess
    import json

    class OllamaJudge:
        def complete(self, prompt: str) -> str:
            result = subprocess.run(
                ["ollama", "run", "mistral-nemo"],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
            )
            return result.stdout.decode("utf-8")

    judge_client = OllamaJudge()
    main(judge_client)
