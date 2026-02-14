import json
import math
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
FT_MODEL_PATH   = "./detective-qwen-sft"
RL_MODEL_PATH   = "./detective-qwen-rl"      # from PPO script
EMB_MODEL_NAME  = "thenlper/gte-large"

heldout_path    = "heldout_eval.jsonl"
eval_prompts    = "eval_prompts.jsonl"

# ----------------------------
# 1. Load models
# ----------------------------
print("Loading base model...")
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(DEVICE)

print("Loading SFT model...")
ft_tok = AutoTokenizer.from_pretrained(FT_MODEL_PATH)
ft_model = AutoModelForCausalLM.from_pretrained(FT_MODEL_PATH).to(DEVICE)

print("Loading RL model...")
rl_tok = AutoTokenizer.from_pretrained(RL_MODEL_PATH)
rl_model = AutoModelForCausalLM.from_pretrained(RL_MODEL_PATH).to(DEVICE)

print("Loading embedding model...")
emb_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

# ----------------------------
# 2. Helpers
# ----------------------------
def load_jsonl_texts(path: str, field: str = "text") -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj[field])
    return out

def compute_ppl(model, tokenizer, texts: List[str], max_length: int = 1024) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(DEVICE)
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            losses.append(out.loss.item())
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def style_similarity(a: str, b: str) -> float:
    embs = emb_model.encode([a, b], convert_to_tensor=True)
    return float(util.cos_sim(embs[0], embs[1]).item())

# ----------------------------
# 3. Perplexity on held‑out text
# ----------------------------
heldout_texts = load_jsonl_texts(heldout_path, field="text")

print("Computing perplexity on held‑out Holmes/Poirot text...")
base_ppl = compute_ppl(base_model, base_tok, heldout_texts)
ft_ppl   = compute_ppl(ft_model,   ft_tok,   heldout_texts)
rl_ppl   = compute_ppl(rl_model,   rl_tok,   heldout_texts)

print(f"Base PPL: {base_ppl:.2f}")
print(f"SFT  PPL: {ft_ppl:.2f}")
print(f"RL   PPL: {rl_ppl:.2f}")

import os, json
os.makedirs("logs", exist_ok=True)
with open("logs/eval_metrics.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "base_ppl": base_ppl,
        "ft_ppl": ft_ppl,
        "rl_ppl": rl_ppl,
        # optionally add aggregated style sims if you compute them
    }) + "\n")
    
# ----------------------------
# 4. Style similarity vs original passages
# ----------------------------
print("Evaluating style similarity on a few samples...")
for i, ref in enumerate(heldout_texts[:5]):
    prompt = "Continue in the same style:\n" + ref[:200]

    base_gen = generate(base_model, base_tok, prompt, max_new_tokens=200)
    ft_gen   = generate(ft_model,   ft_tok,   prompt, max_new_tokens=200)
    rl_gen   = generate(rl_model,   rl_tok,   prompt, max_new_tokens=200)

    base_sim = style_similarity(base_gen, ref)
    ft_sim   = style_similarity(ft_gen,   ref)
    rl_sim   = style_similarity(rl_gen,   ref)

    print(f"\nSample {i}:")
    print(f"  Base style sim: {base_sim:.3f}")
    print(f"  SFT  style sim: {ft_sim:.3f}")
    print(f"  RL   style sim: {rl_sim:.3f}")