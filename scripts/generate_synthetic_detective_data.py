# scripts/generate_synthetic_detective_data.py
"""
Generate expanded instruction dataset for detective novel fine-tuning.

Requirements:
    pip install transformers datasets torch accelerate vllm  # if using vLLM
    # or just transformers if using normal pipeline

Usage:
    python scripts/generate_synthetic_detective_data.py \
        --model_name meta-llama/Llama-3.1-70B-Instruct \
        --output_path data/train/detective_synthetic.jsonl \
        --max_new_tokens 1200 \
        --temperature 0.8 \
        --num_examples_per_chapter 3
"""

import json
import random
import glob
import os
from pathlib import Path
from typing import List, Dict

from transformers import pipeline, AutoTokenizer
import torch

import yaml
from huggingface_hub import login

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = "data/cleaned"
OUTPUT_FILE = "data/train/detective_synthetic_v1.jsonl"

# Recommended strong models for generation (pick one you have access to)
# MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"          # Access Requested again on 16th of Feb
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
# MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"

MAX_NEW_TOKENS = 1400
TEMPERATURE = 0.82
TOP_P = 0.92
NUM_EXAMPLES_PER_CHAPTER = 4      # how many synthetic examples per chapter

# Seed prompts / templates (expand this list!)
INSTRUCTION_TEMPLATES = [
    # 1. Continuation
    {
        "instruction": "Continue the following detective story excerpt in the authentic style of {author_full}. Maintain the period tone, sentence rhythm, vocabulary, and narrative voice. Build suspense and include subtle observations or deductions. Write {target_length}–{target_length_upper} words.",
        "target_length": 600,
        "target_length_upper": 900,
    },
    # 2. Scene from outline
    {
        "instruction": "Write a complete scene (~{target_length}–{target_length_upper} words) in the style of {author_full} based on this brief outline. Include authentic dialogue, atmosphere, misdirection or clever clues, and period-appropriate details.",
        "target_length": 800,
        "target_length_upper": 1300,
    },
    # 3. Rewrite / style transfer
    {
        "instruction": "Rewrite the following modern text excerpt as if it were written by {author_full} in the 1920s–1930s (for Christie) or late Victorian/Edwardian era (for Doyle). Use elegant phrasing, understated tension, precise observations, and the characteristic narrative voice.",
        "target_length": 400,
        "target_length_upper": 700,
    },
    # 4. Open generation from seed
    {
        "instruction": "Write the opening {target_length}–{target_length_upper} words of a new {detective_type} mystery in the style of {author_full}. Introduce the detective, a puzzling situation or crime, and at least two suspects with plausible motives. Use wit, misdirection, and atmospheric description.",
        "target_length": 700,
        "target_length_upper": 1100,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_author_full(author: str) -> str:
    mapping = {
        "agatha_christie": "Agatha Christie",
        "arthur_conan_doyle": "Arthur Conan Doyle",
    }
    return mapping.get(author, author.replace("_", " ").title())


def read_chapter(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # Optional: remove excessive newlines
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text


def split_text_for_continuation(text: str, split_ratio: float = 0.45) -> tuple[str, str]:
    """Split chapter into context + target continuation"""
    paragraphs = text.split("\n\n")
    total_len = len(paragraphs)
    split_idx = max(3, int(total_len * split_ratio))
    context = "\n\n".join(paragraphs[:split_idx])
    continuation = "\n\n".join(paragraphs[split_idx:])
    return context, continuation


def generate_with_model(
    pipe: pipeline,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Generate text with safeguards"""
    try:
        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=TOP_P,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Extract only the newly generated part
        if output.startswith(prompt):
            return output[len(prompt):].strip()
        return output.strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN GENERATION LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_NAME}")
    # For very large models → consider vLLM or accelerate for speed
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=True,
    )
    tokenizer = pipe.tokenizer

    all_examples: List[Dict] = []

    chapter_files = sorted(glob.glob(f"{DATA_ROOT}/**/*.txt", recursive=True))
    print(f"Found {len(chapter_files)} chapter files")

    for idx, chapter_path in enumerate(chapter_files, 1):
        print(f"[{idx}/{len(chapter_files)}] Processing {chapter_path}")
        rel_path = Path(chapter_path).relative_to(DATA_ROOT)
        parts = rel_path.parts
        author = parts[0]
        book = parts[1]
        chapter = parts[2].replace(".txt", "")

        full_text = read_chapter(chapter_path)
        if len(full_text) < 800:
            print(" → Chapter too short, skipping")
            continue

        author_full = get_author_full(author)

        for template_idx, template in enumerate(INSTRUCTION_TEMPLATES):
            if template_idx >= NUM_EXAMPLES_PER_CHAPTER:
                break

            instr_base = template["instruction"]

            if "continuation" in instr_base.lower():
                # Continuation type
                context, real_continuation = split_text_for_continuation(full_text)
                instruction = instr_base.format(
                    author_full=author_full,
                    target_length=template["target_length"],
                    target_length_upper=template["target_length_upper"],
                )
                example = {
                    "instruction": instruction,
                    "input": context,
                    "output": real_continuation,
                }

            elif "rewrite" in instr_base.lower():
                # Take a random paragraph or two as modern text to rewrite
                paragraphs = full_text.split("\n\n")
                sample_para = random.choice(paragraphs[:10]) if paragraphs else ""
                instruction = instr_base.format(author_full=author_full)
                example = {
                    "instruction": instruction,
                    "input": sample_para,
                    "output": "",  # ← to be filled by model
                }

            else:
                # Open generation or outline-based
                instruction = instr_base.format(
                    author_full=author_full,
                    target_length=template["target_length"],
                    target_length_upper=template["target_length_upper"],
                    detective_type=random.choice(["Poirot", "Sherlock Holmes", "classic detective"]),
                )
                example = {
                    "instruction": instruction,
                    "input": "",
                    "output": "",  # ← to be filled
                }

            # If output is empty → generate it
            if not example["output"]:
                full_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                generated = generate_with_model(pipe, full_prompt)
                example["output"] = generated

            if len(example["output"].strip()) > 200:
                all_examples.append(example)
                print(f"  → Added example {len(all_examples)} ({len(example['output'])} chars)")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone! Generated {len(all_examples)} examples → {OUTPUT_FILE}")

def load_hf_token():
    config_path = "config.yaml"

    # 1. Check if config.yaml exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        token = cfg.get("huggingface", {}).get("token")
        if token:
            return token
        else:
            raise ValueError("config.yaml found but token missing under huggingface.token")

    # 2. Fallback to colab secrets
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
    if token:
        return token

    # 3. Neither found → fail clearly
    raise EnvironmentError(
        "No Hugging Face token found. Provide config.yaml or set HF_TOKEN env var."
    )

if __name__ == "__main__":
    hf_token = load_hf_token()
    login(token=hf_token)

    main()