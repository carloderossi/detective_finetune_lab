# scripts/generate_synthetic_detective_data_fast.py
"""
High-throughput synthetic data generator with memory monitoring and GC.

Usage:
    python scripts/generate_synthetic_detective_data_fast.py
"""

import json
import random
import glob
import os
from pathlib import Path
from typing import List, Dict, Tuple

import gc
import psutil
import torch
from transformers import pipeline, AutoTokenizer
import yaml
from huggingface_hub import login
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = "data/cleaned"
OUTPUT_FILE = "data/train/detective_synthetic_v1.jsonl"

# MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Faster but lower-quality generations, still good for synthetic data
# MODEL_NAME= "meta-llama/Llama-4-Scout-17B-16E-Instruct" # requested access, better than Qwen 7B but slower and more expensive

MAX_NEW_TOKENS = 1400
TEMPERATURE = 0.82
TOP_P = 0.92
NUM_EXAMPLES_PER_CHAPTER = 4

# Reload pipeline every N chapters to avoid long-run memory creep
RELOAD_PIPELINE_EVERY = 12

# Log memory every N examples
MEMORY_LOG_EVERY = 15

INSTRUCTION_TEMPLATES = [
    {
        "instruction": "Continue the following detective story excerpt in the authentic style of {author_full}. "
                       "Maintain the period tone, sentence rhythm, vocabulary, and narrative voice. Build suspense "
                       "and include subtle observations or deductions. Write {target_length}–{target_length_upper} words.",
        "target_length": 600,
        "target_length_upper": 900,
    },
    {
        "instruction": "Write a complete scene (~{target_length}–{target_length_upper} words) in the style of {author_full} "
                       "based on this brief outline. Include authentic dialogue, atmosphere, misdirection or clever clues, "
                       "and period-appropriate details.",
        "target_length": 800,
        "target_length_upper": 1300,
    },
    {
        "instruction": "Rewrite the following modern text excerpt as if it were written by {author_full} in the 1920s–1930s "
                       "(for Christie) or late Victorian/Edwardian era (for Doyle). Use elegant phrasing, understated tension, "
                       "precise observations, and the characteristic narrative voice.",
        "target_length": 400,
        "target_length_upper": 700,
    },
    {
        "instruction": "Write the opening {target_length}–{target_length_upper} words of a new {detective_type} mystery "
                       "in the style of {author_full}. Introduce the detective, a puzzling situation or crime, and at least "
                       "two suspects with plausible motives. Use wit, misdirection, and atmospheric description.",
        "target_length": 700,
        "target_length_upper": 1100,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS: MEMORY + GC
# ──────────────────────────────────────────────────────────────────────────────

def print_memory(label: str = "") -> None:
    process = psutil.Process()
    rss = process.memory_info().rss / (1024 ** 2)

    msg = f"[MEM] {label} | RSS: {rss:.1f} MB"
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        msg += f" | CUDA alloc: {allocated:.1f} MB, reserved: {reserved:.1f} MB"
    print(msg)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clear_pipeline_cache(pipe) -> None:
    # Best-effort clearing of internal caches
    if hasattr(pipe, "model") and hasattr(pipe.model, "clear_cache"):
        try:
            pipe.model.clear_cache()
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS: DATA + PROMPTS
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
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text


def split_text_for_continuation(text: str, split_ratio: float = 0.45) -> Tuple[str, str]:
    paragraphs = text.split("\n\n")
    total_len = len(paragraphs)
    split_idx = max(3, int(total_len * split_ratio))
    context = "\n\n".join(paragraphs[:split_idx])
    continuation = "\n\n".join(paragraphs[split_idx:])
    return context, continuation


def generate_with_model(
    pipe,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    try:
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=TOP_P,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]

        if out.startswith(prompt):
            return out[len(prompt):].strip()
        return out.strip()
    except Exception as e:
        print(f"[WARN] Generation error: {e}")
        return ""


def build_example_from_template(
    template_idx: int,
    template: Dict,
    full_text: str,
    author_full: str,
) -> Dict:
    instr_base = template["instruction"]

    if "continue" in instr_base.lower():
        context, real_continuation = split_text_for_continuation(full_text)
        instruction = instr_base.format(
            author_full=author_full,
            target_length=template["target_length"],
            target_length_upper=template["target_length_upper"],
        )
        return {
            "instruction": instruction,
            "input": context,
            "output": real_continuation,
        }

    elif "rewrite" in instr_base.lower():
        paragraphs = full_text.split("\n\n")
        sample_para = random.choice(paragraphs[:10]) if paragraphs else ""
        instruction = instr_base.format(author_full=author_full)
        return {
            "instruction": instruction,
            "input": sample_para,
            "output": "",
        }

    else:
        instruction = instr_base.format(
            author_full=author_full,
            target_length=template["target_length"],
            target_length_upper=template["target_length_upper"],
            detective_type=random.choice(["Poirot", "Sherlock Holmes", "classic detective"]),
        )
        return {
            "instruction": instruction,
            "input": "",
            "output": "",
        }

# ──────────────────────────────────────────────────────────────────────────────
#  HF TOKEN
# ──────────────────────────────────────────────────────────────────────────────

def load_hf_token():
    config_path = "config.yaml"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        token = cfg.get("huggingface", {}).get("token")
        if token:
            return token
        else:
            raise ValueError("config.yaml found but token missing under huggingface.token")

    token = os.getenv("HF_TOKEN")
    if token:
        return token

    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    except Exception:
        pass

    raise EnvironmentError(
        "No Hugging Face token found. Provide config.yaml or set HF_TOKEN env var."
    )

# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def create_pipeline():
    print(f"[INIT] Loading model: {MODEL_NAME}")
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=True,
    )
    return pipe, pipe.tokenizer

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    hf_token = load_hf_token()
    login(token=hf_token)

    pipe, tokenizer = create_pipeline()

    chapter_files = sorted(glob.glob(f"{DATA_ROOT}/**/*.txt", recursive=True))
    print(f"[INFO] Found {len(chapter_files)} chapter files under {DATA_ROOT}")

    all_examples: List[Dict] = []
    example_counter = 0

    # Pre-create output dir
    out_dir = Path(OUTPUT_FILE).parent
    os.makedirs(out_dir, exist_ok=True)

    for idx, chapter_path in enumerate(tqdm(chapter_files, desc="Chapters"), 1):
        rel_path = Path(chapter_path).relative_to(DATA_ROOT)
        parts = rel_path.parts
        author = parts[0]
        book = parts[1]
        chapter = parts[2].replace(".txt", "")

        full_text = read_chapter(chapter_path)
        if len(full_text) < 800:
            continue

        author_full = get_author_full(author)

        for template_idx, template in enumerate(INSTRUCTION_TEMPLATES):
            if template_idx >= NUM_EXAMPLES_PER_CHAPTER:
                break

            example = build_example_from_template(
                template_idx, template, full_text, author_full
            )

            # Need generation?
            if not example["output"]:
                full_prompt = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f"{example['instruction']}\n\n{example['input']}",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                generated = generate_with_model(pipe, full_prompt)
                example["output"] = generated

            if len(example["output"].strip()) > 200:
                all_examples.append(example)
                example_counter += 1

                if example_counter % MEMORY_LOG_EVERY == 0:
                    print_memory(f"after {example_counter} examples")
                    clear_pipeline_cache(pipe)
                    cleanup()

        # Periodically reload pipeline to avoid long-run slowdown / fragmentation
        if idx % RELOAD_PIPELINE_EVERY == 0:
            print(f"[INFO] Reloading pipeline at chapter {idx}")
            del pipe
            cleanup()
            pipe, tokenizer = create_pipeline()
            print_memory(f"after pipeline reload at chapter {idx}")

    # Save incrementally at the end
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Generated {len(all_examples)} examples → {OUTPUT_FILE}")
    print_memory("final")


if __name__ == "__main__":
    main()