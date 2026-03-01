# =============================================================================
# train_unsloth_config.py
# Config-driven Unsloth SFT fine-tuning script.
# Usage:  python train_unsloth_config.py --config path/to/config.yaml
# =============================================================================

import argparse
import os
import sys
import yaml
import torch
import numpy as np

# ---------------------------------------------------------------------------
# 1.  Parse CLI argument FIRST so we can load env-fixes from the config
#     before importing heavy libraries.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Unsloth SFT trainer — config driven")
parser.add_argument(
    "--config", required=True,
    help="Path to the YAML config file, e.g. config_t4_train100_no_evals.yaml"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 2.  Load YAML config
# ---------------------------------------------------------------------------
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

print(f"\n{'='*70}")
print(f"  Config  : {cfg.get('config_name', args.config)}")
print(f"  GPU     : {cfg.get('gpu_target', 'auto-detect')}")
print(f"  Desc    : {cfg.get('description', '')}")
print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# 3.  Apply environment fixes (only the ones enabled in the config)
# ---------------------------------------------------------------------------
env_cfg = cfg.get("env", {})

if env_cfg.get("disable_torchdynamo", False):
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # Also set TORCH_COMPILE_BACKEND before torch is imported
    os.environ["TORCH_COMPILE_BACKEND"] = "eager"
    print("[env] TorchDynamo disabled (Torch 2.6 fix)")

if env_cfg.get("hf_hub_enable_transfer", True):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("[env] HF_HUB fast transfer enabled")

alloc_conf = env_cfg.get("pytorch_alloc_conf", "")
if alloc_conf:
    os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
    print(f"[env] PYTORCH_ALLOC_CONF = {alloc_conf}")

# ---------------------------------------------------------------------------
# 4.  Now import the heavy libraries
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel, unsloth_train
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

if env_cfg.get("suppress_dynamo_errors", False):
    torch._dynamo.config.suppress_errors = True
    print("[env] torch._dynamo suppress_errors = True")

# ---------------------------------------------------------------------------
# 5.  Unpack config sections
# ---------------------------------------------------------------------------
paths_cfg    = cfg["paths"]
model_cfg    = cfg["model"]
lora_cfg     = cfg["lora"]
train_cfg    = cfg["training"]
eval_cfg     = cfg.get("eval", {})
ckpt_cfg     = cfg.get("checkpoint", {})
dataset_cfg  = cfg.get("dataset", {})

DATA_FILE  = paths_cfg["data_file"]
OUTPUT_DIR = paths_cfg["output_dir"]

MODEL_NAME     = model_cfg["name"]
LOAD_IN_4BIT   = model_cfg.get("load_in_4bit", True)

MAX_LENGTH  = train_cfg["max_length"]
MAX_STEPS   = train_cfg["max_steps"]
GRAD_ACCUM  = train_cfg["grad_accum"]
NUM_EPOCHS  = train_cfg["num_epochs"]
BASE_LR     = float(train_cfg["base_lr"])
LOW_LR      = float(train_cfg.get("low_lr", 8e-5))
USE_LOW_LR  = train_cfg.get("use_low_lr", False)
LORA_DROPOUT        = lora_cfg.get("lora_dropout", 0)
USE_UNSLOTH_TRAIN   = train_cfg.get("use_unsloth_train", True)
INCLUDE_CHAPTER     = dataset_cfg.get("include_chapter_in_header", False)

learning_rate = LOW_LR if USE_LOW_LR else BASE_LR

# Eval settings
EVAL_ENABLED          = eval_cfg.get("enabled", True)
EVAL_STEPS            = eval_cfg.get("eval_steps", 20)
EVAL_BATCH_SIZE       = eval_cfg.get("per_device_eval_batch_size", 1)
EVAL_ACCUM_STEPS      = eval_cfg.get("eval_accumulation_steps", 4)
FP16_FULL_EVAL        = eval_cfg.get("fp16_full_eval", True)

# Checkpoint / resume settings
SAVE_STEPS            = ckpt_cfg.get("save_steps", 20)
SAVE_TOTAL_LIMIT      = ckpt_cfg.get("save_total_limit", 2)
LOAD_BEST_AT_END      = ckpt_cfg.get("load_best_model_at_end", False)
RESUME                = ckpt_cfg.get("resume", True)

# Dataset split
TEST_SIZE = dataset_cfg.get("test_size", 0.05)
SEED      = dataset_cfg.get("seed", 42)

# ---------------------------------------------------------------------------
# 6.  Detect hardware precision capability
# ---------------------------------------------------------------------------
major_v, _ = torch.cuda.get_device_capability()
HAS_BF16 = major_v >= 8
print(f"[hw] GPU compute capability {major_v}.x — bf16={'yes' if HAS_BF16 else 'no (fp16 mode)'}")

# ---------------------------------------------------------------------------
# 7.  Load model & tokenizer
# ---------------------------------------------------------------------------
print(f"\n[model] Loading {MODEL_NAME} with Unsloth (4bit={LOAD_IN_4BIT})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)
model.gradient_checkpointing_enable()

tokenizer.pad_token = tokenizer.eos_token
print(f"[model] Pad token → {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

# ---------------------------------------------------------------------------
# 8.  Build LoRA / PEFT model
# ---------------------------------------------------------------------------
print("[lora] Attaching LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    lora_dropout=LORA_DROPOUT,
    target_modules=lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    bias=lora_cfg.get("bias", "none"),
)
print(f"[lora] r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, dropout={LORA_DROPOUT}")

# ---------------------------------------------------------------------------
# 9.  Load & preprocess dataset
# ---------------------------------------------------------------------------
print(f"\n[data] Loading from {DATA_FILE}...")
raw_dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]
print(f"[data] Total examples: {len(raw_dataset)}")

def extract_text(example):
    author  = example.get("author",  "Unknown").replace("_", " ").title()
    book    = example.get("book",    "Unknown").replace("_", " ").title()

    if INCLUDE_CHAPTER:
        chapter = example.get("chapter", "Unknown Chapter").replace("_", " ").title()
        header = f"### Author: {author}\n### Source: {book} - {chapter}\n\n"
    else:
        header = f"### Author: {author}\n### Source: {book}\n\n"

    if "text" in example:
        return header + example["text"]
    if "instruction" in example and "output" in example:
        return header + example["instruction"] + "\n\n" + example["output"]
    if "output" in example:
        return header + example["output"]

    raise ValueError(f"No usable text field found. Keys: {list(example.keys())}")

dataset = raw_dataset.map(lambda x: {"text": extract_text(x)})
dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
train_ds = dataset["train"]
eval_ds  = dataset["test"]

def tokenize(batch):
    return tokenizer(batch["text"], truncation=False, add_special_tokens=True)

print("[data] Tokenizing...")
train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
eval_tok  = eval_ds.map(tokenize,  batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated  = sum(examples["input_ids"], [])
    total_length  = (len(concatenated) // MAX_LENGTH) * MAX_LENGTH
    if total_length == 0:
        return {"input_ids": [], "attention_mask": []}
    input_ids      = [concatenated[i: i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
    attention_mask = [[1] * MAX_LENGTH for _ in range(len(input_ids))]
    return {"input_ids": input_ids, "attention_mask": attention_mask}

print("[data] Packing sequences...")
train_tok = train_tok.map(group_texts, batched=True, batch_size=None, remove_columns=train_tok.column_names)
eval_tok  = eval_tok.map( group_texts, batched=True, batch_size=None, remove_columns=eval_tok.column_names)
print(f"[data] Packed train samples: {len(train_tok)}")
print(f"[data] Packed eval  samples: {len(eval_tok)}")

# ---------------------------------------------------------------------------
# 10. Build TrainingArguments
# ---------------------------------------------------------------------------
eval_strategy_value = "steps" if EVAL_ENABLED else "no"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Batching & steps
    per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    num_train_epochs=NUM_EPOCHS,

    # LR & schedule
    learning_rate=learning_rate,
    warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
    lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
    weight_decay=train_cfg.get("weight_decay", 0.01),

    # Logging
    logging_steps=train_cfg.get("logging_steps", 5),
    report_to="none",
    optim=train_cfg.get("optim", "paged_adamw_8bit"),

    # Precision — auto-detected from GPU
    fp16=not HAS_BF16,
    bf16=HAS_BF16,

    # Gradient checkpointing
    gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),

    # Eval
    eval_strategy=eval_strategy_value,
    eval_steps=EVAL_STEPS if EVAL_ENABLED else None,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    eval_accumulation_steps=EVAL_ACCUM_STEPS,
    fp16_full_eval=FP16_FULL_EVAL if not HAS_BF16 else False,

    # Checkpointing
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=LOAD_BEST_AT_END,
)

# ---------------------------------------------------------------------------
# 11. Data collator & metrics
# ---------------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    return {"perplexity": float(np.exp(loss.item()))}

# ---------------------------------------------------------------------------
# 12. Build Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok if EVAL_ENABLED else None,
    data_collator=data_collator,
    compute_metrics=compute_metrics if EVAL_ENABLED else None,
)

# ---------------------------------------------------------------------------
# 13. Auto-resume logic
# ---------------------------------------------------------------------------
resume_from_checkpoint = None
if RESUME and os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if "checkpoint" in d]
    if checkpoints:
        resume_from_checkpoint = True
        print(f"[resume] Found checkpoint(s): {checkpoints} — resuming training.")
    else:
        print("[resume] No checkpoint found — starting fresh.")
else:
    print("[resume] Resume disabled or output dir does not exist — starting fresh.")

# ---------------------------------------------------------------------------
# 14. Train
# ---------------------------------------------------------------------------
print(f"\n[train] Starting training — {MAX_STEPS} steps, evals={'ON' if EVAL_ENABLED else 'OFF'}...")

if USE_UNSLOTH_TRAIN:
    print("[train] Using unsloth_train() (optimised kernel path)")
    unsloth_train(trainer, resume_from_checkpoint=resume_from_checkpoint)
else:
    print("[train] Using standard HF trainer.train()")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# ---------------------------------------------------------------------------
# 15. Save final adapters
# ---------------------------------------------------------------------------
print("\n[save] Saving final LoRA adapters and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅  Done — model saved to {OUTPUT_DIR}")
