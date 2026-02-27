# Save this to /content/drive/MyDrive/scripts/train_scout_unsloth.py
import os
import torch
import numpy as np

# FIX: Disables the broken Dynamo compiler in Torch 2.6 that causes the Traceback
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" # Enable expandable segments to reduce fragmentation and OOMs on large models with LoRA adapters, especially on 16GB GPUs. This allows PyTorch to better manage memory when loading and training large models with adapters, which can help prevent out-of-memory errors during training.

from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from unsloth import FastLanguageModel
from unsloth import unsloth_train

# FIX: Disable Torch Dynamo to prevent the __import__ graph break error
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
torch._dynamo.config.suppress_errors = True

# --- GOOGLE DRIVE PATHS ---
# Mounting is handled in the notebook; here we set the target paths
DATA_FILE = "/content/drive/MyDrive/data/train/detective_finetune.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/outputs/detective-qwen-sft"

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # 4 dry runs with 7B on T4, 100 steps takes ~20 mins, final LoRA ~1.5GB, total checkpoints ~2GB with 2 saved at a time. Qwen 14B is about 2x slower and larger, Llama-3.1-70B is about 4x slower and larger, Llama-4-Scout-17B is about 5-6x slower and larger but may give better results.
# MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # need to request access, better than Qwen 7B but slower and more expensive, estimated 5-6x slower and larger than Qwen 7B
# This model is "distilled" from a massive reasoning model (DeepSeek-R1). It excels at the "thinking" process (Chain of Thought), which is perfect for investigative/detective data where the model needs to connect clues.
# Unsloth Support: Unsloth has specialized kernels for Qwen-based architectures that make this run 2x faster on A100.
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # final production grade
# or even better for speed/memory:
# MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"   # Unsloth's pre-quantized 4-bit version (recommended)

# Change from 4096 to 2048 if using T4 to avoid OOM, A100 can handle 4096 but 2048 is safer for batch size and gradient accumulation
MAX_LENGTH = 2048
MAX_STEPS = 100      
GRAD_ACCUM = 4  # 8 is ideal for 16GB GPUs but may cause OOM on T4, 4 is safer and still gives good results with LoRA       
NUM_EPOCHS = 3           
BASE_LR = 1.5e-4
LOW_LR = 8e-5
USE_LOW_LR = False
LORA_DROPOUT = 0 

learning_rate = LOW_LR if USE_LOW_LR else BASE_LR

# Detect Hardware for A100 optimization
major_v, _ = torch.cuda.get_device_capability()
HAS_BF16 = major_v >= 8

print(f"Loading model {MODEL_NAME} with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_LENGTH,
    load_in_4bit = True,
)
model.gradient_checkpointing_enable()

tokenizer.pad_token = tokenizer.eos_token
print(f"Pad token set to: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

print("Creating PEFT model using Unsloth’s built‑in LoRA config helper...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Reduced from 32
    lora_alpha=32, # Adjusted to match r
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)
print("PEFT model created with LoRA adapters.")

print(f"Loading dataset from {DATA_FILE}...")
raw_dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]
print(f"Total examples: {len(raw_dataset)}")

# --- DataSet and Tokenizer ---

def extract_text(example):
    # Format metadata into a readable string
    author = example.get("author", "Unknown").replace("_", " ").title()
    book = example.get("book", "Unknown").replace("_", " ").title()
    
    header = f"### Author: {author}\n### Source: {book}\n\n"
    
    # Priority check for fields
    if "text" in example:
        return header + example["text"]
    if "output" in example:
        return header + example["output"]
    
    raise ValueError(f"Found sample with no usable text. Keys: {example.keys()}")

dataset = raw_dataset.map(lambda x: {"text": extract_text(x)})

# train / eval split
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=False,
        add_special_tokens=True,
    )

print("Tokenizing...")
train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
eval_tok  = eval_ds.map(tokenize,  batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // MAX_LENGTH) * MAX_LENGTH

    if total_length == 0:
        return {"input_ids": [], "attention_mask": []}

    input_ids = [
        concatenated[i : i + MAX_LENGTH]
        for i in range(0, total_length, MAX_LENGTH)
    ]
    attention_mask = [
        [1] * MAX_LENGTH for _ in range(len(input_ids))
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

print("Packing sequences...")
train_tok = train_tok.map(
    group_texts,
    batched=True,
    batch_size=None,
    remove_columns=train_tok.column_names,
)
eval_tok = eval_tok.map(
    group_texts,
    batched=True,
    batch_size=None,
    remove_columns=eval_tok.column_names,
)

print(f"Packed train samples: {len(train_tok)}")
print(f"Packed eval samples:  {len(eval_tok)}")

# --- UPDATED TRAINING ARGUMENTS FOR DRIVE & RESUME ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=learning_rate,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=5,
    report_to="none",
    optim="paged_adamw_8bit", # Good choice, keeps optimizer low memory

    # Precision
    fp16 = not HAS_BF16,
    bf16 = HAS_BF16,

    # 15GB Drive Protection & Auto-Resume Logic
    # eval_strategy="steps", # set eval_strategy="no" temporarily (skip eval), train to 100, then re-enable eval later on a machine with more VRAM.
    eval_strategy="no",
    # eval_steps=20,
    save_strategy="steps",
    save_steps=20,           # Save every 20 steps to handle Colab timeouts
    save_total_limit=2,      # CRITICAL: Keep only 2 checkpoints to fit in 15GB Drive
    load_best_model_at_end=False,
    per_device_eval_batch_size=1,          # ← CRITICAL: force eval micro-batch=1 (default may be higher)
    eval_accumulation_steps=4,        # accumulate eval over more micro-steps → lower peak VRAM
    fp16_full_eval=True,                   # Use FP16 for eval → halves eval memory (~50% savings)
    # OR if still tight: bf16_full_eval=True (but T4 doesn't support bf16 well → stick to fp16)
    gradient_checkpointing=True,           # Already enabled via model, but ensure
    
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    ppl = float(np.exp(loss.item()))
    return {"perplexity": ppl}

print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Check if a checkpoint exists in Drive to resume from
resume_from_checkpoint = True if os.path.exists(OUTPUT_DIR) and any("checkpoint" in d for d in os.listdir(OUTPUT_DIR)) else None
# trainer.train(resume_from_checkpoint=resume_from_checkpoint)
unsloth_train(
    trainer,
    resume_from_checkpoint=resume_from_checkpoint
)

print("Saving final model adapters to Drive...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Training complete. Model and adapters saved to {OUTPUT_DIR}")