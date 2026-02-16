from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

DATA_FILE = "data/train/detective_finetune.jsonl"

MAX_LENGTH = 4096
# Updated for a small dataset of 205 examples
MAX_STEPS = 100      # This will take ~2-3 hours instead of 2 days
GRAD_ACCUM = 8       # More updates for a small dataset
NUM_EPOCHS = 3           # Pass through the data 3 times
BASE_LR = 1.5e-4
LOW_LR = 8e-5
USE_LOW_LR = False

LORA_DROPOUT = 0 # 0.05 or 0.1 can help regularize when training for many steps on a small dataset, but it also slows down Unsloth (non-deterministic). 

learning_rate = LOW_LR if USE_LOW_LR else BASE_LR

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
    r=32,
    lora_alpha=64,
    lora_dropout=LORA_DROPOUT, # Unsloth falls back to slower kernels for LoRA layers. Adding dropout can help regularize and improve generalization when training for many steps on a small dataset.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    # task_type="CAUSAL_LM",
)
print("PEFT model created with LoRA adapters.")

print(f"Loading dataset from {DATA_FILE}...")
raw_dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]
print(f"Total examples: {len(raw_dataset)}")

def extract_text(example):
    if "text" in example:
        return example["text"]
    if "output" in example:
        return example["output"]
    if "instruction" in example and "output" in example:
        return example["instruction"] + "\n\n" + example["output"]
    raise ValueError("No usable text field found.")

dataset = raw_dataset.map(lambda x: {"text": extract_text(x)})

# train / eval split
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=False,      # no truncation here; we’ll pack later
        add_special_tokens=True,
    )

print("Tokenizing...")
train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
eval_tok  = eval_ds.map(tokenize,  batched=True, remove_columns=["text"])

# --- sequence packing: HuggingFace expects each batch to return a flat list of samples, not a nested structure.
# we concatenate all tokenized sequences and then split into chunks of MAX_LENGTH, creating new samples on the fly.
def group_texts(examples):
    # Concatenate all tokens
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
# HuggingFace  expects each column returned by  to have the same number of rows as the input batch. 
# Since we are concatenating and re-splitting, we set batch_size=None to process the entire dataset at once, and remove the original columns.
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

training_args = TrainingArguments(
    output_dir="./detective-qwen-sft",

    # --- BATCHING & GRADIENT QUALITY ---
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,   # 8
    max_steps=MAX_STEPS,                      # 100
    num_train_epochs=NUM_EPOCHS,              # 3

    # --- LEARNING ---
    learning_rate=BASE_LR if not USE_LOW_LR else LOW_LR,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=0.01,

    # --- EVALUATION (LIGHTWEIGHT & SAFE) ---
    eval_strategy="epoch",                    # evaluate once per epoch
    per_device_eval_batch_size=1,             # prevents OOM

    # --- PRECISION ---
    fp16=True,                                # your version supports this
    bf16=False,                               # ensure disabled on Colab T4

    # --- SAVING ---
    save_strategy="epoch",
    save_total_limit=2,

    # --- LOGGING ---
    logging_steps=5,
    report_to="none",

    # --- OPTIMIZER ---
    optim="paged_adamw_8bit",
)

# training_args = TrainingArguments(
#     output_dir="./detective-qwen-sft",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=GRAD_ACCUM,
#     learning_rate=learning_rate,
    
#     # --- EPOCH-BASED SETTINGS ---
#     num_train_epochs=NUM_EPOCHS,      # Use this instead of max_steps
#     eval_strategy="epoch",            # Evaluate at the end of every epoch
#     save_strategy="epoch",            # Save a checkpoint at the end of every epoch
#     # ----------------------------

#     warmup_ratio=0.05,
#     lr_scheduler_type="cosine",
#     logging_steps=5,                  # Log more frequently since total steps are low
#     save_total_limit=2,
#     fp16=True,
#     optim="paged_adamw_8bit",
#     report_to="none",
#     weight_decay=0.01,
# )

# training_args = TrainingArguments(
#     output_dir="./detective-qwen-sft",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=GRAD_ACCUM,
#     learning_rate=learning_rate,
#     max_steps=MAX_STEPS,
#     warmup_ratio=0.05,
#     lr_scheduler_type="cosine",
#     logging_steps=25,
#     eval_strategy="steps", # Use this! Delete 'evaluation_strategy' if it's there
#     eval_steps=20,
#     save_steps=20,
#     #save_total_limit=4,
#     fp16=True,
#     optim="paged_adamw_8bit", # Crucial for fitting the 7B model on a 16GB T4.
#     report_to="none",
#     weight_decay=0.01, # Adding a small weight decay helps prevent the model from "forgetting" its base knowledge during long runs
#     save_total_limit=2 # During a long run, generating checkpoints every 250 steps will quickly fill up your disk. Limit this to the last 2 best checkpoints
# )

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

def compute_metrics(eval_pred):
    # simple perplexity
    logits, labels = eval_pred
    # shift for causal LM
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
    # compute_metrics=compute_metrics,  # enable if you want PPL during eval
)

trainer.train()

print("Saving model...")
model.save_pretrained("./detective-qwen-sft")
tokenizer.save_pretrained("./detective-qwen-sft")