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
#MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

DATA_FILE = "data/train/detective_finetune.jsonl"

MAX_LENGTH = 4096
GRAD_ACCUM = 24
MAX_STEPS = 2200
BASE_LR = 1.5e-4
LOW_LR = 8e-5
USE_LOW_LR = False

learning_rate = LOW_LR if USE_LOW_LR else BASE_LR

print(f"Loading model {MODEL_NAME} with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_LENGTH,
    load_in_4bit = True,
)

tokenizer.pad_token = tokenizer.eos_token
print(f"Pad token set to: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

lora_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0.05,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias = "none",
    task_type = "CAUSAL_LM",
)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(model, lora_config)

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

# --- sequence packing: concatenate then chunk into MAX_LENGTH ---
def group_texts(examples):
    # concatenate
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // MAX_LENGTH) * MAX_LENGTH
    concatenated = concatenated[:total_length]
    # chunk
    result = {
        "input_ids": [
            concatenated[i : i + MAX_LENGTH]
            for i in range(0, total_length, MAX_LENGTH)
        ]
    }
    result["attention_mask"] = [
        [1] * MAX_LENGTH for _ in range(len(result["input_ids"]))
    ]
    return result

print("Packing sequences...")
train_tok = train_tok.map(group_texts, batched=True)
eval_tok  = eval_tok.map(group_texts,  batched=True)

print(f"Packed train samples: {len(train_tok)}")
print(f"Packed eval samples:  {len(eval_tok)}")

training_args = TrainingArguments(
    output_dir="./detective-qwen-sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=learning_rate,
    max_steps=MAX_STEPS,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=4,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

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