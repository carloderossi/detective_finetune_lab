from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig

MODEL_NAME = "meta-llama/Llama-4-8B-Scout"
DATA_FILE = "detective_finetune.jsonl"

# ----------------------------
# Hyperparameters
# ----------------------------
MAX_LENGTH = 4096
GRAD_ACCUM = 24
MAX_STEPS = 2200
BASE_LR = 1.5e-4
LOW_LR = 8e-5
USE_LOW_LR = False

learning_rate = LOW_LR if USE_LOW_LR else BASE_LR

# ----------------------------
# Load model with Unsloth
# ----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_LENGTH,
    load_in_4bit = True,
)

tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# LoRA configuration
# ----------------------------
lora_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0.05,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias = "none",
    task_type = "CAUSAL_LM",
)

model = FastLanguageModel.get_peft_model(model, lora_config)

# ----------------------------
# Load dataset
# ----------------------------
dataset = load_dataset("json", data_files={"train": DATA_FILE})

def extract_text(example):
    if "text" in example:
        return example["text"]
    if "output" in example:
        return example["output"]
    if "instruction" in example and "output" in example:
        return example["instruction"] + "\n\n" + example["output"]
    raise ValueError("No usable text field found.")

dataset = dataset.map(lambda x: {"text": extract_text(x)})

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./detective-scout-unsloth",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=learning_rate,
    max_steps=MAX_STEPS,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=25,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./detective-scout-unsloth")
tokenizer.save_pretrained("./detective-scout-unsloth")