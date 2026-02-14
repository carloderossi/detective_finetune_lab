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
print(f"Loading model {MODEL_NAME} with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_LENGTH,
    load_in_4bit = True,
)

print(f"Model loaded. Vocabulary size: {tokenizer.vocab_size}")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer pad token set to: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

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
print(f"Applying LoRA with config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")

print("Applying LoRA to model...")
model = FastLanguageModel.get_peft_model(model, lora_config)
print("LoRA applied. Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape} ({param.numel()} params)")
    else:
        print(f"  {name}: {param.shape} (frozen)")
                 

# ----------------------------
# Load dataset
# ----------------------------
print(f"Loading dataset from {DATA_FILE}...")
dataset = load_dataset("json", data_files={"train": DATA_FILE})
print(f"Dataset loaded. Number of training examples: {len(dataset['train'])}")

def extract_text(example):
    if "text" in example:
        return example["text"]
    if "output" in example:
        return example["output"]
    if "instruction" in example and "output" in example:
        return example["instruction"] + "\n\n" + example["output"]
    raise ValueError("No usable text field found.")

dataset = dataset.map(lambda x: {"text": extract_text(x)})
print("Extracted text from dataset. Sample:")
print(dataset["train"][0]["text"][:500])

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

print("Tokenizing dataset...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
print("Dataset tokenized. Sample:")
print(tokenized["train"][0]["input_ids"][:50])

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
print(f"Training arguments set: batch_size={training_args.per_device_train_batch_size}, "
      f"grad_accum={training_args.gradient_accumulation_steps}, "   
      f"learning_rate={training_args.learning_rate}, max_steps={training_args.max_steps}")    

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
)

print("Trainer initialized. Beginning training loop...")
trainer.train()

print("Training completed. Saving model and tokenizer...")
model.save_pretrained("./detective-scout-unsloth")
tokenizer.save_pretrained("./detective-scout-unsloth")

print("Model and tokenizer saved to ./detective-scout-unsloth")