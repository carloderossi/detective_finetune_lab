from unsloth import FastLanguageModel
import os
import torch

model_dir = "/content/drive/MyDrive/outputs/detective-qwen-sft"   
save_dir   = "/content/drive/MyDrive/models/detective-qwen-gguf"

os.makedirs(save_dir, exist_ok=True)

from datetime import datetime

# Path to this file
path = __file__

# Get last modified timestamp
modified_ts = os.path.getmtime(path)

# Convert to readable datetime
modified_dt = datetime.fromtimestamp(modified_ts)

print(f"Last modified [{path}]:", modified_dt)

print("Loading base + LoRA in one step...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = model_dir,
    max_seq_length   = 2048,
    load_in_4bit     = True,
    dtype            = None,
)

print("Merging LoRA into base model...")
model = model.merge_and_unload()

print("Saving merged full model...")
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)

print("Saving to GGUF ...")
# for q in ["q4_k_m", "q5_k_m", "q6_k"]:  # Uncomment if you want multiple quants
#     print(f"\nQuantizing to {q} ...")
#     model.save_pretrained_gguf(
#         save_directory   = save_dir,
#         tokenizer        = tokenizer,
#         quantization_method = q,
#     )
model.save_pretrained_gguf(
    save_directory   = save_dir,  
    tokenizer        = tokenizer,
    quantization_method = "q5_k_m" # "q4_k_m",   # alternatives: q5_k_m, q6_k, q8_0
)

print(f"✅ GGUF file saved in {save_dir}. You can now load this GGUF file for inference.")