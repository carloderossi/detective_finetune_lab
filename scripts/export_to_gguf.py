from unsloth import FastLanguageModel
import os

model_dir = "/content/drive/MyDrive/outputs/detective-qwen-sft"   
save_dir = "/content/drive/MyDrive/models/detective-qwen-gguf"
os.makedirs(save_dir, exist_ok=True)

print("Loading base model + LoRA adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = "Qwen/Qwen2.5-7B-Instruct",   # or "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length   = 2048,
    load_in_4bit     = True,
    dtype            = None,
)

print("Loading your trained LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r                  = 16,
    lora_alpha         = 32,
    target_modules     = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout       = 0,
    bias               = "none",
)
model.load_adapter(model_dir)   # ← this loads your adapter_model.safetensors

print("Merging LoRA into base model...")
model = model.merge_and_unload()

print("Saving to GGUF ...")
for q in ["q4_k_m", "q5_k_m", "q6_k"]:
    print(f"\nQuantizing to {q} ...")
    model.save_pretrained_gguf(
        save_directory   = save_dir,
        tokenizer        = tokenizer,
        quantization_method = q,
    )
# model.save_pretrained_gguf(
#     save_directory   = save_dir,  
#     tokenizer        = tokenizer,
#     quantization_method = "q4_k_m",   # alternatives: q5_k_m, q6_k, q8_0
# )

print(f"Done! GGUF file should be in {save_dir}. You can now load this GGUF file with libraries like vLLM or gguf.js for inference.")