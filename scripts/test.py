import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/detective-qwen-gguf"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)

# 2. Load Model 
# We remove the custom quant_config since it's already in the config.json
# We use low_cpu_mem_usage=True to prevent the "meta tensor" crash
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="sequential", # 'sequential' is often more stable than 'auto' on Windows
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16, # The compute dtype for the 4-bit weights
)

# 3. Simple Test Prompt
messages = [
    {"role": "system", "content": "You are a detective."},
    {"role": "user", "content": "Analyze the crime scene."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Model loaded! Generating...")
generated_ids = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])