# test_unsloth_import.py
import os
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability())

# Optional: block zoo/RL parts that cause your error (uncomment if needed)
# sys.modules["unsloth_zoo"] = object()
# sys.modules["unsloth.models.rl"] = object()

try:
    print("\n=== Trying to import FastLanguageModel ===")
    from unsloth import FastLanguageModel
    
    print("Import successful!")
    
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"   # or your exact model
    MAX_LENGTH = 4096
    
    print(f"\nLoading model: {MODEL_NAME} (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_LENGTH,
        load_in_4bit   = True,
        dtype          = None,          # auto
    )
    
    print("Model loaded successfully!")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Has 4-bit quantization: {hasattr(model, 'quantization_method')}")
    
    # Very light test forward pass
    inputs = tokenizer(["Hello, this is a test."], return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model(**inputs)
    print("Forward pass OK!")
    
except Exception as e:
    print("\n!!! FAILED !!!")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All basic checks passed. You should be able to run training now.")