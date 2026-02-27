# test_unsloth_smoke.py
import sys

print("Starting smoke test...")

# These two lines reproduce your exact crash point
try:
    from unsloth import FastLanguageModel
    print("Import of FastLanguageModel succeeded ✓")
except Exception as e:
    print("Import failed:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Optional safety for your specific error
try:
    import unsloth.models.rl
    print("WARNING: rl module was importable — may still crash later")
except:
    print("rl module import blocked or not present (good for SFT)")

print("\nSmoke test passed → imports look healthy.")