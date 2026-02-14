# 🚀 Project Improvement Suggestions
## Detective Fiction Fine-Tuning Lab

---

## 🎯 Executive Summary

Your detective fine-tuning lab is well-structured with solid foundations. Below are **prioritized improvements** across security, code quality, features, and production readiness.

**Priority Levels:**
- 🟠 **HIGH**: Significantly improves project quality
- 🟡 **MEDIUM**: Nice-to-have enhancements
- 🟢 **LOW**: Future considerations

---

## 🔴 CRITICAL PRIORITIES

### Missing Error Handling in Training Scripts

**Issue**: Training scripts lack robust error handling, risking data loss on crashes.

**Current Code** (from `train_scout_unsloth.py`):
```python
# No try-except around training loop
model = FastLanguageModel.from_pretrained(...)
trainer = Trainer(...)
trainer.train()  # ⚠️ If this crashes, you lose everything
```

**Recommended Fix**:
```python
import sys
import traceback
from datetime import datetime

def train_with_safety():
    """Training with comprehensive error handling."""
    try:
        # Setup logging
        log_file = f"logs/training_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # Training code
        model = FastLanguageModel.from_pretrained(...)
        trainer = Trainer(...)
        
        # Checkpoint callback
        from transformers import TrainerCallback
        
        class SafetyCheckpoint(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 100 == 0:
                    print(f"✓ Checkpoint at step {state.global_step}")
        
        trainer.add_callback(SafetyCheckpoint())
        trainer.train()
        
        # Save final model
        model.save_pretrained("./detective-qwen-sft")
        print("✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        model.save_pretrained("./detective-qwen-sft-interrupted")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        traceback.print_exc()
        
        # Emergency save
        try:
            model.save_pretrained("./detective-qwen-sft-emergency")
            print("✓ Emergency checkpoint saved")
        except:
            print("❌ Could not save emergency checkpoint!")
        
        sys.exit(1)

if __name__ == "__main__":
    train_with_safety()
```

---

### No Backup Strategy for Models/Data

**Issue**: No automated backups for trained models or datasets.

**Recommended Solution**:

```python
# scripts/backup_manager.py
import shutil
from pathlib import Path
from datetime import datetime
import json

class BackupManager:
    def __init__(self, backup_root="backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
    
    def backup_model(self, model_path, metadata=None):
        """Backup a trained model with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{Path(model_path).name}_{timestamp}"
        backup_path = self.backup_root / backup_name
        
        # Copy model
        shutil.copytree(model_path, backup_path)
        
        # Save metadata
        if metadata:
            with open(backup_path / "metadata.json", "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "source": str(model_path),
                    **metadata
                }, f, indent=2)
        
        print(f"✓ Backed up to: {backup_path}")
        return backup_path
    
    def list_backups(self):
        """List all available backups."""
        backups = []
        for backup in self.backup_root.iterdir():
            if backup.is_dir():
                meta_path = backup / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    backups.append((backup.name, meta))
        return backups
    
    def restore(self, backup_name, target_path):
        """Restore a backup to target location."""
        backup_path = self.backup_root / backup_name
        shutil.copytree(backup_path, target_path, dirs_exist_ok=True)
        print(f"✓ Restored from: {backup_path}")

# Usage in training script
backup_mgr = BackupManager()
backup_mgr.backup_model(
    "./detective-qwen-sft",
    metadata={"step": trainer.state.global_step, "loss": final_loss}
)
```

---

## 🟠 HIGH PRIORITY

### Add Comprehensive Logging

**Current Issue**: Limited visibility into training progress and debugging.

**Recommended Solution**:

```python
# utils/logger.py
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console_fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)
    
    # File handler
    if log_file:
        log_path = Path("logs") / log_file
        log_path.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger

# Usage in training scripts
logger = setup_logger(
    "detective_training",
    log_file=f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
)

logger.info("Starting training...")
logger.debug(f"Model config: {model.config}")
logger.warning("High memory usage detected")
logger.error("Training failed!", exc_info=True)
```

---

### Implement Experiment Tracking

**Recommended**: Use Weights & Biases or MLflow for experiment tracking.

**Option 1: Weights & Biases (Simple)**:
```python
# Add to pyproject.toml
# dependencies = [..., "wandb"]

import wandb

# In train script
wandb.init(
    project="detective-finetune",
    config={
        "model": MODEL_NAME,
        "learning_rate": BASE_LR,
        "max_steps": MAX_STEPS,
        "lora_r": 32
    }
)

# TrainingArguments integration
training_args = TrainingArguments(
    ...,
    report_to="wandb",  # Auto-log metrics
    run_name="holmes-poirot-v1"
)

# Manual logging
wandb.log({"custom_metric": value, "step": step})
```

**Option 2: MLflow (Self-hosted)**:
```python
import mlflow

mlflow.set_experiment("detective-finetune")

with mlflow.start_run(run_name="holmes-poirot-v1"):
    # Log parameters
    mlflow.log_params({
        "model": MODEL_NAME,
        "learning_rate": BASE_LR,
        "max_steps": MAX_STEPS
    })
    
    # Log metrics
    mlflow.log_metrics({"loss": loss, "step": step})
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

---

### Add Unit Tests

**Current Issue**: No automated testing for dataset processing or utilities.

**Recommended Structure**:
```
tests/
├── __init__.py
├── test_dataset_quality.py
├── test_data_loader.py
├── test_generation.py
└── conftest.py  # pytest fixtures
```

**Example Test**:
```python
# tests/test_dataset_quality.py
import pytest
from pathlib import Path
import json

def test_jsonl_format():
    """Ensure all training data is valid JSONL."""
    data_file = Path("data/train/detective_finetune.jsonl")
    
    with open(data_file) as f:
        for i, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                assert "author" in entry
                assert "text" in entry
                assert len(entry["text"]) > 0
            except Exception as e:
                pytest.fail(f"Line {i} invalid: {e}")

def test_no_duplicates():
    """Ensure no duplicate text in dataset."""
    data_file = Path("data/train/detective_finetune.jsonl")
    texts = set()
    
    with open(data_file) as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            assert text not in texts, "Duplicate text found!"
            texts.add(text)

def test_token_length_distribution():
    """Check token length distribution is reasonable."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    lengths = []
    
    data_file = Path("data/train/detective_finetune.jsonl")
    with open(data_file) as f:
        for line in f:
            entry = json.loads(line)
            tokens = tokenizer.encode(entry["text"])
            lengths.append(len(tokens))
    
    avg_length = sum(lengths) / len(lengths)
    assert 4000 < avg_length < 7000, f"Avg length {avg_length} outside expected range"

# Run with: pytest tests/
```

---

### Create Environment Setup Script

**Recommended**: One-command setup for new contributors.

```bash
# setup.sh
#!/bin/bash
set -e

echo "🔍 Detective Fine-Tune Lab Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.12+ required (found $python_version)"
    exit 1
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create venv
echo "🐍 Creating virtual environment..."
uv venv

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
uv pip install -e .

# Create directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,cleaned,train,evaluation,rl}
mkdir -p models logs reports backups

# Check for config
if [ ! -f "config.yaml" ]; then
    echo "⚠️  config.yaml not found!"
    echo "📝 Creating template..."
    cat > config.yaml << EOF
huggingface:
  token: ""  # Add your HF token here or set HF_TOKEN env var

training:
  model: "Qwen/Qwen2.5-7B-Instruct"
  max_seq_length: 4096
  batch_size: 1
  gradient_accumulation_steps: 24
EOF
    echo "✏️  Please edit config.yaml and add your HuggingFace token"
fi

# Download sample data (if needed)
if [ ! -f "data/train/detective_finetune.jsonl" ]; then
    echo "⚠️  No training data found in data/train/"
    echo "   Run data preparation scripts first"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml and add your HF token"
echo "  2. Run: python scripts/dataset_quality_checker.py"
echo "  3. Run: python scripts/train_scout_unsloth.py"
echo ""
echo "Activate environment with: source .venv/bin/activate"
```

**Make executable**:
```bash
chmod +x setup.sh
./setup.sh
```

---

## 🟡 MEDIUM PRIORITY

### Add Data Versioning (DVC)

**Purpose**: Track dataset changes and experiment reproducibility.

```bash
# Install DVC
pip install dvc dvc-gdrive  # or dvc-s3, dvc-azure

# Initialize
dvc init

# Track data
dvc add data/train/detective_finetune.jsonl
git add data/train/detective_finetune.jsonl.dvc .gitignore
git commit -m "Track training data with DVC"

# Setup remote (Google Drive example)
dvc remote add -d myremote gdrive://folder-id
dvc push  # Upload to remote storage
```

**Benefits**:
- Version control for large datasets
- Easy rollback to previous data versions
- Share data without bloating git repo

---

### Implement Hyperparameter Tuning

**Current**: Hardcoded hyperparameters in scripts.

**Recommended**: Use Optuna for automated tuning.

```python
# scripts/hyperparam_search.py
import optuna
from train_scout_unsloth import train_model

def objective(trial):
    """Optuna optimization objective."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    lora_r = trial.suggest_categorical("lora_r", [16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [32, 64, 128])
    
    # Train with these params
    final_loss = train_model(
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    
    return final_loss

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best params:", study.best_params)
print("Best loss:", study.best_value)
```

---

