# Unsloth SFT Fine-Tuning — Configuration Guide

This document explains how to use `train_unsloth_config.py` and describes each of the 6 YAML configuration files.

---

## How It Works

The script `train_unsloth_config.py` is a single, config-driven fine-tuning script. Every parameter that used to be hardcoded (model name, GPU settings, LoRA rank, eval toggles, etc.) is now read from a YAML file at runtime.

**Basic usage:**
```bash
python train_unsloth_config.py --config path/to/your_config.yaml
```

**Example on Google Colab:**
```python
!python /content/drive/MyDrive/scripts/train_unsloth_config.py \
    --config /content/drive/MyDrive/scripts/config_t4_train100_no_evals.yaml
```

You switch between scenarios simply by changing the `--config` argument — no code edits needed.

---

## Config File Structure

Every YAML config is divided into these sections:

| Section | What it controls |
|---|---|
| `env` | OS-level environment fixes (TorchDynamo, HF transfer, memory allocator) |
| `paths` | `data_file` (JSONL) and `output_dir` (Drive) |
| `model` | HuggingFace model name, 4-bit quantisation |
| `lora` | LoRA rank `r`, `lora_alpha`, dropout, target modules |
| `training` | Steps, epochs, LR, batch size, grad accumulation, optimiser |
| `eval` | Whether evals run, how often, memory-saving settings |
| `checkpoint` | Save frequency, how many to keep, auto-resume toggle |
| `dataset` | Train/eval split ratio, random seed, chapter header toggle |

---

## The 6 Configurations

### The 6 configs at a glance

| Config File                         | GPU          | Model                     | Steps | Evals | LoRA r | Context |
|-------------------------------------|--------------|---------------------------|-------|-------|--------|---------|
| config_t4_train100_no_evals.yaml    | T4 16GB      | Qwen2.5‑7B                | 100   | ❌    | 16     | 2048    |
| config_t4_train20_full_evals.yaml   | T4 16GB      | Qwen2.5‑7B                | 20    | ✅    | 16     | 2048    |
| config_a10g_14b_train100.yaml       | A10G 24GB    | Qwen2.5‑14B               | 100   | ✅    | 32     | 4096    |
| config_a10g_17b_scout_full.yaml     | A10G 24GB    | Llama‑4‑Scout‑17B         | 120   | ✅    | 32     | 4096    |
| config_a100_32b_production.yaml     | A100 40GB    | DeepSeek‑R1‑Qwen‑32B      | 200   | ✅    | 64     | 4096    |
| config_a100_80gb_fullblown.yaml     | A100 80GB    | DeepSeek‑R1‑32B‑4bit      | 300   | ✅    | 128    | 8192    |


### 1. `config_t4_train100_no_evals.yaml`
**Scenario:** Fast iterative training on a free Colab T4 — no evals to save VRAM.

| Parameter | Value |
|---|---|
| GPU target | T4 (16 GB) |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Max steps | 100 |
| Evaluations | ❌ Disabled |
| LoRA r / alpha | 16 / 32 |
| Context length | 2048 |
| Grad accumulation | 4 |
| Auto-resume | ✅ Yes |
| Est. time | ~20 min on T4 |

**When to use:** Your go-to config for the bulk of training on a T4. Evals are turned off because running them on a T4 with a 7B model can cause OOM errors. Run this first to accumulate steps, then follow up with config #2 to evaluate the checkpoint.

**Key settings:**
- `eval.enabled: false` — skips all evaluation, frees ~3 GB VRAM
- `max_length: 2048` — safer than 4096 on 16 GB
- `grad_accum: 4` — 8 would OOM on T4 with LoRA; 4 is stable
- `save_total_limit: 2` — keeps only 2 checkpoints to stay within Google Drive's 15 GB free quota

---

### 2. `config_t4_train20_full_evals.yaml`
**Scenario:** Short top-up training run (20 steps) with full evaluation on a T4.

| Parameter | Value |
|---|---|
| GPU target | T4 (16 GB) |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Max steps | 20 |
| Evaluations | ✅ Full (every 20 steps) |
| LoRA r / alpha | 16 / 32 |
| Context length | 2048 |
| Grad accumulation | 4 |
| Auto-resume | ✅ Yes |
| Est. time | ~5 min + eval on T4 |

**When to use:** After running config #1 to accumulate training steps, use this to evaluate the quality of the saved checkpoint. The 20 training steps are minimal — the primary purpose here is to trigger the eval loop.

**Key settings:**
- `eval_accumulation_steps: 8` — spreads eval across 8 micro-steps to cut peak VRAM
- `per_device_eval_batch_size: 1` — critical for T4; higher values cause OOM during eval
- `fp16_full_eval: true` — halves eval memory footprint on T4 (T4 doesn't support bf16)
- `eval_steps: 20` — eval fires once, at the end of the 20-step run

---

### 3. `config_a10g_14b_train100.yaml`
**Scenario:** Mid-size model on A10G — step up from T4 experiments to a 14B model.

| Parameter | Value |
|---|---|
| GPU target | A10G (24 GB) |
| Model | `Qwen/Qwen2.5-14B-Instruct` |
| Max steps | 100 |
| Evaluations | ✅ Every 20 steps |
| LoRA r / alpha | 32 / 64 |
| Context length | 4096 |
| Grad accumulation | 8 |
| Auto-resume | ✅ Yes |
| Est. time | ~40–50 min on A10G |

**When to use:** When you have access to a Colab Pro+ A10G or equivalent and want to move beyond the 7B model. The 14B model produces meaningfully better output on complex detective/reasoning data with only ~2x the cost.

**Key settings:**
- `max_length: 4096` — A10G can handle full context; 2048 is no longer necessary
- `grad_accum: 8` — ideal for 24 GB with 14B + LoRA
- `lora.r: 32` — double the T4 configs; richer adapter capacity
- `include_chapter_in_header: true` — adds chapter-level metadata to training examples

---

### 4. `config_a10g_17b_scout_full.yaml`
**Scenario:** Llama-4-Scout-17B on A10G — pushing A10G to its comfortable ceiling.

| Parameter | Value |
|---|---|
| GPU target | A10G (24 GB) |
| Model | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| Max steps | 120 (100 base + 20 extension) |
| Evaluations | ✅ Every 20 steps |
| LoRA r / alpha | 32 / 64 |
| Context length | 4096 |
| Grad accumulation | 8 |
| Auto-resume | ✅ Yes |
| Est. time | ~2–3 hrs on A10G |

**When to use:** When quality matters more than speed and you have sustained A10G access. Scout-17B is ~5–6x slower than Qwen2.5-7B but produces stronger results on investigative/reasoning-heavy data. Requires requesting access on HuggingFace first.

> ⚠️ **Access required:** You must request access to `meta-llama/Llama-4-Scout-17B-16E-Instruct` on HuggingFace before using this config. Set your HF token in Colab secrets.

**Key settings:**
- `base_lr: 1.2e-4` — slightly lower than smaller models for stability at 17B
- `max_steps: 120` — mirrors the `evalsonly` version's 100+20 pattern

---

### 5. `config_a100_32b_production.yaml`
**Scenario:** Production fine-tune on A100 40GB with DeepSeek-R1-Distill-Qwen-32B.

| Parameter | Value |
|---|---|
| GPU target | A100 (40 GB) |
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| Max steps | 200 |
| Evaluations | ✅ Every 20 steps |
| LoRA r / alpha | 64 / 128 |
| Context length | 4096 |
| Grad accumulation | 8 |
| Auto-resume | ✅ Yes |
| Est. time | ~3–4 hrs on A100-40GB |

**When to use:** Serious production run. DeepSeek-R1 is distilled from a large reasoning model — it is the best choice for detective/Chain-of-Thought data because it natively excels at multi-step inference. Unsloth has dedicated Qwen-architecture kernels that run 2x faster on A100.

**Key settings:**
- `eval_accumulation_steps: 16` — more aggressive accumulation on A100 for lower peak eval VRAM
- `lora.r: 64` — much richer adapter; A100 40GB handles this comfortably
- `save_total_limit: 3` — one extra checkpoint kept vs T4 configs
- `base_lr: 1.0e-4` — conservative LR for a 32B model

---

### 6. `config_a100_80gb_fullblown.yaml`
**Scenario:** Maximum everything — A100 80GB, 32B model, 8K context, r=128, 300 steps.

| Parameter | Value |
|---|---|
| GPU target | A100 (80 GB) |
| Model | `unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit` |
| Max steps | 300 |
| Evaluations | ✅ Every 20 steps |
| LoRA r / alpha | 128 / 256 |
| Context length | 8192 |
| Grad accumulation | 16 |
| Batch size | 2 |
| Epochs | 5 |
| Auto-resume | ✅ Yes |
| Est. time | ~4–6 hrs on A100-80GB |

**When to use:** Final, full-scale fine-tune. No compromises. The Unsloth pre-quantized 4-bit variant (`bnb-4bit`) loads faster and is more stable than loading raw weights and quantizing on the fly. The 8K context window captures much longer detective passages without truncation.

**Key settings:**
- `per_device_train_batch_size: 2` — 80 GB can sustain batch=2 even at 8K context
- `lora.r: 128` — maximum adapter expressiveness; captures subtle stylistic patterns
- `max_length: 8192` — double the A100-40GB config; full long-form passages
- `save_total_limit: 5` — valuable run; keep more checkpoints for rollback
- `base_lr: 8.0e-5` — conservative LR for large r + large model combination

---

## Recommended Progression

```
① config_t4_train100_no_evals      → bulk training, cheap, fast
         ↓
② config_t4_train20_full_evals     → validate checkpoint quality on T4
         ↓
③ config_a10g_14b_train100         → step up to 14B on A10G
         ↓
④ config_a10g_17b_scout_full       → push A10G with Scout-17B
         ↓
⑤ config_a100_32b_production       → production run on A100-40GB
         ↓
⑥ config_a100_80gb_fullblown       → full-blown final fine-tune on A100-80GB
```

You do not have to follow every step — jump to whichever tier matches your current GPU access.

---

## Key Parameters Cheat Sheet

| Config | GPU | Model size | Steps | Evals | LoRA r | Context | Grad acc | Est. time |
|---|---|---|---|---|---|---|---|---|
| t4_train100_no_evals | T4 16GB | 7B | 100 | ❌ | 16 | 2048 | 4 | ~20 min |
| t4_train20_full_evals | T4 16GB | 7B | 20 | ✅ | 16 | 2048 | 4 | ~5 min |
| a10g_14b_train100 | A10G 24GB | 14B | 100 | ✅ | 32 | 4096 | 8 | ~50 min |
| a10g_17b_scout_full | A10G 24GB | 17B | 120 | ✅ | 32 | 4096 | 8 | ~2–3 hrs |
| a100_32b_production | A100 40GB | 32B | 200 | ✅ | 64 | 4096 | 8 | ~3–4 hrs |
| a100_80gb_fullblown | A100 80GB | 32B | 300 | ✅ | 128 | 8192 | 16 | ~4–6 hrs |

---

## Adding a New Config

1. Copy the closest existing YAML file and rename it
2. Change `config_name` and `description` at the top
3. Adjust the sections you need — everything else stays as-is
4. Run with `--config your_new_config.yaml`

No changes to `train_unsloth_config.py` are ever needed.


---

## Cost, VRAM & Time Estimates

> **Assumptions:** Small dataset (<5K examples / <50MB JSONL), ~3,000 training examples at ~300 avg tokens each → ~440 packed samples at 2048 context, ~220 at 4096, ~110 at 8192. Colab Pay As You Go pricing at $0.10/compute unit.

---

### ⚠️ Critical: What GPUs Are Actually Available on Colab PAYG?

This is the single most important thing to understand before planning your runs:

| GPU | Colab PAYG (no subscription) | Colab Pro $9.99/mo | Colab Pro+ $49.99/mo |
|---|---|---|---|
| **T4 (16 GB)** | ✅ Yes | ✅ Yes | ✅ Yes |
| **L4 (24 GB)** | ⚠️ Sometimes | ⚠️ Sometimes | ✅ Better access |
| **A10G (24 GB)** | ❌ Not available | ❌ Not available | ❌ Not available |
| **A100 40GB** | ❌ Not available | ⚠️ Rarely, not guaranteed | ✅ Possible but not guaranteed |
| **A100 80GB** | ❌ Not available | ❌ Not available | ❌ Not available |

**Bottom line:**
- On pure Pay As You Go, you can reliably run only **configs #1 and #2** (T4).
- Configs #3 and #4 (A10G) require an **alternative platform** — Colab does not offer A10G at all.
- Config #5 (A100 40GB) requires at minimum **Colab Pro+** and even then is not guaranteed.
- Config #6 (A100 80GB) is **not available on Colab** at any tier. Use RunPod, Lambda Labs, or Paperspace.

For configs #3–#6, recommended alternatives are listed in the cost table below.

---

### Compute Unit Rates (Colab)

| GPU | Compute units/hr | Cost/hr (at $0.10/CU) |
|---|---|---|
| T4 | ~1.96 CU/hr | **~$0.20/hr** |
| L4 | ~4.5 CU/hr | **~$0.45/hr** |
| A100 40GB | ~13–15 CU/hr | **~$1.30–1.50/hr** |
| A100 80GB | Not on Colab | — |

---

### Per-Config Estimates

#### Config 1 — `config_t4_train100_no_evals` (T4, 7B, 100 steps, no evals)

| Dimension | Estimate | Notes |
|---|---|---|
| **Platform** | Colab PAYG ✅ | T4 available on all plans |
| **GPU VRAM** | ~8–10 GB peak | Comfortable on T4 (15 GB usable) |
| **System RAM** | ~6–8 GB | Colab T4 provides 12.7–25.5 GB; select High RAM to be safe |
| **Google Drive storage** | ~3–4 GB | 2 checkpoints (~1 GB each) + final LoRA (~1.5 GB) |
| **Time** | ~20–25 min | ~10 sec/step with Unsloth, plus ~4 min model load |
| **Cost (Colab PAYG)** | **~$0.08** | 0.42 hr × $0.20/hr |

---

#### Config 2 — `config_t4_train20_full_evals` (T4, 7B, 20 steps + full evals)

| Dimension | Estimate | Notes |
|---|---|---|
| **Platform** | Colab PAYG ✅ | T4 available on all plans |
| **GPU VRAM** | ~10–12 GB peak | Eval spike +2 GB vs training; fp16_full_eval keeps it manageable |
| **System RAM** | ~6–8 GB | Same as config #1 |
| **Google Drive storage** | ~3–4 GB | Same as config #1 |
| **Time** | ~10–15 min | ~3 min training + ~5 min eval + ~4 min model load |
| **Cost (Colab PAYG)** | **~$0.05** | 0.25 hr × $0.20/hr |

> 💡 **Tip:** Run config #1 five times (5 × $0.08 = $0.40) to reach 500 steps, then run config #2 once ($0.05) to evaluate. Total cost for a solid T4 training + eval cycle: **~$0.45**.

---

#### Config 3 — `config_a10g_14b_train100` (A10G, 14B, 100 steps + evals)

| Dimension | Estimate | Notes |
|---|---|---|
| **Platform** | ❌ Not on Colab — use RunPod / Lambda Labs / Paperspace |
| **GPU VRAM** | ~13–16 GB peak | A10G has 24 GB — comfortable headroom |
| **System RAM** | ~10–14 GB | Paged optimizer offloads ~7 GB to CPU RAM for 14B model |
| **Google Drive storage** | ~5–7 GB | 2 checkpoints (~2.5 GB each) + final LoRA (~3 GB) |
| **Time** | ~45–55 min | ~17 sec/step, 5 eval runs × ~2 min, ~6 min model load |
| **Cost (RunPod A10G ~$0.65/hr)** | **~$0.60** | 0.92 hr × $0.65/hr |
| **Cost (Lambda Labs A10G ~$0.75/hr)** | **~$0.69** | 0.92 hr × $0.75/hr |

---

#### Config 4 — `config_a10g_17b_scout_full` (A10G, Scout-17B, 120 steps + evals)

| Dimension | Estimate | Notes |
|---|---|---|
| **Platform** | ❌ Not on Colab — use RunPod / Lambda Labs / Paperspace |
| **GPU VRAM** | ~15–19 GB peak | Scout is MoE (16 experts) — all must be in VRAM simultaneously |
| **System RAM** | ~12–16 GB | Paged optimizer for 17B ~8.5 GB to CPU RAM |
| **Google Drive storage** | ~6–8 GB | Scout LoRA adapter + 2 checkpoints |
| **Time** | ~85–100 min | ~30 sec/step (MoE overhead), 6 eval runs × ~3 min, ~8 min model load |
| **Cost (RunPod A10G ~$0.65/hr)** | **~$1.08** | 1.67 hr × $0.65/hr |
| **Cost (Lambda Labs A10G ~$0.75/hr)** | **~$1.25** | 1.67 hr × $0.75/hr |

> ⚠️ **HuggingFace access required** for Scout-17B — request at hf.co/meta-llama before running.

---

#### Config 5 — `config_a100_32b_production` (A100 40GB, DeepSeek-32B, 200 steps + evals)

| Dimension | Estimate | Notes |
|---|---|---|
| **Platform** | ⚠️ Colab Pro+ (not guaranteed) — RunPod/Paperspace more reliable |
| **GPU VRAM** | ~25–30 GB peak | A100 40GB — good headroom; Unsloth Qwen kernels reduce peak ~15% |
| **System RAM** | ~20–26 GB | Paged optimizer for 32B ~16 GB to CPU RAM; Colab A100 provides ~83 GB |
| **Google Drive storage** | ~10–14 GB | 3 checkpoints (~3.5 GB each) + final LoRA (~4.5 GB) |
| **Time** | ~105–130 min (~2 hrs) | ~20 sec/step with Unsloth on A100, 10 eval runs × ~3 min, ~10 min load |
| **Cost (Colab Pro+ A100 ~$1.40/hr)** | **~$2.80–3.00** | Plus $49.99/mo subscription |
| **Cost (RunPod A100 40GB ~$1.49/hr)** | **~$2.80** | 1.88 hr × $1.49/hr — no subscription needed |
| **Cost (Paperspace A100 ~$1.15/hr)** | **~$2.16** | 1.88 hr × $1.15/hr |

---

#### Config 6 — `config_a100_80gb_fullblown` (A100 80GB, DeepSeek-32B, 300 steps, 8K context)

| Dimension | Estimate | Notes |
|---|---|---|
| **Platform** | ❌ Not on Colab at any tier — use RunPod / Vast.ai / Lambda Labs |
| **GPU VRAM** | ~37–44 GB peak | batch=2 + 8192 context pushes usage; A100 80GB has ample room |
| **System RAM** | ~22–28 GB | Paged optimizer for 32B; external providers typically give 64–128 GB |
| **Google Drive storage** | ~18–24 GB | 5 checkpoints (~3.5 GB each) + final LoRA (~4.5 GB) |
| **Time** | ~190–225 min (~3.5 hrs) | ~25 sec/step at 8K context + batch=2, 15 eval runs × ~4 min, ~12 min load |
| **Cost (RunPod A100 80GB ~$2.00/hr)** | **~$7.00–7.50** | ~3.7 hr × $2.00/hr |
| **Cost (Vast.ai A100 80GB ~$1.50/hr)** | **~$5.50** | ~3.7 hr × $1.50/hr (spot pricing) |

---

### All-in-One Summary Table

| Config | GPU | Time | Peak VRAM | System RAM | Drive storage | Cost estimate | Platform |
|---|---|---|---|---|---|---|---|
| t4_train100_no_evals | T4 | ~20–25 min | ~9 GB | ~7 GB | ~3–4 GB | **~$0.08** | Colab PAYG ✅ |
| t4_train20_full_evals | T4 | ~10–15 min | ~11 GB | ~7 GB | ~3–4 GB | **~$0.05** | Colab PAYG ✅ |
| a10g_14b_train100 | A10G | ~45–55 min | ~15 GB | ~12 GB | ~6 GB | **~$0.60–0.70** | RunPod/Lambda ⚠️ |
| a10g_17b_scout_full | A10G | ~85–100 min | ~18 GB | ~14 GB | ~7 GB | **~$1.10–1.25** | RunPod/Lambda ⚠️ |
| a100_32b_production | A100 40GB | ~110–130 min | ~28 GB | ~23 GB | ~13 GB | **~$2.20–3.00** | RunPod/Paperspace ⚠️ |
| a100_80gb_fullblown | A100 80GB | ~190–225 min | ~42 GB | ~25 GB | ~21 GB | **~$5.50–7.50** | RunPod/Vast.ai ❌Colab |

> All costs are estimates based on small dataset (<5K examples). Times include model download/load, training, evals, and saving. Costs are for a single run — resume runs from checkpoint are proportionally cheaper.

---

### Google Drive Space Planning

With 15 GB free Drive quota, here is what fits:

| Scenario | Drive usage | Fits in 15 GB free? |
|---|---|---|
| T4 configs (#1 + #2) | ~3–4 GB | ✅ Yes — comfortably |
| A10G configs (#3 + #4) | ~6–8 GB | ✅ Yes — if not mixing with T4 outputs |
| A100 40GB config (#5) | ~13 GB | ⚠️ Tight — clear old checkpoints first |
| A100 80GB config (#6) | ~21 GB | ❌ Exceeds 15 GB — upgrade to 100 GB Drive ($2.99/mo) |


---

## Files Overview

```
train_unsloth_config.py               ← the single training script
config_t4_train100_no_evals.yaml      ← T4, 100 steps, no evals
config_t4_train20_full_evals.yaml     ← T4, 20 steps, full evals
config_a10g_14b_train100.yaml         ← A10G, Qwen-14B, 100 steps
config_a10g_17b_scout_full.yaml       ← A10G, Scout-17B, 120 steps
config_a100_32b_production.yaml       ← A100-40GB, DeepSeek-32B, 200 steps
config_a100_80gb_fullblown.yaml       ← A100-80GB, DeepSeek-32B, 300 steps, max settings
config_readme.md                      ← this file
```
