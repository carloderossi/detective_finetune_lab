# scripts/generate_deduction_prompts.py
import os
import json
from pathlib import Path

OUT_PATH = Path("data/rl/deduction_prompts.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

holmes_templates = [
    "Holmes explains to Watson how he deduced the identity of the thief.",
    "Holmes reconstructs the sequence of events leading to the crime.",
    "Holmes reveals the crucial clue everyone else overlooked.",
    "Holmes explains how he inferred the suspect's occupation from small details.",
    "Holmes describes the chain of reasoning that led him to the murderer.",
    "Holmes explains why the alibi could not possibly be true.",
    "Holmes reveals the hidden motive behind the crime.",
    "Holmes explains how he deduced the location of the missing object.",
    "Holmes walks Watson through the logical steps of his deduction.",
    "Holmes explains how he identified the culprit from footprints alone.",
]

poirot_templates = [
    "Poirot reveals the murderer in the drawing room.",
    "Poirot explains the psychological motive behind the crime.",
    "Poirot describes how he used order and method to solve the case.",
    "Poirot explains how he uncovered the hidden relationship between the suspects.",
    "Poirot reveals the flaw in the suspect's carefully constructed alibi.",
    "Poirot explains how a small detail led him to the truth.",
    "Poirot reconstructs the events leading to the murder.",
    "Poirot explains how he deduced the identity of the blackmailer.",
    "Poirot reveals the significance of a seemingly trivial clue.",
    "Poirot explains how he solved the case using psychology rather than evidence.",
]

generic_templates = [
    "The detective explains the chain of reasoning that reveals the culprit.",
    "The detective describes how a minor detail exposed the murderer.",
    "The detective explains how the suspect's alibi was disproved.",
    "The detective reconstructs the events leading up to the crime.",
    "The detective reveals the hidden motive behind the crime.",
    "The detective explains how the crucial clue was discovered.",
    "The detective describes how conflicting testimonies led to the truth.",
    "The detective explains how the crime scene evidence revealed the culprit.",
    "The detective reveals how a small inconsistency exposed the lie.",
    "The detective explains how the pattern of clues pointed to the murderer.",
]

def expand_templates(templates, n_per_template):
    prompts = []
    for t in templates:
        for i in range(n_per_template):
            prompts.append({"prompt": t})
    return prompts

if __name__ == "__main__":
    # 200 prompts: 80 Holmes, 80 Poirot, 40 generic
    holmes_prompts  = expand_templates(holmes_templates, 8)   # 10 * 8 = 80
    poirot_prompts  = expand_templates(poirot_templates, 8)   # 10 * 8 = 80
    generic_prompts = expand_templates(generic_templates, 4)  # 10 * 4 = 40

    all_prompts = holmes_prompts + poirot_prompts + generic_prompts

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for obj in all_prompts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_prompts)} prompts to {OUT_PATH}")