import os
import json
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

token = cfg["huggingface"]["token"]

DATA_FILE = "detective_finetune.jsonl"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
embedder = SentenceTransformer("all-mpnet-base-v2")

def load_dataset():
    with open(DATA_FILE, "r") as f:
        return [json.loads(line) for line in f]

def check_empty(entries):
    return [e for e in entries if len(e.get("text", "")) < 10]

def check_duplicates(entries):
    seen = set()
    dups = []
    for e in entries:
        t = e["text"]
        if t in seen:
            dups.append(t)
        seen.add(t)
    return dups

def avg_token_length(entries):
    lengths = [len(tokenizer.encode(e["text"])) for e in entries]
    return sum(lengths) / len(lengths)

def style_drift(entries):
    holmes = [e["text"] for e in entries if "Holmes" in e.get("source", "")]
    poirot = [e["text"] for e in entries if "Poirot" in e.get("source", "")]

    h_sample = holmes[:20]
    p_sample = poirot[:20]

    scores = []
    for h, p in zip(h_sample, p_sample):
        eh = embedder.encode(h, convert_to_tensor=True)
        ep = embedder.encode(p, convert_to_tensor=True)
        scores.append(util.cos_sim(eh, ep).item())

    return sum(scores) / len(scores)

if __name__ == "__main__":
    data = load_dataset()

    print("Empty entries:", len(check_empty(data)))
    print("Duplicate entries:", len(check_duplicates(data)))
    print("Average token length:", avg_token_length(data))
    print("Holmes–Poirot style drift score:", style_drift(data))