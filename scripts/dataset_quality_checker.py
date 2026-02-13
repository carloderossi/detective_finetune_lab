import os
import json
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_FILE = "data/jsonl/detective_finetune.jsonl"

# Use an open, ungated tokenizer
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Better embedding model for style drift
embedder = SentenceTransformer(
    "Alibaba-NLP/gte-large-en-v1.5",
# otherwise we get 
# "ValueError: Alibaba-NLP/new-impl You can inspect the repository content at https://hf.co/Alibaba-NLP/gte-large-en-v1.5.
# Please pass the argument `trust_remote_code=True` to allow custom code to be run."    
    trust_remote_code=True      
)

def load_dataset():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def check_empty(entries):
    empties = [e for e in entries if len(e.get("text", "")) < 10]
    for i, e in enumerate(empties):
        print(f"[EMPTY #{i}] author={e.get('author')} book={e.get('book')} chapter={e.get('chapter')}")
    return empties

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
    # Normalize author field
    def is_holmes(e):
        return e.get("author", "").lower() == "arthur_conan_doyle"

    def is_poirot(e):
        return e.get("author", "").lower() == "agatha_christie"

    holmes = [e["text"] for e in entries if is_holmes(e)]
    poirot = [e["text"] for e in entries if is_poirot(e)]

    if len(holmes) == 0 or len(poirot) == 0:
        return {
            "error": "Not enough samples",
            "holmes_count": len(holmes),
            "poirot_count": len(poirot),
            "drift": None
        }

    # Balance sample sizes and limit to 20 pairs
    n = min(len(holmes), len(poirot), 20)
    h_sample = holmes[:n]
    p_sample = poirot[:n]

    scores = []
    for h, p in zip(h_sample, p_sample):
        eh = embedder.encode(h, convert_to_tensor=True)
        ep = embedder.encode(p, convert_to_tensor=True)
        scores.append(util.cos_sim(eh, ep).item())

    return {
        "holmes_count": len(holmes),
        "poirot_count": len(poirot),
        "drift": sum(scores) / len(scores)
    }

if __name__ == "__main__":
    data = load_dataset()

    print("Empty entries:", len(check_empty(data)))
    print("Duplicate entries:", len(check_duplicates(data)))
    print("Average token length:", avg_token_length(data))
    drift = style_drift(data)
    print("Holmes–Poirot style drift:", drift)
