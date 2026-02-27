from datasets import load_dataset

DATA_FILE = "data/train/detective_finetune.jsonl"


def extract_text(example):
    # Format metadata into a readable string
    author = example.get("author", "Unknown").replace("_", " ").title()
    book = example.get("book", "Unknown").replace("_", " ").title()
    
    header = f"### Author: {author}\n### Source: {book}\n\n"
    
    # Priority check for fields
    if "text" in example:
        return header + example["text"]
    if "output" in example:
        return header + example["output"]
    
    raise ValueError(f"Found sample with no usable text. Keys: {example.keys()}")

print(f"Loading dataset from {DATA_FILE}...")
raw_dataset = load_dataset("json", data_files={"train": DATA_FILE})["train"]
print(f"Total examples: {len(raw_dataset)}")

dataset = raw_dataset.map(lambda x: {"text": extract_text(x)})

# # train / eval split
# dataset = dataset.train_test_split(test_size=0.05, seed=42)
# train_ds = dataset["train"]
# eval_ds = dataset["test"]


# train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
# eval_tok  = eval_ds.map(tokenize,  batched=True, remove_columns=["text"])


# print("Packing sequences...")
# train_tok = train_tok.map(
#     group_texts,
#     batched=True,
#     batch_size=None,
#     remove_columns=train_tok.column_names,
# )
# eval_tok = eval_tok.map(
#     group_texts,
#     batched=True,
#     batch_size=None,
#     remove_columns=eval_tok.column_names,
# )

# print(f"Packed train samples: {len(train_tok)}")
# print(f"Packed eval samples:  {len(eval_tok)}")