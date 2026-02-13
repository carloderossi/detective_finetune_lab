import json
from pathlib import Path

INPUT_ROOT = Path("data/cleaned")
OUTPUT_FILE = Path("data/jsonl/detective_finetune.jsonl")

def collect_chapters():
    entries = []
    for author_dir in INPUT_ROOT.iterdir():
        if not author_dir.is_dir():
            continue

        author = author_dir.name

        for book_dir in author_dir.iterdir():
            if not book_dir.is_dir():
                continue

            book = book_dir.name

            for chapter_file in sorted(book_dir.glob("chapter_*.txt")):
                with open(chapter_file, "r", encoding="utf-8") as f:
                    text = f.read()

                # --- NEW: remove carriage returns ---
                text = text.replace("\r", "")

                # Normalize whitespace (optional but recommended)
                text = text.strip()

                entries.append({
                    "author": author,
                    "book": book,
                    "chapter": chapter_file.stem,
                    "text": text
                })

    return entries

def write_jsonl(entries):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    entries = collect_chapters()
    write_jsonl(entries)
    print(f"Wrote {len(entries)} chapters to {OUTPUT_FILE}")