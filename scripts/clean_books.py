import os
import re
import shutil
import json
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat" #"http://localhost:11434/api/generate" cannot support 500KB txt
MODEL = "qwen2.5:7b-instruct"   # e.g. "llama3", "mistral", etc.


# ---------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------
def ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]


# ---------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------
START_MARKER = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
END_MARKER   = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"


def strip_gutenberg(text: str) -> str:
    """Remove Gutenberg header/footer based on markers."""
    start = re.search(START_MARKER, text)
    end = re.search(END_MARKER, text)

    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace but preserve paragraph breaks."""
    # Convert Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple blank lines to a single blank line
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Strip trailing spaces
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


# ---------------------------------------------------------
# Chapter splitting via Ollama
# ---------------------------------------------------------
def split_into_chapters(text: str) -> list[str]:
    """
    Ask the model to split the text into chapters.
    Returns a list of chapter strings.
    """
    prompt = f"""
You are given the text of a public-domain detective novel.
Split it into chapters. Return ONLY a JSON list of chapters.
Each list element must contain the full text of one chapter.

Text:
{text}
"""

    response = ollama_generate(prompt)

    try:
        chapters = json.loads(response)
        assert isinstance(chapters, list)
        return chapters
    except Exception:
        print("Model did not return valid JSON. Saving whole text as one chapter.")
        return [text]


# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")


def clean_output_folder():
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def measure_size(text: str) -> int:
    return len(text.encode("utf-8"))

def process_books():
    for author_dir in RAW_DIR.iterdir():
        if not author_dir.is_dir():
            continue

        author = author_dir.name

        for book_file in author_dir.glob("*.txt"):
            book_name = book_file.stem

            print(f"Processing: {author}/{book_name}")

            # Read raw text
            raw_text = book_file.read_text(encoding="utf-8", errors="ignore")

            # Clean
            cleaned = strip_gutenberg(raw_text)
            cleaned = normalize_whitespace(cleaned)

            size_bytes = measure_size(cleaned)
            print(f'Book {author}/{book_name} Cleaned text size: {size_bytes / 1024:.2f} KB')
            continue

            # Split into chapters
            chapters = split_into_chapters(cleaned)

            # Save
            out_dir = CLEAN_DIR / author / book_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, chapter in enumerate(chapters, start=1):
                out_path = out_dir / f"chapter_{i}.txt"
                out_path.write_text(chapter, encoding="utf-8")

            print(f"Saved {len(chapters)} chapters.")


if __name__ == "__main__":
    clean_output_folder()
    process_books()
    print("Done.")