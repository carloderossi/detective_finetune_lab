import os
import re
import shutil
from pathlib import Path

RAW_DIR = Path(r"C:\Carlo\projects\detective_finetune_lab\data\raw\poirot")
CLEAN_BASE = Path(r"C:\Carlo\projects\detective_finetune_lab\data\cleaned\agatha_christie")

BOOKS = {
    "Poirot Investigates by Agatha Christie.txt": "poirot investigates",
    "The mystery of the Blue Train by Agatha Christie.txt": "the mystery of the blue train",
}

START_MARKER = "*** START OF THE PROJECT GUTENBERG"
END_MARKER = "*** END OF THE PROJECT GUTENBERG"


# ------------------------------------------------------------
# Clean Gutenberg markers
# ------------------------------------------------------------
def clean_gutenberg(text: str) -> str:
    start = text.find(START_MARKER)
    if start != -1:
        text = text[start:].split("\n", 1)[1]

    end = text.find(END_MARKER)
    if end != -1:
        text = text[:end]

    return text.strip()


# ------------------------------------------------------------
# Extract CONTENTS block (robust for both books)
# ------------------------------------------------------------
def extract_contents_block(text: str):
    """
    Extract CONTENTS block allowing blank lines between entries.
    Stop only when we hit a line that is clearly NOT part of contents.
    """
    m = re.search(r"\bcontents\b", text, flags=re.IGNORECASE)
    if not m:
        return []

    after = text[m.end():]
    lines = after.splitlines()

    contents = []
    started = False
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        # Skip initial blank lines after "CONTENTS"
        if not started and stripped == "":
            continue

        # Detect end of contents:
        # - 3 consecutive blank lines
        # - OR a heading like "POIROT INVESTIGATES" or "CAST OF CHARACTERS"
        if started:
            if stripped == "":
                blank_count += 1
                if blank_count >= 3:
                    break
                # allow blank lines inside contents
                contents.append("")
                continue
            else:
                blank_count = 0

            # Stop when we hit a non‑chapter heading
            if re.match(r"^[A-Z][A-Z\s]{5,}$", stripped):
                break

        # Accept chapter/story entries
        if re.match(r"^[IVXLC]+\s+.+$", stripped):          # Roman numerals
            contents.append(stripped)
            started = True
            continue

        if re.match(r"^\d+\.\s+.+$", stripped):             # Numbered chapters
            contents.append(stripped)
            started = True
            continue

        # If we already started, accept any non-empty line
        if started and stripped != "":
            contents.append(stripped)
            continue

        # If we haven't started yet, skip noise
        if not started:
            continue

    # Remove empty lines
    return [c for c in contents if c.strip() != ""]

# ------------------------------------------------------------
# Parse chapter titles from CONTENTS
# ------------------------------------------------------------
def parse_chapter_titles(contents_lines):
    titles = []

    for line in contents_lines:
        # Poirot Investigates: "I The Adventure of …"
        m = re.match(r"^[IVXLC]+\s+(.*)$", line)
        if m:
            titles.append(m.group(1).strip())
            continue

        # Blue Train: "1. The Man with the White Hair"
        m = re.match(r"^\d+\.\s+(.*)$", line)
        if m:
            titles.append(m.group(1).strip())
            continue

        # Fallback: keep line as-is
        titles.append(line.strip())

    return titles


# ------------------------------------------------------------
# Find chapter positions
# ------------------------------------------------------------
def find_positions(text, titles):
    positions = []
    for t in titles:
        pattern = re.escape(t)
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            positions.append((t, m.start()))
    return sorted(positions, key=lambda x: x[1])


# ------------------------------------------------------------
# Split into chapters
# ------------------------------------------------------------
def split_chapters(text, positions):
    chapters = []
    for i, (_, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        chapters.append(text[start:end].strip())
    return chapters


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def process_poirot_books():

    # Clean ONLY the two target folders
    # for folder in BOOKS.values():
    #     out_dir = CLEAN_BASE / folder
    #     if out_dir.exists():
    #         shutil.rmtree(out_dir)

    print("=" * 60)
    print("Processing AGATHA CHRISTIE books")
    print("=" * 60)
    print()

    for filename, folder_name in BOOKS.items():
        book_path = RAW_DIR / filename
        print(f"Processing: {book_path}")

        raw = book_path.read_text(encoding="utf-8", errors="ignore")
        text = clean_gutenberg(raw)

        contents_lines = extract_contents_block(text)
        if not contents_lines:
            print("  No CONTENTS found, skipping.\n")
            continue

        print("CONTENTS\n")
        for line in contents_lines:
            print(" ", line)
        print()

        titles = parse_chapter_titles(contents_lines)
        positions = find_positions(text, titles)
        chapters = split_chapters(text, positions)

        print(f"Found {len(chapters)} chapters")

        out_dir = CLEAN_BASE / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, chapter in enumerate(chapters, start=1):
            out_file = out_dir / f"chapter_{i:02d}.txt"
            out_file.write_text(chapter, encoding="utf-8")
            print(f"  Saved: chapter_{i:02d}.txt ({len(chapter)} chars)")

        print()


if __name__ == "__main__":
    process_poirot_books()