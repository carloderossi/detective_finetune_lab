import os
import re
import shutil
from pathlib import Path

RAW_DIR = Path(r"C:\Carlo\projects\detective_finetune_lab\data\raw")
CLEAN_DIR = Path(r"C:\Carlo\projects\detective_finetune_lab\data\cleaned")

START_MARKER = "*** START OF THE PROJECT GUTENBERG"
END_MARKER = "*** END OF THE PROJECT GUTENBERG"

# ------------------------------------------------------------
# STEP 1 — Clean Gutenberg header/footer but KEEP CONTENTS
# ------------------------------------------------------------

def clean_gutenberg_keep_contents(raw_text: str) -> str:
    """Return text between START and END markers, preserving CONTENTS."""
    start_idx = raw_text.find(START_MARKER)
    if start_idx != -1:
        # Keep everything AFTER the START marker line
        raw_text = raw_text[start_idx:].split("\n", 1)[1]

    end_idx = raw_text.find(END_MARKER)
    if end_idx != -1:
        raw_text = raw_text[:end_idx]

    return raw_text.strip()


# ------------------------------------------------------------
# STEP 2 — Extract CONTENTS block and chapter titles
# ------------------------------------------------------------

def extract_contents(text: str):
    """Extract CONTENTS block and chapter titles from cleaned text."""
    m = re.search(r"\bcontents\b", text, flags=re.IGNORECASE)
    if not m:
        return None, []

    contents_start = m.end()
    after = text[contents_start:]

    lines = after.splitlines()
    contents_lines = []
    started = False

    for line in lines:
        stripped = line.strip()

        # Skip initial blank lines after "CONTENTS"
        if not started and stripped == "":
            continue

        # Once we hit the first non-empty line, we are inside the contents block
        if stripped != "":
            started = True
            contents_lines.append(line)
            continue

        # If we hit a blank line *after* collecting content, stop
        if started and stripped == "":
            break

    contents_text = "\n".join(contents_lines)

    # Extract chapter titles
    chapter_titles = []
    for line in contents_lines:
        stripped = line.strip()
        if not stripped:
            continue

        cleaned = re.sub(
            r"^(PART|CHAPTER)?\s*[IVXLC0-9]+\.*\s*",
            "",
            stripped,
            flags=re.IGNORECASE
        ).strip()

        if cleaned:
            chapter_titles.append(cleaned)

    return contents_text, chapter_titles


# ------------------------------------------------------------
# STEP 3 — Find chapter boundaries
# ------------------------------------------------------------

def find_chapter_positions(text: str, chapter_titles):
    positions = []
    for title in chapter_titles:
        pattern = re.escape(title)
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            positions.append((title, m.start()))
    return positions


def split_into_chapters(text: str, positions):
    chapters = []
    positions = sorted(positions, key=lambda x: x[1])

    for i, (title, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        chapters.append(text[start:end].strip())

    return chapters


# ------------------------------------------------------------
# STEP 4 — Main processing
# ------------------------------------------------------------

def process_books():
    # Clean output folder
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for author_folder in RAW_DIR.iterdir():
        if not author_folder.is_dir():
            continue

        print("=" * 60)
        print(f"Processing {author_folder.name.upper()} books")
        print("=" * 60)
        print()

        for book_file in author_folder.glob("*.txt"):
            print(f"Processing: {book_file}")

            raw_text = book_file.read_text(encoding="utf-8", errors="ignore")

            # Clean Gutenberg but KEEP CONTENTS
            text = clean_gutenberg_keep_contents(raw_text)

            # Extract contents from cleaned text
            contents_text, chapter_titles = extract_contents(text)

            if not chapter_titles:
                print("  No CONTENTS found, skipping.\n")
                continue

            print("CONTENTS\n")
            print(contents_text)
            print()

            # Find chapter boundaries
            positions = find_chapter_positions(text, chapter_titles)
            chapters = split_into_chapters(text, positions)

            print(f"Found {len(chapters)} chapters")

            # Prepare output directory
            book_name = book_file.stem
            out_dir = CLEAN_DIR / author_folder.name / book_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save chapters
            for i, chapter_text in enumerate(chapters, start=1):
                out_path = out_dir / f"chapter_{i:02d}.txt"
                out_path.write_text(chapter_text, encoding="utf-8")
                print(f"  Saved: chapter_{i:02d}.txt ({len(chapter_text)} chars)")

            print()


if __name__ == "__main__":
    process_books()