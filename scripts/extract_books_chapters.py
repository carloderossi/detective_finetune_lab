import os
import re
from pathlib import Path

def extract_content(text):
    """Extract content between START and END markers."""
    # Find the START marker
    start_pattern = r'\*\*\* START OF (THE |THIS )?PROJECT GUTENBERG EBOOK[^\*]*\*\*\*'
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    
    # Find the END marker
    end_pattern = r'\*\*\* END OF (THE |THIS )?PROJECT GUTENBERG EBOOK[^\*]*\*\*\*'
    end_match = re.search(end_pattern, text, re.IGNORECASE)
    
    if start_match and end_match:
        # Extract content between markers
        content = text[start_match.end():end_match.start()]
        return content.strip()
    else:
        print("Warning: Could not find START or END markers")
        return text

def detect_chapter_pattern(text):
    """Detect which chapter pattern is used in the text."""
    patterns = [
        # Numbered chapters with all-caps titles (Agatha Christie style)
        (r'^\s*\d+\.\s+[A-Z][A-Z\s]+$', 'numbered_caps'),
        # Roman numerals with all-caps titles
        (r'^\s*[IVXLCDM]+\.\s+[A-Z][A-Z\s]+$', 'roman_caps'),
        # Standard CHAPTER formats
        (r'^CHAPTER [IVXLCDM]+\.?\s*$', 'chapter_roman'),
        (r'^CHAPTER \d+\.?\s*$', 'chapter_num'),
        (r'^Chapter [IVXLCDM]+\.?\s*$', 'chapter_roman_mixed'),
        (r'^Chapter \d+\.?\s*$', 'chapter_num_mixed'),
        # CHAPTER with title on same line
        (r'^CHAPTER [IVXLCDM]+[\.:]?\s+[A-Z]', 'chapter_roman_title'),
        (r'^CHAPTER \d+[\.:]?\s+[A-Z]', 'chapter_num_title'),
        # Adventure/Story/Part formats
        (r'^(?:ADVENTURE|STORY|PART) [IVXLCDM]+', 'adventure_roman'),
        (r'^(?:Adventure|Story|Part) \d+', 'adventure_num'),
    ]
    
    for pattern, pattern_name in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) >= 3:  # At least 3 chapters found
            return pattern, pattern_name
    
    # Check for Roman numerals alone on a line (Sherlock Holmes style)
    # where title follows on next line
    lines = text.split('\n')
    roman_alone_count = 0
    for i, line in enumerate(lines):
        if re.match(r'^\s*[IVXLCDM]+\s*$', line.strip()):
            # Check if next non-empty line looks like a title
            for j in range(i+1, min(i+3, len(lines))):
                next_line = lines[j].strip()
                if next_line and re.match(r'^[A-Z\s]+$', next_line) and len(next_line) > 10:
                    roman_alone_count += 1
                    break
    
    if roman_alone_count >= 3:
        return r'^\s*[IVXLCDM]+\s*$', 'roman_alone'
    
    return None, None

def split_into_chapters(text, book_name):
    """Split text into chapters based on detected pattern."""
    # Detect chapter pattern
    pattern, pattern_name = detect_chapter_pattern(text)
    
    if not pattern:
        print(f"Warning: Could not detect chapter pattern for {book_name}")
        # Try a more flexible pattern as fallback
        pattern = r'^(?:CHAPTER|Chapter|ADVENTURE|Adventure|STORY|Story|PART|Part)\s+[IVXLCDM\d]+.*$'
        pattern_name = 'fallback'
    
    print(f"Using pattern: {pattern_name} -> {pattern}")
    
    # Special handling for roman_alone pattern (Roman numeral on separate line from title)
    if pattern_name == 'roman_alone':
        return split_chapters_roman_alone(text)
    
    # Split by chapter markers
    lines = text.split('\n')
    chapters = []
    current_chapter = []
    chapter_count = 0
    in_content = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check if this line is a chapter marker
        if re.match(pattern, line_stripped):
            # Save previous chapter if it exists and has content
            if current_chapter:
                chapter_text = '\n'.join(current_chapter).strip()
                if len(chapter_text) > 100:  # Only save if chapter has substantial content
                    chapters.append(chapter_text)
                    chapter_count += 1
            
            # Start new chapter with the chapter heading
            current_chapter = [line]
            in_content = True
        elif in_content:
            current_chapter.append(line)
    
    # Add the last chapter
    if current_chapter:
        chapter_text = '\n'.join(current_chapter).strip()
        if len(chapter_text) > 100:
            chapters.append(chapter_text)
            chapter_count += 1
    
    print(f"Found {len(chapters)} chapters")
    
    # If we didn't find any chapters with the pattern, try a different approach
    if len(chapters) == 0:
        print("No chapters found with pattern matching. Trying alternative approach...")
        chapters = split_chapters_alternative(text, book_name)
    
    return chapters

def split_chapters_roman_alone(text):
    """Split chapters when Roman numerals are on separate lines from titles."""
    lines = text.split('\n')
    chapters = []
    chapter_starts = []
    
    # Find all chapter start positions
    for i, line in enumerate(lines):
        if re.match(r'^\s*[IVXLCDM]+\s*$', line.strip()):
            # Verify next non-empty line looks like a title
            for j in range(i+1, min(i+3, len(lines))):
                next_line = lines[j].strip()
                if next_line and re.match(r'^[A-Z\s]+$', next_line) and len(next_line) > 10:
                    chapter_starts.append(i)
                    break
    
    # Split text at chapter boundaries
    for idx, start in enumerate(chapter_starts):
        end = chapter_starts[idx + 1] if idx + 1 < len(chapter_starts) else len(lines)
        chapter_lines = lines[start:end]
        chapter_text = '\n'.join(chapter_lines).strip()
        if len(chapter_text) > 100:
            chapters.append(chapter_text)
    
    print(f"Found {len(chapters)} chapters using roman_alone method")
    return chapters

def split_chapters_alternative(text, book_name):
    """Alternative method to split chapters by looking for common patterns."""
    # Try to find repeated patterns of chapter markers
    # Look for lines that are: short, mostly caps, start with number or roman numeral
    lines = text.split('\n')
    potential_markers = []
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Look for short lines that could be chapter headings
        if len(line_stripped) < 100 and len(line_stripped) > 3:
            # Check if it matches common chapter patterns
            if (re.match(r'^\d+\.?\s+[A-Z]', line_stripped) or 
                re.match(r'^[IVXLCDM]+\.?\s+[A-Z]', line_stripped) or
                re.match(r'^(?:CHAPTER|Chapter)\s+[\dIVXLCDM]+', line_stripped)):
                potential_markers.append((i, line_stripped))
    
    if len(potential_markers) < 3:
        print(f"Could not split {book_name} into chapters")
        return []
    
    # Split based on these markers
    chapters = []
    for idx, (line_num, marker) in enumerate(potential_markers):
        start_line = line_num
        end_line = potential_markers[idx + 1][0] if idx + 1 < len(potential_markers) else len(lines)
        
        chapter_lines = lines[start_line:end_line]
        chapter_text = '\n'.join(chapter_lines).strip()
        if len(chapter_text) > 100:
            chapters.append(chapter_text)
    
    print(f"Found {len(chapters)} chapters using alternative method")
    return chapters

def get_book_name(filename):
    """Extract book name from filename."""
    # Remove .txt extension
    name = filename.replace('.txt', '')
    # Clean up the name
    name = name.replace('_', ' ').title()
    # Remove common suffixes
    name = re.sub(r'\s+\d+$', '', name)
    return name

def get_author_name(folder_name):
    """Map folder name to author name."""
    author_map = {
        'holmes': 'arthur_conan_doyle',
        'poirot': 'agatha_christie'
    }
    return author_map.get(folder_name.lower(), folder_name.lower())

def process_file(filepath, raw_base, cleaned_base):
    """Process a single book file."""
    print(f"\nProcessing: {filepath}")
    
    # Read the file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(filepath, 'r', encoding='latin-1') as f:
            text = f.read()
    
    # Extract content between markers
    content = extract_content(text)
    
    # Get book and author information
    relative_path = os.path.relpath(filepath, raw_base)
    parts = Path(relative_path).parts
    author_folder = parts[0]  # 'holmes' or 'poirot'
    filename = parts[-1]
    
    author_name = get_author_name(author_folder)
    book_name = get_book_name(filename)
    book_name_clean = book_name.lower().replace(' ', '_')
    
    # Split into chapters
    chapters = split_into_chapters(content, book_name)
    
    if not chapters:
        print(f"Warning: No chapters found in {filepath}")
        return
    
    # Create output directory
    output_dir = os.path.join(cleaned_base, author_name, book_name_clean)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each chapter
    for i, chapter in enumerate(chapters, 1):
        output_file = os.path.join(output_dir, f'chapter_{i:02d}.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chapter)
        print(f"  Saved: chapter_{i:02d}.txt ({len(chapter)} chars)")
    
    print(f"✓ Processed {len(chapters)} chapters from {book_name}")

def main():
    # Define base directories
    raw_base = r'C:\Carlo\projects\detective_finetune_lab\data\raw'
    cleaned_base = r'C:\Carlo\projects\detective_finetune_lab\data\cleaned'
    import shutil
    # Create cleaned directory if it doesn't exist
    if  Path(cleaned_base).exists():
        shutil.rmtree(Path(cleaned_base))
    Path(cleaned_base).mkdir(parents=True, exist_ok=True)
    #os.makedirs(cleaned_base, exist_ok=True)
    
    # Process both holmes and poirot folders
    for author_folder in ['holmes', 'poirot']:
        author_path = os.path.join(raw_base, author_folder)
        
        if not os.path.exists(author_path):
            print(f"Warning: Folder not found: {author_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {author_folder.upper()} books")
        print('='*60)
        
        # Process all .txt files in the folder
        for filename in os.listdir(author_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(author_path, filename)
                process_file(filepath, raw_base, cleaned_base)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print('='*60)

if __name__ == '__main__':
    main()