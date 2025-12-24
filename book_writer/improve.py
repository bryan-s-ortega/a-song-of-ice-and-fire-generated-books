import os
import sys
import argparse
import glob
import re
from dotenv import load_dotenv
from book_writer.write import load_llama3_model, generate_text

def get_chapter_number(filename):
    """Extracts chapter number from filename 'Chapter_X.txt'."""
    match = re.search(r"Chapter_(\d+)\.txt", filename)
    if match:
        return int(match.group(1))
    return 0

def parse_reviews(review_file):
    """
    Parses the markdown review file to map Chapter Number -> Review Text.
    Assumes format:
    ## Chapter X
    ... review content ...
    ---
    """
    reviews = {}
    if not os.path.exists(review_file):
        print(f"Warning: Review file '{review_file}' not found.")
        return reviews

    with open(review_file, "r") as f:
        content = f.read()

    # Split by chapters
    # Regex to find "## Chapter <num>" sections
    # We use a pattern that captures the number and the content until the next chapter or end
    pattern = re.compile(r"## Chapter (\d+)(.*?)(?=\n## Chapter \d+|\Z)", re.DOTALL)
    
    matches = pattern.findall(content)
    for chapter_num, review_text in matches:
        reviews[int(chapter_num)] = review_text.strip()
        
    return reviews

def improve_chapter(chapter_path, review_text, tokenizer, model, author="George R.R. Martin", title="The Winds of Winter"):
    """Reads a chapter, applies the review, and generates an improved version."""
    with open(chapter_path, "r") as f:
        content = f.read()

    # Extract prompt if present; we want to keep it for the new file
    prompt_match = re.search(r"PROMPT: (.*?)\n\n", content, re.DOTALL)
    chapter_prompt = prompt_match.group(1) if prompt_match else "Unknown Prompt"
    
    # Original text without prompt line
    original_text = re.sub(r"PROMPT: .*?\n\n", "", content, count=1, flags=re.DOTALL)

    # Construct the improvement prompt
    improvement_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are {author}. You are rewriting a chapter for '{title}' based on editorial feedback.
Your goal is to IMPROVE the chapter while keeping the core plot points, unless the review specifically asks to change them.
Enhance the prose, dialogue, and immersion.

Original Chapter Prompt: {chapter_prompt}

Editorial Review/Feedback:
{review_text}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the ORIGINAL Chapter draft:
{original_text}

Please rewrite the chapter to address the feedback. Output ONLY the new chapter text.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    print(f"Generating improved version for {os.path.basename(chapter_path)}...")
    # Increase max tokens as we are rewriting a full chapter
    improved_text = generate_text(improvement_prompt, tokenizer, model, max_new_tokens=2500)
    
    return f"PROMPT: {chapter_prompt}\n\n{improved_text}"

def main():
    parser = argparse.ArgumentParser(description="Improve book chapters based on reviews.")
    parser.add_argument("--book_dir", type=str, required=True, help="Directory containing the original book chapters")
    parser.add_argument("--review_file", type=str, required=True, help="Path to the markdown review file")
    parser.add_argument("--author", type=str, default="George R.R. Martin", help="Author name to emulate")
    parser.add_argument("--title", type=str, default="The Winds of Winter", help="Book title")
    args = parser.parse_args()

    load_dotenv()
    
    book_name = os.path.basename(os.path.normpath(args.book_dir))
    output_dir = f"{book_name}_improved"
    
    if not os.path.exists(args.book_dir):
        print(f"Error: Directory '{args.book_dir}' not found.")
        return

    # Parse reviews
    reviews = parse_reviews(args.review_file)
    if not reviews:
        print("No reviews found or file is empty. Exiting.")
        return

    print(f"Loaded reviews for {len(reviews)} chapters.")

    # Load Model
    print("Loading Llama 3 model for improvement...")
    tokenizer, model = load_llama3_model()
    print("Model loaded.")

    os.makedirs(output_dir, exist_ok=True)

    # Get all chapters
    files = glob.glob(os.path.join(args.book_dir, "Chapter_*.txt"))
    files.sort(key=get_chapter_number)

    print(f"Found {len(files)} chapters to process.")

    for file_path in files:
        chapter_num = get_chapter_number(os.path.basename(file_path))
        
        if chapter_num not in reviews:
            print(f"Skipping Chapter {chapter_num} (No review found).")
            # Option: Copy original file? Let's just skip for now, or maybe copy it so we have a full book.
            # Better to copy it so the output dir is a complete book.
            with open(file_path, 'r') as src, open(os.path.join(output_dir, os.path.basename(file_path)), 'w') as dst:
                dst.write(src.read())
            continue

        print(f"Improving Chapter {chapter_num}...")
        improved_content = improve_chapter(file_path, reviews[chapter_num], tokenizer, model, author=args.author, title=args.title)
        
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_path, "w") as f:
            f.write(improved_content)
        
        print(f"Saved improved chapter to {output_path}")

    print(f"All improvements completed. Saved to {output_dir}/")

if __name__ == "__main__":
    main()
