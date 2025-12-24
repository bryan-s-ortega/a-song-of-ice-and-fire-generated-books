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

def review_chapter(chapter_path, tokenizer, model, series="A Song of Ice and Fire", title="The Winds of Winter"):
    """Reads a chapter and generates a review."""
    with open(chapter_path, "r") as f:
        content = f.read()

    # Extract prompt if present (assumes format "PROMPT: ...\n\n")
    prompt_match = re.search(r"PROMPT: (.*?)\n\n", content, re.DOTALL)
    chapter_prompt = prompt_match.group(1) if prompt_match else "Unknown Prompt"
    
    # Remove the prompt line from the content to save tokens/confusion
    chapter_text = re.sub(r"PROMPT: .*?\n\n", "", content, count=1, flags=re.DOTALL)
    
    # Truncate if too long (simple safety, though 4bit model might handle context well)
    # Llama 3 8B has 8k context, so we should be careful.
    # Let's keep first 6000 chars approx.
    if len(chapter_text) > 12000:
        chapter_text = chapter_text[:12000] + "\n[...Trimmed for length...]"

    review_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a senior editor and expert on '{series}'.
Your task is to review a newly written chapter for '{title}'.
Evaluate the chapter based on:
1.  **Consistency**: Does it fit with previous books and character voices?
2.  **Plot Logic**: Does the sequence of events make sense?
3.  **Tone**: Is it gritty, immersive, and detailed enough (GRRM style)?
4.  **Improvements**: Briefly suggest 1-2 major improvements.

Be constructive but critical. Keep the review concise (bullet points).

<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the chapter outline/prompt: {chapter_prompt}

Here is the chapter text:
{chapter_text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    print(f"Generating review for {os.path.basename(chapter_path)}...")
    review = generate_text(review_prompt, tokenizer, model, max_new_tokens=512)
    return review

def main(args):
    load_dotenv()
    
    book_dir = args.book_dir
    book_name = os.path.basename(os.path.normpath(book_dir))
    output_file = f"reviews/{book_name}_review.md"
    
    if not os.path.exists(book_dir):
        print(f"Error: Directory '{book_dir}' not found.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("Loading Llama 3 model for review...")
    tokenizer, model = load_llama3_model()
    print("Model loaded.")

    # Get all chapters and sort them numerically
    files = glob.glob(os.path.join(book_dir, "Chapter_*.txt"))
    files.sort(key=get_chapter_number)

    print(f"Found {len(files)} chapters to review.")

    with open(output_file, "w") as f:
        f.write(f"# Book Review: {book_dir}\n\n")

    for file_path in files:
        chapter_num = get_chapter_number(os.path.basename(file_path))
        print(f"Reviewing Chapter {chapter_num}...")
        
        review = review_chapter(file_path, tokenizer, model, series=args.series, title=args.title)
        
        # Append to report immediately (so we have partial results if it crashes)
        with open(output_file, "a") as f:
            f.write(f"## Chapter {chapter_num}\n\n")
            f.write(review)
            f.write("\n\n---\n\n")
            f.flush()
            
    print(f"All reviews completed. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review generated book chapters.")
    parser.add_argument("--dir", dest="book_dir", type=str, required=True, help="Directory containing the book chapters")
    parser.add_argument("--series", type=str, default="A Song of Ice and Fire", help="Series name for context")
    parser.add_argument("--title", type=str, default="The Winds of Winter", help="Book title for context")
    args = parser.parse_args()
    
    main(args)
