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

def summarize_chapter(chapter_path, tokenizer, model):
    """Reads a chapter and generates a summary."""
    with open(chapter_path, "r") as f:
        content = f.read()

    # Remove the prompt line (PROMPT: ...) to just get the story text
    chapter_text = re.sub(r"PROMPT: .*?\n\n", "", content, count=1, flags=re.DOTALL)
    
    # Truncate if extremely long to avoid OOM, though 12k chars is usually fine for 8B model
    if len(chapter_text) > 20000:
        chapter_text = chapter_text[:20000] + "\n[...Trimmed...]"

    summary_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert literary summarizer.
Your task is to write a concise plot summary for the following chapter of 'The Winds of Winter'.
Focus on the key events, character actions, and plot progressions.
Do not include analysis or review, just the story summary.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Chapter Text:
{chapter_text}

Write a summary of this chapter (approx 200-300 words).
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    print(f"Generating summary for {os.path.basename(chapter_path)}...")
    summary = generate_text(summary_prompt, tokenizer, model, max_new_tokens=400)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Summarize book chapters.")
    parser.add_argument("--book_dir", type=str, required=True, help="Directory containing the book chapters")
    args = parser.parse_args()

    load_dotenv()
    
    book_dir = args.book_dir
    if not os.path.exists(book_dir):
        print(f"Error: Directory '{book_dir}' not found.")
        return

    book_name = os.path.basename(os.path.normpath(book_dir))
    output_dir = "summaries"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{book_name}_summary.md")

    print(f"Summarizing book: {book_name}")
    print("Loading Llama 3 model...")
    tokenizer, model = load_llama3_model()
    print("Model loaded.")

    # Get all chapters
    files = glob.glob(os.path.join(book_dir, "Chapter_*.txt"))
    if not files:
        print("No Chapter_*.txt files found.")
        return
        
    files.sort(key=get_chapter_number)

    print(f"Found {len(files)} chapters.")

    full_book_summary = []

    with open(output_file, "w") as f:
        f.write(f"# Book Summary: {book_name}\n\n")

    for file_path in files:
        chapter_num = get_chapter_number(os.path.basename(file_path))
        print(f"Processing Chapter {chapter_num}...")
        
        summary = summarize_chapter(file_path, tokenizer, model)
        full_book_summary.append(f"## Chapter {chapter_num}\n\n{summary}")
        
        # Append immediately to file
        with open(output_file, "a") as f:
            f.write(f"## Chapter {chapter_num}\n\n")
            f.write(summary)
            f.write("\n\n---\n\n")
            
    # Optional: Generate Overall Summary (can be done by summarizing the summaries)
    # For now, let's keep it chapter-by-chapter as requested, but maybe add an "Overall" section at top if we want later.
    
    print(f"Summary generation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
