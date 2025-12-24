import os
import sys
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

def get_rag_context(query, db_path="./chroma_db", k=3):
    """Retrieve relevant context chunks from ChromaDB."""
    # Re-initialize embedding model here to ensure freshness or thread safety if needed
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    
    docs = vectorstore.similarity_search(query, k=k)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text

def load_llama3_model():
    """Load Llama 3 8B in 4-bit quantization."""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found in environment.")
        sys.exit(1)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_new_tokens=2000): # Increased tokens for a full chapter
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=0.8, # Slightly higher for creativity
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.1
    )
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate a book from an outline.")
    parser.add_argument("--outline", type=str, default="outline.txt", help="Path to the outline file")
    parser.add_argument("--output-dir", type=str, default="generated_book", help="Directory to save generated chapters")
    parser.add_argument("--author", type=str, default="George R.R. Martin", help="Name of the author to emulate")
    parser.add_argument("--title", type=str, default="The Winds of Winter", help="Title of the book being written")
    args = parser.parse_args()

    load_dotenv()
    
    # 1. Check for Outline
    if not os.path.exists(args.outline):
        print(f"Error: '{args.outline}' not found. Please create one with chapter prompts.")
        return

    # 2. Load Model
    print("Loading Llama 3 (4-bit)... This may take a minute.")
    tokenizer, model = load_llama3_model()
    print("Model loaded.")

    # 3. Create Output Directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 4. Read Outline
    with open(args.outline, "r") as f:
        chapters = [line.strip() for line in f if line.strip() and not line.strip().startswith(('*', '#'))]

    print(f"Found {len(chapters)} chapters to generate.")

    # 5. Generation Loop
    previous_chapter_summary = "None. This is the first chapter."

    for i, chapter_prompt in enumerate(chapters):
        chapter_num = i + 1
        print(f"\n--- Processing Chapter {chapter_num} ---")
        print(f"Prompt: {chapter_prompt}")

        # RAG Search
        print("Retrieving context from books...")
        book_context = get_rag_context(chapter_prompt)
        
        # prompt Construction (Includes Memory)
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are {args.author}. You are writing Chapter {chapter_num} of '{args.title}'.
Write a long, detailed, and immersive chapter based on the user's prompt.
Capture the gritty tone, complex dialogue, and sensory details of the series.

Key Context from previous books:
{book_context}

Summary of what happened in the previous generated chapter (for continuity):
{previous_chapter_summary}
<|eot_id|><|start_header_id|>user<|end_header_id|>

Write the chapter based on this outline: {chapter_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        # Generate
        print("Generating text (this will take time)...")
        generated_chapter = generate_text(full_prompt, tokenizer, model)
        
        # Save to file
        filename = f"{output_dir}/Chapter_{chapter_num}.txt"
        with open(filename, "w") as f:
            f.write(f"PROMPT: {chapter_prompt}\n\n")
            f.write(generated_chapter)
        
        print(f"Saved to {filename}")

        # Update "Memory" (Simple approach: Take key sentences or just last chunk)
        # For a better system, LLM should summarize itself. For now, we use the last 500 chars as 'recent memory'
        # to prevent prompt overflow, or just a placeholder if we want to be fancy later.
        # Let's try to grab the first paragraph and last paragraph as summary.
        lines = generated_chapter.split('\n')
        if len(lines) > 2:
            previous_chapter_summary = lines[0] + " ... " + lines[-1]
        else:
            previous_chapter_summary = generated_chapter[:500] + "..."

    print("\nAll chapters generated successfully!")

if __name__ == "__main__":
    main()
