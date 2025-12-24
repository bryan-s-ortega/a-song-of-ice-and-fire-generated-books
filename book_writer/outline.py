import os
import sys
import torch
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

# --- Shared Logic (Copied to keep script standalone) ---
def get_rag_context(query, db_path="./chroma_db", k=5): # Increased k for better breadth
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def load_llama3_model():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not found.")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_new_tokens=2000):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def clean_outline(text):
    """
    Cleans the generated outline to only include valid chapter lines.
    Expected format: "Chapter X: [Character Name] - [Summary]" or similar.
    """
    lines = text.split('\n')
    cleaned_lines = []
    # Regex to match "Chapter <number>:" or "Chapter <Roman>:"
    pattern = re.compile(r'^Chapter\s+([0-9]+|[IVXLCDM]+)[:\.]', re.IGNORECASE)
    
    for line in lines:
        stripped = line.strip()
        if pattern.match(stripped):
            cleaned_lines.append(stripped)
            
    return "\n".join(cleaned_lines)

# --- Outline Generation Logic ---

def get_dynamic_roster(tokenizer, model):
    """Asks the model to identify key surviving characters."""
    print("Identifying key surviving characters...")
    
    # We rely on the model's internal knowledge + broad RAG context to pick the roster
    # Llama 3 knows ASOIAF well.
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are George R.R. Martin. List the 10-15 most important living characters active at the end of "A Dance with Dragons" whose storylines need resolution in "The Winds of Winter".
Return ONLY a comma-separated list of names. Do not number them.
Example: Jon Snow, Daenerys Targaryen, Tyrion Lannister
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    response = generate_text(prompt, tokenizer, model, max_new_tokens=100)
    
    # Parse List
    try:
        # Clean up response (remove newlines, extra spaces)
        cleaned = response.replace("\n", "").strip()
        characters = [c.strip() for c in cleaned.split(",") if c.strip()]
        
        # Fallback if model fails to output efficient list
        if len(characters) < 3: 
            print(f"Warning: Model returned sparse roster: {characters}. Using defaults.")
            return [
                "Jon Snow", "Daenerys Targaryen", "Tyrion Lannister", "Cersei Lannister", 
                "Jaime Lannister", "Arya Stark", "Sansa Stark", "Bran Stark", "Theon Greyjoy"
            ]
            
        return characters
    except Exception as e:
        print(f"Error parsing roster: {e}")
        return []

def analyze_world_state(tokenizer, model):
    """Queries RAG for key characters to build a status report."""
    # Step 1: Get Characters Dynamically
    characters = get_dynamic_roster(tokenizer, model)
    print(f"Selected Roster: {characters}")
    
    world_state = "CURRENT STATE OF WESTEROS (End of A Dance with Dragons):\n"
    
    print("\nAnalyzing character states...")
    for char in characters:
        print(f"  - Analyzing {char}...")
        # 1. Get raw context
        context = get_rag_context(f"What is the current location and situation of {char} at the end of A Dance with Dragons?")
        
        # 2. Summarize it with LLM
        summary_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Summarize {char}'s current situation in 2-3 sentences based on the context.
Context: {context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        summary = generate_text(summary_prompt, tokenizer, model, max_new_tokens=150)
        world_state += f"- {char}: {summary.strip()}\n"
        
    return world_state

def main():
    load_dotenv()
    print("Loading Llama 3 (4-bit)...")
    tokenizer, model = load_llama3_model()
    
    # 1. Analyze State
    world_state = analyze_world_state(tokenizer, model)
    print("\n--- World State Analysis ---\n")
    print(world_state)
    
    # 2. Generate Book 6 Outline
    print("\nGenerating 'The Winds of Winter' Outline...")
    prompt_book6 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are George R.R. Martin. Based on the provided character states, create a detailed Chapter Outline for "The Winds of Winter" (Book 6).
- Create a list of 20-25 chapters.
- USE POV FORMAT: "Chapter X: [Character Name] - [Summary]".
- IMPORTANT: Ensure major characters (Jon, Daenerys, Tyrion, Arya, Cersei) have MULTIPLE chapters (e.g., Jon I, Jon II, Jon III) to develop their arcs.
- Focus on resolving the immediate cliffhangers.
- Maintain the dark, complex tone.

Character States:
{world_state}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    outline_6 = generate_text(prompt_book6, tokenizer, model, max_new_tokens=1500)
    outline_6 = clean_outline(outline_6)
    
    with open("winds_of_winter_outline.txt", "w") as f:
        f.write(outline_6)
    print("Saved to winds_of_winter_outline.txt")

    # 3. Generate Book 7 Outline
    print("\nGenerating 'A Dream of Spring' Outline...")
    prompt_book7 = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are George R.R. Martin. Based on the outline you just created for Book 6, create the final Chapter Outline for "A Dream of Spring" (Book 7).
- Create a list of 20-25 chapters using POV Format (e.g. "Chapter X: [Character Name]").
- Ensure major characters have multiple chapters to conclude their arcs properly.
- Bring the saga to a bittersweet conclusion.
- Resolve the White Walker threat and the Iron Throne succession.

Previous Book (Winds of Winter) Outline:
{outline_6}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    outline_7 = generate_text(prompt_book7, tokenizer, model, max_new_tokens=1500)
    outline_7 = clean_outline(outline_7)
    
    with open("dream_of_spring_outline.txt", "w") as f:
        f.write(outline_7)
    print("Saved to dream_of_spring_outline.txt")

if __name__ == "__main__":
    main()
