import os
import glob
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
import argparse

def extract_text_from_epub(epub_path):
    try:
        book = epub.read_epub(epub_path)
        full_text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                full_text.append(text)
        return ' '.join(full_text)
    except Exception as e:
        print(f"Error reading {epub_path}: {e}")
        return ""

def load_text_from_dir(directory):
    text_files = glob.glob(os.path.join(directory, "*.txt"))
    full_text = ""
    for txt_file in text_files:
        print(f"  - Reading {os.path.basename(txt_file)}")
        try:
            with open(txt_file, "r") as f:
                full_text += f.read() + "\n\n"
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    return full_text

def main():
    parser = argparse.ArgumentParser(description="Ingest books into ChromaDB")
    parser.add_argument("--dir", type=str, help="Directory containing .txt files to ingest")
    args = parser.parse_args()

    all_text = ""
    
    # Mode 1: Ingest Specific Directory (Generated Text)
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Directory not found: {args.dir}")
            return
        print(f"Ingesting text files from {args.dir}...")
        all_text = load_text_from_dir(args.dir)

    # Mode 2: Default Ingest (Initial EPUBs)
    else:
        epub_files = glob.glob("extracted_books/*.epub")
        if not epub_files:
            print("No epub files found in extracted_books/!")
            return
        print("Extracting text from EPUBs...")
        for epub_path in epub_files:
            print(f"  - {os.path.basename(epub_path)}")
            all_text += extract_text_from_epub(epub_path) + "\n\n"

    if not all_text:
        print("No text extracted.")
        return
    
    print(f"Total extracted characters: {len(all_text)}")

    # 2. Chunk Text
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.create_documents([all_text])
    print(f"Created {len(chunks)} chunks.")

    # 3. Create/Update Vector DB
    print("Updating Vector Database (ChromaDB)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./chroma_db"
    
    # Chroma automatically persists if persist_directory is set
    # Using .from_documents appends to existing DB if persist_directory matches
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"Database updated at {persist_directory}")

if __name__ == "__main__":
    main()
