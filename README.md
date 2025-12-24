# AI-Powered Book Completion: A Song of Ice and Fire

This project utilizes advanced Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to autonomously generate, review, and refine the final two books of *A Song of Ice and Fire*: **The Winds of Winter** and **A Dream of Spring**.

## 🚀 Key Features

*   **RAG-Enhanced Writing**: Uses a vector database (ChromaDB) containing the text of the previous books to ensure generated content maintains continuity with the established lore, character voices, and world-building of George R.R. Martin.
*   **Iterative Workflow**: employing a "Writer -> Editor -> Improver" loop:
    1.  **Write**: Generates initial chapters based on a structured outline.
    2.  **Review**: An "Editor" LLM critiques each chapter for consistency, tone, and plot logic.
    3.  **Improve**: Rewrites the chapters to address the editor's feedback.
*   **Automatic Summarization**: Generates chapter-by-chapter plot summaries for quick review of the generated narrative.
*   **Local LLM Support**: Optimized for running locally with Llama 3 8B using 4-bit quantization (via `bitsandbytes`) for efficient performance on consumer hardware.

## 🛠️ Tech Stack & Decisions

*   **Python 3.13+**: Core language for AI/ML development.
*   **Meta Llama 3 (8B Instruct)**: Chosen for its high reasoning capabilities and efficiency. It can run locally on consumer GPUs (e.g. RTX 3090/4090) when quantized, avoiding API costs and privacy concerns.
*   **LangChain**: Used as the orchestration layer for the RAG pipeline.
    *   *Why?* It provides abstractions for easy text splitting (`RecursiveCharacterTextSplitter`) and seamless integration with ChromaDB and HuggingFace embeddings.
*   **ChromaDB**: A lightweight, open-source vector database.
    *   *Why?* It's easy to set up locally (persists to disk), requires no external server process, and integrates well with Python.
*   **HuggingFace Embeddings (`all-MiniLM-L6-v2`)**: fast and efficient sentence transformer model.
    *   *Why?* perfect balance of speed and semantic search quality for this use case.
*   **Just**: A handy command runner.
    *   *Why?* Simplifies complex `uv run python ...` commands into simple recipes like `just write-winds`.
*   **uv**: An extremely fast Python package and project manager.
    *   *Why?* Replaces `pip`, `poetry`, and `virtualenv` with a single, faster tool.

## 📋 Workflow

The project is managed via a `justfile` which automates the entire pipeline:

1.  **Ingest Content**: Loads existing books into the vector database.
    ```bash
    just ingest
    ```
2.  **Generate Outlines**: Creates detailed chapter outlines for the new books.
    ```bash
    just outlines
    ```
3.  **Write Drafts**: Generates the raw book text chapter by chapter.
    ```bash
    just write-winds
    just write-dream
    ```
4.  **Review**: Generates editorial critiques for every chapter.
    ```bash
    just review-winds
    just review-dream
    ```
5.  **Improve**: Rewrites the books based on the editorial feedback.
    ```bash
    just improve-winds
    just improve-dream
    ```
6.  **Summarize**: Creates a high-level plot summary of the final improved books.
    ```bash
    just summarize-winds
    just summarize-dream
    ```

## 📂 Project Structure

*   `ingest_books.py`: Parses text files and builds the ChromaDB knowledge base.
*   `autowrite_book.py`: The core generation script using RAG and Llama 3.
*   `review_books.py`: The "Editor" agent that critiques consistency and tone.
*   `improve_books.py`: The "Refiner" agent that rewrites text based on reviews.
*   `summarize_books.py`: Generates concise plot summaries.
*   `justfile`: Command runner for all project tasks.
