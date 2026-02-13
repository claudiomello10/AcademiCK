"""AcademiCK CLI — Simplified RAG Study Assistant.

A minimal command-line version of AcademiCK for learning how the
RAG pipeline works without Docker, microservices, or databases.

Usage:
    python main.py
"""

import os
import sys

import nltk
from dotenv import load_dotenv


def main():
    load_dotenv()

    # Download NLTK data needed for text splitting
    nltk.download("punkt_tab", quiet=True)

    print("=" * 60)
    print("  AcademiCK — RAG Study Assistant (Simple CLI)")
    print("  A simplified version for learning RAG concepts")
    print("=" * 60)
    print()

    # Initialize the system
    print("Initializing... (first run downloads ~2 GB of ML models)")
    print()

    from student_helper import StudentHelper
    from pdf_processor import PDFProcessor

    helper = StudentHelper(
        data_path=os.getenv("DATA_PATH", "data/embeddings.pkl"),
        model=os.getenv("DEFAULT_MODEL", "gpt-5-mini"),
        rag_model=os.getenv("QUERY_ENHANCEMENT_MODEL", "gpt-5-nano"),
        subject=os.getenv("DEFAULT_SUBJECT", "Machine Learning"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        device=os.getenv("DEVICE"),
    )

    processor = PDFProcessor(
        embedding_model=helper.embedding_model,
        llm_model=os.getenv("QUERY_ENHANCEMENT_MODEL", "gpt-5-nano"),
    )

    print("\nReady!\n")

    # Main menu loop
    while True:
        print_menu(helper)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            chat_loop(helper)
        elif choice == "2":
            process_pdfs(processor, helper)
        elif choice == "3":
            list_books(helper)
        elif choice == "4":
            change_subject(helper)
        elif choice == "5":
            change_model(helper)
        elif choice == "6":
            helper.clear_history()
            print("Chat history cleared.\n")
        elif choice == "7" or choice.lower() == "q":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option.\n")


def print_menu(helper):
    book_count = len(helper.get_books())
    exchanges = len(helper.messages) // 2
    print(f"--- Main Menu ---")
    print(f"  Subject: {helper.subject}  |  Model: {helper.model}  |  Books: {book_count}  |  History: {exchanges} exchanges")
    print()
    print("  1. Chat (ask questions)")
    print("  2. Process PDFs")
    print("  3. List books")
    print("  4. Change subject")
    print("  5. Change model")
    print("  6. Clear chat history")
    print("  7. Exit")
    print()


def chat_loop(helper):
    if helper.dense_matrix is None:
        print("\nNo books loaded. Process PDFs first (option 2).\n")
        return

    print("\n--- Chat Mode ---")
    print("Type your question, or 'back' to return to menu.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() == "back":
            print()
            break

        try:
            print("\nThinking...\n")
            response = helper.ask(query)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def process_pdfs(processor, helper):
    print("\n--- Process PDFs ---")
    default_dir = os.getenv("PDF_DIRECTORY", "books/")
    path = input(f"PDF file or directory [{default_dir}]: ").strip() or default_dir

    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        default_name = os.path.basename(path).replace(".pdf", "")
        book_name = input(f"Book name [{default_name}]: ").strip() or default_name

        print(f"\nProcessing '{book_name}'...")
        df = processor.process_pdf(path, book_name)
        if not df.empty:
            helper.save_embeddings(df)
            print(f"Saved! {len(df)} chunks added.\n")
        else:
            print("No chunks were extracted.\n")

    elif os.path.isdir(path):
        pdf_files = sorted([f for f in os.listdir(path) if f.lower().endswith(".pdf")])
        if not pdf_files:
            print("No PDF files found in that directory.\n")
            return

        print(f"\nFound {len(pdf_files)} PDF(s):")
        for f in pdf_files:
            print(f"  - {f}")

        confirm = input("\nProcess all? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.\n")
            return

        for f in pdf_files:
            book_name = f.replace(".pdf", "")
            print(f"\nProcessing '{book_name}'...")
            df = processor.process_pdf(os.path.join(path, f), book_name)
            if not df.empty:
                helper.save_embeddings(df)
                print(f"  {len(df)} chunks added.")

        print("\nAll PDFs processed!\n")
    else:
        print("Invalid path. Provide a .pdf file or a directory.\n")


def list_books(helper):
    books = helper.get_books()
    if not books:
        print("\nNo books loaded. Process PDFs first (option 2).\n")
    else:
        print(f"\n--- Books ({len(books)}) ---")
        for i, book in enumerate(books, 1):
            print(f"  {i}. {book}")
        print()


def change_subject(helper):
    print(f"\nCurrent subject: {helper.subject}")
    new = input("New subject: ").strip()
    if new:
        helper.set_subject(new)
        print(f"Subject set to: {new}\n")
    else:
        print("No change.\n")


def change_model(helper):
    print(f"\nCurrent model: {helper.model}")
    print("Examples: gpt-4o-mini, claude-3-5-haiku-latest, deepseek-chat")
    new = input("New model: ").strip()
    if new:
        helper.set_model(new)
        print(f"Model set to: {new}\n")
    else:
        print("No change.\n")


if __name__ == "__main__":
    main()
