"""PDF processing, chunking, and embedding generation.

Extracts text from PDFs, splits into chunks by chapter/topic,
generates dense embeddings using BGE-M3, and returns a pandas DataFrame.
"""

import os
import re

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import NLTKTextSplitter
from tqdm import tqdm

from llm_provider import generate_response


class PDFProcessor:
    """Processes PDFs into chunked text with embeddings."""

    def __init__(self, embedding_model, llm_model: str = "gpt-5-mini"):
        """
        Args:
            embedding_model: Already-loaded BGEM3FlagModel instance (shared with StudentHelper)
            llm_model: Model name for chapter identification via LLM
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def process_pdf(self, pdf_path: str, book_name: str = None) -> pd.DataFrame:
        """
        Full pipeline: extract TOC -> identify chapters -> chunk text -> generate embeddings.

        Args:
            pdf_path: Path to the PDF file
            book_name: Name for the book (defaults to filename)

        Returns:
            DataFrame with columns: Book, Chapter, Text, Topic, is_introduction, dense_embedding
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        if book_name is None:
            book_name = os.path.basename(pdf_path).replace(".pdf", "")

        # Step 1: Extract table of contents
        toc = self._get_toc(pdf_path)
        if not toc:
            print(f"No table of contents found in {book_name}. Skipping.")
            return pd.DataFrame()

        # Step 2: Identify chapters via LLM
        summary_list = self._get_summary_list(toc, book_name)
        if not summary_list:
            print(f"No chapters identified in {book_name}. Skipping.")
            return pd.DataFrame()

        # Step 3: Extract and chunk text
        reader = PdfReader(pdf_path)
        text_splitter = NLTKTextSplitter(chunk_size=3000, separator="\n", chunk_overlap=1000)

        rows = []
        for index, chapter in tqdm(
            enumerate(summary_list),
            desc=f"Processing: {book_name}",
            total=len(summary_list),
            unit="chapter",
        ):
            title = chapter["Title"]
            chapter_page = chapter["Page"] - 1
            topics = chapter["topics"]

            try:
                # Process chapter introduction (or whole chapter if no topics)
                if topics:
                    pre_topic_text = ""
                    for i in range(chapter_page, topics[0]["Page"]):
                        page_text = reader.pages[i].extract_text() or ""
                        title_test = topics[0]["Topic"] + "\n"
                        if title_test in page_text:
                            page_text = page_text.split(title_test)[0]
                        pre_topic_text += page_text
                else:
                    if index == len(summary_list) - 1:
                        next_page = len(reader.pages)
                    else:
                        next_page = summary_list[index + 1]["Page"]

                    pre_topic_text = ""
                    for i in range(chapter_page, next_page):
                        pre_topic_text += reader.pages[i].extract_text() or ""

                for text in text_splitter.split_text(pre_topic_text):
                    text = text.encode("utf-8", errors="ignore").decode("utf-8")
                    if self._should_skip_chunk(text):
                        continue

                    rows.append({
                        "Book": book_name,
                        "Chapter": title,
                        "Text": text,
                        "Topic": "Chapter Introduction",
                        "is_introduction": True,
                    })
            except Exception as e:
                print(f"  Error processing intro for '{title}': {e}")
                continue

            # Process each topic within the chapter
            try:
                for topic in topics:
                    topic_title = topic["Topic"]
                    topic_page = topic["Page"] - 1

                    # Determine end page for this topic
                    if topic == topics[-1]:
                        if chapter == summary_list[-1]:
                            next_topic_page = len(reader.pages)
                            next_topic_title = ""
                        else:
                            next_topic_title = summary_list[index + 1]["topics"][0]["Topic"]
                            next_topic_page = summary_list[index + 1]["Page"]
                    else:
                        next_idx = topics.index(topic) + 1
                        next_topic_page = topics[next_idx]["Page"]
                        next_topic_title = topics[next_idx]["Topic"]

                    topic_text = ""
                    for i in range(topic_page, next_topic_page):
                        page_text = reader.pages[i].extract_text() or ""
                        title_test = topic_title + "\n"
                        if title_test in page_text:
                            page_text = page_text.split(title_test)[1]
                        if next_topic_title:
                            title_test = next_topic_title + "\n"
                            if title_test in page_text:
                                page_text = page_text.split(title_test)[0]
                        topic_text += page_text

                    for text in text_splitter.split_text(topic_text):
                        text = text.encode("utf-8", errors="ignore").decode("utf-8")
                        if self._should_skip_chunk(text):
                            continue

                        # Skip index sections
                        topic_lower = re.sub(r"[\s\.\,\:\;\-\_]+", "", topic_title.lower())
                        if topic_lower == "index" or re.match(r"^\d+\.?\s*index$", topic_lower):
                            continue

                        rows.append({
                            "Book": book_name,
                            "Chapter": title,
                            "Text": text,
                            "Topic": topic_title,
                            "is_introduction": False,
                        })
            except Exception as e:
                print(f"  Error processing topics for '{title}': {e}")
                continue

        if not rows:
            print(f"No valid chunks extracted from {book_name}.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Step 4: Generate embeddings
        print(f"Generating embeddings for {len(df)} chunks...")
        embeddings = []
        for _, row in tqdm(df.iterrows(), desc="Embedding", total=len(df)):
            dense = self.embedding_model.encode(
                [row["Text"]], return_dense=True
            )["dense_vecs"][0]
            embeddings.append(dense)

        df["dense_embedding"] = embeddings
        print(f"Done! {len(df)} chunks processed for {book_name}.")
        return df

    def process_multiple_pdfs(self, paths: list[str], book_names: list[str] = None) -> pd.DataFrame:
        """Process multiple PDFs and return a combined DataFrame."""
        all_dfs = []
        for i, path in enumerate(paths):
            name = book_names[i] if book_names else None
            df = self.process_pdf(path, name)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _get_toc(self, pdf_path: str) -> list:
        """Extract table of contents from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        doc.close()
        return toc

    def _get_summary_list(self, toc: list, book_name: str) -> list | None:
        """Identify chapters from TOC using LLM, then build structured summary."""
        # Format TOC for the LLM prompt
        toc_text = f"Book: {book_name}\n\nTable of Contents:\n\n"
        for item in toc:
            toc_text += (
                f"Importance Index: {item[0]} -- "
                f"Topic name: {item[1]} -- "
                f"Topic page: {item[2]}\n"
            )

        prompt = """Given the table of contents of this book containing the importance index, topic name and topic page, provide me with the list of chapters and apendixes in the book.
        Do not inclue figures, tables, preface, index, bibliography,  or any other non-chapter or non-appendix sections.
        If an appendix is present with sub-sections, the sub-sections should not be included in the list, only the appendix name. For example, if the appendix is "Appendix A" and it has sub-sections "A.1", "A.2", "A.3", only "Appendix A" should be included in the list.
        If the book is separated into parts, the parts should not be included in the list.
        This list must be contain only the Name of the sections.
        The Names must be exactly as they appear in the table of contents.
        The response must have only the list in python format. For example, if the list is ['a', 'b', 'c'], the response must be ['a', 'b', 'c']. It cannot have any other text. If the list is empty, the response must be []."""

        messages = [{"role": "user", "content": toc_text + prompt}]
        answer = generate_response(messages, self.llm_model, temperature=0.3)

        # Parse the LLM response as a Python list
        try:
            chapters = eval(answer)
        except Exception:
            try:
                start = answer.find("```python")
                end = answer.find("```", start + 1)
                chapters = eval(answer[start + 9:end].strip())
            except Exception:
                print(f"Could not parse chapter list from LLM response.")
                return None

        if not chapters:
            return None

        # Normalize chapter names for matching
        def normalize(name):
            return name.lower().strip().replace("\n", " ").replace("  ", " ").replace("\xa0", " ")

        selected = [normalize(c) for c in chapters]

        # Build the summary list with topics
        summary_list = []
        current_chapter = None
        for item in toc:
            processed = normalize(item[1])
            if processed in selected:
                summary_list.append({"Page": item[2], "Title": item[1], "topics": []})
                current_chapter = item[1]
            elif current_chapter is not None:
                summary_list[-1]["topics"].append({"Page": item[2], "Topic": item[1]})

        return summary_list if summary_list else None

    def _should_skip_chunk(self, text: str) -> bool:
        """Check if a chunk should be skipped (too short or too many dots)."""
        if len(text) < 300:
            return True

        # Skip chunks with too many dots (likely TOC or index pages)
        normalized = re.sub(r"\.{2,}", "", text)
        filtered = re.sub(r"\d\.\d", lambda m: m.group().replace(".", ""), normalized)
        if filtered.count(".") / len(text) > 0.02:
            return True

        return False
