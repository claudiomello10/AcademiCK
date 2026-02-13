"""
This processor uses LLM-based chapter identification and NLTK chunking for
higher quality, more structured output compared to the fallback processor.
"""

import os
import re
import logging
from typing import List, Dict, Optional

import fitz  # PyMuPDF
from pypdf import PdfReader
from langchain.text_splitter import NLTKTextSplitter
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class OriginalPDFProcessor:
    """
    PDF processor that uses LLM-based chapter identification and NLTK chunking.

    This produces higher quality, hierarchically structured output:
    - Chapter → Topics hierarchy
    - 3000 char chunks with 1000 char overlap
    - Period-based quality filter (skip chunks with >2% periods)
    - Minimum chunk length filter (300 chars)
    """

    def __init__(self, llm_model: str = "gpt-5-mini"):
        """Initialize the OriginalPDFProcessor."""
        self.model = llm_model
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.chunk_size = settings.chunk_size  # Default 3000
        self.chunk_overlap = settings.chunk_overlap  # Default 1000
        self.min_chunk_length = settings.min_chunk_length  # Default 300

    def get_table_of_contents_from_PDF(self, path: str) -> List:
        """
        Get the table of contents (TOC) from a PDF file.

        Args:
            path: Path to the PDF file

        Returns:
            List of TOC entries: [[level, title, page], ...]
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")

        doc = fitz.open(path)
        toc = doc.get_toc()
        doc.close()
        return toc

    def get_processed_name(self, name: str) -> str:
        """Process text names for consistency."""
        name = name.lower().strip().replace("\n", " ").replace("  ", " ")
        name = name.replace("\xa0", " ")
        return name

    def generate_toc_text_for_prompting(self, toc: List, book_name: str) -> str:
        """Generate prompt text from table of contents."""
        text = f"Book: {book_name}\n\nTable of Contents:\n\n"

        for item in toc:
            text += (
                f"Importance Index: {item[0]} -- "
                f"Topic name: {item[1]} -- "
                f"Topic page: {item[2]}\n"
            )
        return text

    def get_model_answer_of_chapters(self, toc_prompt: str, model: str = None) -> List[str]:
        """
        Get model's analysis of chapters from TOC.

        Uses LLM to identify actual chapters from the TOC, filtering out
        figures, tables, preface, index, bibliography, etc.

        Args:
            toc_prompt: TOC prompt text
            model: Model to use (defaults to self.model)

        Returns:
            List of chapter names
        """
        if model is None:
            model = self.model

        prompt_engineering = """Given the table of contents of this book containing the importance index, topic name and topic page, provide me with the list of chapters and apendixes in the book.
        Do not inclue figures, tables, preface, index, bibliography, or any other non-chapter or non-appendix sections.
        If an appendix is present with sub-sections, the sub-sections should not be included in the list, only the appendix name. For example, if the appendix is "Appendix A" and it has sub-sections "A.1", "A.2", "A.3", only "Appendix A" should be included in the list.
        If the book is separated into parts, the parts should not be included in the list.
        This list must be contain only the Name of the sections.
        The Names must be exactly as they appear in the table of contents.
        The response must have only the list in python format. For example, if the list is ['a', 'b', 'c'], the response must be ['a', 'b', 'c']. It cannot have any other text. If the list is empty, the response must be []."""

        full_prompt = toc_prompt + prompt_engineering
        messages = [{"role": "user", "content": full_prompt}]

        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            answer = completion.choices[0].message.content

            # Try to parse the response as a Python list
            try:
                return eval(answer)
            except:
                # Try to extract from markdown code block
                try:
                    start = answer.find("```python")
                    end = answer.find("```", start + 1)
                    answer = answer[start + 9 : end].strip()
                    return eval(answer)
                except:
                    # Try to find a list pattern
                    import ast
                    match = re.search(r'\[.*\]', answer, re.DOTALL)
                    if match:
                        return ast.literal_eval(match.group())
                    raise Exception("Error processing chapters from table of contents")
        except Exception as e:
            logger.error(f"Error getting chapters from LLM: {e}")
            raise

    def get_summary_list_from_PDF(self, path: str, book_name: str = None) -> Optional[List[Dict]]:
        """
        Get a summary list from PDF with chapter and topic structure.

        Args:
            path: Path to PDF file
            book_name: Name of the book

        Returns:
            List of chapters with their topics:
            [
                {
                    "Page": 1,
                    "Title": "Chapter 1",
                    "topics": [
                        {"Page": 5, "Topic": "Introduction"},
                        ...
                    ]
                },
                ...
            ]
        """
        if book_name is None:
            book_name = os.path.basename(path).split(".")[0]

        toc = self.get_table_of_contents_from_PDF(path)

        if not toc:
            logger.warning(f"No TOC found in {book_name}")
            return None

        toc_prompt = self.generate_toc_text_for_prompting(toc, book_name)
        selected_chapters = self.get_model_answer_of_chapters(toc_prompt)
        selected_chapters = [
            self.get_processed_name(chapter) for chapter in selected_chapters
        ]

        if not selected_chapters:
            logger.warning(f"No chapters selected for {book_name}")
            return None

        summary_list = []
        current_chapter = None

        for item in toc:
            processed_name = self.get_processed_name(item[1])

            if processed_name in selected_chapters:
                summary_list.append({
                    "Page": item[2],
                    "Title": item[1],
                    "topics": []
                })
                current_chapter = item[1]
            elif current_chapter is not None:
                summary_list[-1]["topics"].append({
                    "Page": item[2],
                    "Topic": item[1]
                })

        return summary_list

    def _should_skip_chunk(self, text: str) -> bool:
        """
        Check if a chunk should be skipped based on quality filters.

        Filters:
        - Chunks shorter than min_chunk_length
        - Chunks with too many periods (likely TOC or index content)
        """
        # Skip chunks that are too small
        if len(text) < self.min_chunk_length:
            return True

        # Skip chunks with too many periods
        # Remove consecutive periods and periods between numbers
        normalized_text = re.sub(r"\.{2,}", "", text)
        filtered_text = re.sub(
            r"\d\.\d", lambda m: m.group().replace(".", ""), normalized_text
        )
        if filtered_text.count(".") / len(text) > 0.02:
            return True

        return False

    def process_chapter(
        self,
        reader: PdfReader,
        chapter: Dict,
        chapter_index: int,
        summary_list: List[Dict],
        book_name: str,
        text_splitter: NLTKTextSplitter
    ) -> List[Dict]:
        """
        Process a single chapter and return chunks with metadata.

        Args:
            reader: PDF reader instance
            chapter: Chapter dict with Page, Title, topics
            chapter_index: Index of this chapter in summary_list
            summary_list: Full summary list for context
            book_name: Name of the book
            text_splitter: NLTK text splitter instance

        Returns:
            List of chunk dicts with text and metadata
        """
        chunks = []
        title = chapter["Title"]
        chapter_page = chapter["Page"] - 1
        topics = chapter["topics"]

        try:
            # Process chapter introduction or whole chapter if no topics
            if topics:
                # If there are topics, process the introduction
                pre_topic_text = ""
                for i in range(chapter_page, topics[0]["Page"]):
                    page_text = reader.pages[i].extract_text()
                    title_test_text = topics[0]["Topic"] + "\n"
                    if title_test_text in page_text:
                        page_text = page_text.split(title_test_text)[0]
                    pre_topic_text += page_text
            else:
                # If there are no topics, process the entire chapter
                if chapter_index == len(summary_list) - 1:
                    next_chapter_page = len(reader.pages)
                else:
                    next_chapter_page = summary_list[chapter_index + 1]["Page"]

                pre_topic_text = ""
                for i in range(chapter_page, next_chapter_page):
                    pre_topic_text += reader.pages[i].extract_text()

            # Process introduction chunks
            for text in text_splitter.split_text(pre_topic_text):
                text = text.encode("utf-8", errors="ignore").decode("utf-8")

                if self._should_skip_chunk(text):
                    continue

                chunks.append({
                    "book_name": book_name,
                    "chapter": title,
                    "text": text,
                    "topic": "Chapter Introduction",
                    "is_introduction": True
                })

        except Exception as e:
            logger.error(f"Error processing chapter introduction for {title}: {e}")

        # Process each topic
        try:
            for topic_idx, topic in enumerate(topics):
                topic_title = topic["Topic"]
                topic_page = topic["Page"] - 1

                # Determine next topic page
                if topic_idx == len(topics) - 1:
                    if chapter_index == len(summary_list) - 1:
                        next_topic_page = len(reader.pages)
                        next_topic_title = ""
                    else:
                        next_chapter = summary_list[chapter_index + 1]
                        next_topic_title = (
                            next_chapter["topics"][0]["Topic"]
                            if next_chapter["topics"]
                            else ""
                        )
                        next_topic_page = next_chapter["Page"]
                else:
                    next_topic_page = topics[topic_idx + 1]["Page"]
                    next_topic_title = topics[topic_idx + 1]["Topic"]

                # Extract and process topic text
                topic_text = ""
                for i in range(topic_page, next_topic_page):
                    page_text = reader.pages[i].extract_text()

                    title_test_text = topic_title + "\n"
                    if title_test_text in page_text:
                        page_text = page_text.split(title_test_text)[1]

                    if next_topic_title:
                        title_test_text = next_topic_title + "\n"
                        if title_test_text in page_text:
                            page_text = page_text.split(title_test_text)[0]

                    topic_text += page_text

                # Skip if the topic is exclusively an index section
                topic_lower = re.sub(r"[\s\.\,\:\;\-\_]+", "", topic_title.lower())
                if topic_lower == "index" or re.match(r"^\d+\.?\s*index$", topic_lower):
                    continue

                # Process topic chunks
                for text in text_splitter.split_text(topic_text):
                    text = text.encode("utf-8", errors="ignore").decode("utf-8")

                    if self._should_skip_chunk(text):
                        continue

                    chunks.append({
                        "book_name": book_name,
                        "chapter": title,
                        "text": text,
                        "topic": topic_title,
                        "is_introduction": False
                    })

        except Exception as e:
            logger.error(f"Error processing topics for chapter {title}: {e}")

        return chunks

    def get_all_chunks(self, path: str, book_name: str = None) -> List[Dict]:
        """
        Get all chunks from a PDF file.

        Args:
            path: Path to PDF file
            book_name: Name of the book

        Returns:
            List of chunk dicts with text and metadata
        """
        if book_name is None:
            book_name = os.path.basename(path).split(".")[0]

        # Get chapter structure using LLM
        summary_list = self.get_summary_list_from_PDF(path, book_name)

        if not summary_list:
            raise ValueError(f"No chapters found in {book_name}")

        # Initialize reader and text splitter
        reader = PdfReader(path)
        text_splitter = NLTKTextSplitter(
            chunk_size=self.chunk_size,
            separator="\n",
            chunk_overlap=self.chunk_overlap
        )

        all_chunks = []
        for idx, chapter in enumerate(summary_list):
            chapter_chunks = self.process_chapter(
                reader, chapter, idx, summary_list, book_name, text_splitter
            )
            all_chunks.extend(chapter_chunks)

        return all_chunks
