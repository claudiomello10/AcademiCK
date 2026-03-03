"""Fallback PDF processor — last-resort flat processing with page-aware chunking."""

import os
import re
import logging
from bisect import bisect_right
from typing import List, Dict

import fitz  # PyMuPDF
from langchain.text_splitter import NLTKTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


class FallbackPDFProcessor:
    """
    Last-resort PDF processor. No chapter/topic detection — just extracts
    all text page-by-page, splits into chunks, and tracks source pages.

    Processing order in tasks.py:
        1. DefaultPDFProcessor  (LLM-based, primary)
        2. DoclingPDFProcessor  (layout-based)
        3. FallbackPDFProcessor (this class, last resort)
    """

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.min_chunk_length = settings.min_chunk_length

    def get_all_chunks(self, file_path: str, book_name: str = None) -> List[Dict]:
        """
        Extract all chunks with per-chunk page numbers.

        Extracts text from every page, concatenates with offset tracking,
        splits with NLTKTextSplitter, and maps each chunk to its source page.

        Args:
            file_path: Absolute path to the PDF file.
            book_name: Display name of the book.

        Returns:
            List of chunk dicts with keys: book_name, text, page.
        """
        if book_name is None:
            book_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            return []

        full_text = ""
        page_offsets = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = self._clean_text(page.get_text("text"))
            if page_text.strip():
                page_offsets.append((len(full_text), page_num + 1))
                full_text += page_text + "\n\n"

        doc.close()
        full_text = full_text.strip()

        if not full_text or not page_offsets:
            logger.info(f"No text extracted from {file_path}")
            return []

        text_splitter = NLTKTextSplitter(
            chunk_size=self.chunk_size,
            separator="",
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            use_span_tokenize=True,
        )

        docs = text_splitter.create_documents([full_text])
        chunks = []

        for doc in docs:
            text = doc.page_content.encode("utf-8", errors="ignore").decode("utf-8")
            if len(text.strip()) < self.min_chunk_length:
                continue

            page = self._page_from_offset(doc.metadata["start_index"], page_offsets)
            chunks.append({
                "book_name": book_name,
                "text": text,
                "page": page,
            })

        logger.info(f"Fallback processor extracted {len(chunks)} chunks from {file_path}")
        return chunks

    @staticmethod
    def _page_from_offset(start_index: int, page_offsets: list) -> int:
        """Map a character offset to its 1-based page number."""
        offsets = [o for o, _ in page_offsets]
        idx = bisect_right(offsets, start_index) - 1
        return page_offsets[max(idx, 0)][1]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted PDF text."""
        if not text:
            return ""

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not (stripped.isdigit() and len(stripped) < 5):
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')

        return text.strip()
