"""PDF processing service for text extraction and TOC analysis."""

import re
import logging
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and structure analysis."""

    def __init__(self):
        self.chapter_patterns = [
            r'^chapter\s+(\d+)',
            r'^(\d+)\.\s+',
            r'^part\s+(\d+)',
            r'^section\s+(\d+)',
            r'^capítulo\s+(\d+)',
        ]

    def get_table_of_contents(self, file_path: str) -> List[Dict]:
        """
        Extract table of contents from PDF.

        Returns list of TOC entries with title, level, and page number.
        """
        try:
            doc = fitz.open(file_path)
            toc = doc.get_toc()
            doc.close()

            if not toc:
                logger.info(f"No TOC found in {file_path}, will analyze structure")
                return []

            entries = []
            for level, title, page in toc:
                entries.append({
                    "level": level,
                    "title": title.strip(),
                    "page": page
                })

            return entries

        except Exception as e:
            logger.error(f"Error extracting TOC from {file_path}: {e}")
            return []

    def analyze_chapters(
        self,
        toc: List[Dict],
        book_name: str
    ) -> List[Dict]:
        """
        Analyze TOC to identify chapter boundaries.

        Returns list of chapters with title, start_page, end_page.
        """
        if not toc:
            # If no TOC, treat entire book as one chapter
            return [{
                "title": book_name,
                "start_page": 1,
                "end_page": None,
                "level": 1
            }]

        chapters = []
        # Filter to top-level entries (chapters)
        top_level = min(entry["level"] for entry in toc) if toc else 1

        chapter_entries = [e for e in toc if e["level"] == top_level]

        for i, entry in enumerate(chapter_entries):
            start_page = entry["page"]

            # End page is start of next chapter - 1, or None for last chapter
            if i < len(chapter_entries) - 1:
                end_page = chapter_entries[i + 1]["page"] - 1
            else:
                end_page = None

            chapters.append({
                "title": self._clean_title(entry["title"]),
                "start_page": start_page,
                "end_page": end_page,
                "level": entry["level"]
            })

        return chapters

    def extract_chapter_text(
        self,
        file_path: str,
        start_page: Optional[int],
        end_page: Optional[int]
    ) -> str:
        """
        Extract text from a range of pages.

        Args:
            file_path: Path to PDF file
            start_page: Starting page (1-indexed)
            end_page: Ending page (inclusive), None for end of document
        """
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)

            # Convert to 0-indexed
            start_idx = (start_page or 1) - 1
            end_idx = (end_page or total_pages) - 1

            # Ensure valid range
            start_idx = max(0, min(start_idx, total_pages - 1))
            end_idx = max(start_idx, min(end_idx, total_pages - 1))

            text_parts = []

            for page_num in range(start_idx, end_idx + 1):
                page = doc[page_num]
                text = page.get_text("text")
                text_parts.append(text)

            doc.close()

            # Join and clean text
            full_text = "\n\n".join(text_parts)
            return self._clean_text(full_text)

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""

    def extract_full_text(self, file_path: str) -> str:
        """Extract all text from PDF."""
        return self.extract_chapter_text(file_path, None, None)

    def get_page_count(self, file_path: str) -> int:
        """Get total number of pages in PDF."""
        try:
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error(f"Error getting page count for {file_path}: {e}")
            return 0

    def extract_text_with_pages(
        self,
        file_path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> List[Dict]:
        """
        Extract text page by page with page numbers.

        Returns list of {page: int, text: str}.
        """
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)

            start_idx = (start_page or 1) - 1
            end_idx = (end_page or total_pages) - 1

            start_idx = max(0, min(start_idx, total_pages - 1))
            end_idx = max(start_idx, min(end_idx, total_pages - 1))

            pages = []

            for page_num in range(start_idx, end_idx + 1):
                page = doc[page_num]
                text = self._clean_text(page.get_text("text"))
                if text.strip():
                    pages.append({
                        "page": page_num + 1,  # Back to 1-indexed
                        "text": text
                    })

            doc.close()
            return pages

        except Exception as e:
            logger.error(f"Error extracting text with pages from {file_path}: {e}")
            return []

    def _clean_title(self, title: str) -> str:
        """Clean and normalize chapter title."""
        # Remove extra whitespace
        title = " ".join(title.split())

        # Remove common prefixes if too long
        if len(title) > 100:
            title = title[:100] + "..."

        return title

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""

        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)

        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove header/footer patterns (lines that are just numbers or short)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep line if it's not just a page number
            if stripped and not (stripped.isdigit() and len(stripped) < 5):
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')

        return text.strip()

    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words.

        Returns 'en' for English, 'pt' for Portuguese, 'unknown' otherwise.
        """
        text_lower = text.lower()

        # Portuguese indicators
        pt_words = ['que', 'para', 'como', 'não', 'uma', 'com', 'por', 'mais']
        pt_count = sum(1 for word in pt_words if f' {word} ' in text_lower)

        # English indicators
        en_words = ['the', 'and', 'that', 'this', 'with', 'for', 'are', 'have']
        en_count = sum(1 for word in en_words if f' {word} ' in text_lower)

        if pt_count > en_count:
            return 'pt'
        elif en_count > pt_count:
            return 'en'
        else:
            return 'unknown'

    def extract_metadata(self, file_path: str) -> Dict:
        """Extract PDF metadata."""
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata
            doc.close()

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", "")
            }

        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return {}
