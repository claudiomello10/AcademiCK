"""Text chunking service for splitting documents into semantic chunks."""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks for embedding.

    Uses sentence-aware splitting to avoid breaking mid-sentence.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_length: int = 100
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_length: Minimum chunk length (shorter chunks are merged)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        # Sentence ending patterns
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

        # Paragraph pattern
        self.paragraph_pattern = re.compile(r'\n\s*\n')

        # Section header patterns
        self.header_patterns = [
            re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),  # Markdown headers
            re.compile(r'^\d+\.\d*\s+.+$', re.MULTILINE),  # Numbered sections
            re.compile(r'^[A-Z][^.!?]*:$', re.MULTILINE),  # Colon headers
        ]

    def chunk_text(
        self,
        text: str,
        chapter_title: Optional[str] = None
    ) -> List[Dict]:
        """
        Split text into chunks with metadata.

        Args:
            text: Text to chunk
            chapter_title: Optional chapter title for context

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []

        # Clean and normalize text
        text = self._normalize_text(text)

        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)

        # Build chunks from paragraphs
        chunks = self._build_chunks(paragraphs)

        # Add metadata
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                "text": chunk_text,
                "chunk_index": i,
                "char_count": len(chunk_text),
                "is_introduction": i == 0,
                "topic": self._extract_topic(chunk_text)
            }

            if chapter_title:
                chunk_data["chapter"] = chapter_title

            result.append(chunk_data)

        return result

    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into sentence-based chunks."""
        sentences = self._split_sentences(text)
        return self._combine_sentences_to_chunks(sentences)

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Split text keeping paragraphs intact when possible."""
        paragraphs = self._split_paragraphs(text)
        return self._build_chunks(paragraphs)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent chunking."""
        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Normalize whitespace within lines
        text = re.sub(r' +', ' ', text)

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = self.paragraph_pattern.split(text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use sentence ending pattern
        sentences = self.sentence_endings.split(text)

        # Clean up
        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                result.append(sent)

        return result

    def _build_chunks(self, paragraphs: List[str]) -> List[str]:
        """Build chunks from paragraphs with overlap."""
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # If single paragraph exceeds chunk size, split it
            if para_length > self.chunk_size:
                # Finish current chunk if not empty
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split large paragraph into smaller chunks
                sub_chunks = self._split_large_paragraph(para)
                chunks.extend(sub_chunks)
                continue

            # Check if adding paragraph exceeds chunk size
            if current_length + para_length + 1 > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(chunk_text)
                    if overlap_text:
                        current_chunk = [overlap_text, para]
                        current_length = len(overlap_text) + para_length + 1
                    else:
                        current_chunk = [para]
                        current_length = para_length
                else:
                    current_chunk = [para]
                    current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 1

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Filter out chunks that are too short
        chunks = self._merge_short_chunks(chunks)

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph into chunks using sentences."""
        sentences = self._split_sentences(paragraph)

        if not sentences:
            # Fallback: hard split by chunk_size
            return self._hard_split(paragraph)

        return self._combine_sentences_to_chunks(sentences)

    def _combine_sentences_to_chunks(self, sentences: List[str]) -> List[str]:
        """Combine sentences into chunks respecting size limits."""
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sent_length = len(sentence)

            # If single sentence is too long, hard split it
            if sent_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                chunks.extend(self._hard_split(sentence))
                continue

            if current_length + sent_length + 1 > self.chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)

                    overlap_text = self._get_overlap_text(chunk_text)
                    if overlap_text:
                        current_chunk = [overlap_text, sentence]
                        current_length = len(overlap_text) + sent_length + 1
                    else:
                        current_chunk = [sentence]
                        current_length = sent_length
                else:
                    current_chunk = [sentence]
                    current_length = sent_length
            else:
                current_chunk.append(sentence)
                current_length += sent_length + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """Split text by character count when no natural breaks exist."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # Try to find a word boundary
            if end < text_length:
                # Look for space within last 50 chars
                space_pos = text.rfind(' ', end - 50, end)
                if space_pos > start:
                    end = space_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap if end < text_length else text_length

        return chunks

    def _get_overlap_text(self, chunk_text: str) -> str:
        """Get the overlap portion from end of chunk."""
        if len(chunk_text) <= self.chunk_overlap:
            return ""

        overlap = chunk_text[-self.chunk_overlap:]

        # Try to start at a word boundary
        space_pos = overlap.find(' ')
        if space_pos > 0:
            overlap = overlap[space_pos + 1:]

        return overlap.strip()

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too short with adjacent chunks."""
        if not chunks:
            return chunks

        result = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If current chunk is too short and there's a next chunk
            if len(current) < self.min_chunk_length and i < len(chunks) - 1:
                # Merge with next chunk
                merged = current + ' ' + chunks[i + 1]

                # If merged is too long, keep short chunk as is
                if len(merged) > self.chunk_size * 1.5:
                    result.append(current)
                else:
                    chunks[i + 1] = merged
                    i += 1
                    continue
            else:
                result.append(current)

            i += 1

        return result

    def _extract_topic(self, chunk_text: str) -> str:
        """
        Extract a topic/title from chunk text.

        Looks for headers or uses first sentence.
        """
        lines = chunk_text.split('\n')

        # Check first few lines for headers
        for line in lines[:3]:
            line = line.strip()

            # Check header patterns
            for pattern in self.header_patterns:
                if pattern.match(line):
                    return self._clean_topic(line)

            # Check if line looks like a title (short, capitalized)
            if len(line) < 100 and line and line[0].isupper():
                words = line.split()
                if len(words) <= 10:
                    # Check if it's not a regular sentence
                    if not line.endswith('.'):
                        return self._clean_topic(line)

        # Fallback: use first sentence
        first_line = lines[0].strip() if lines else ""
        sentences = self.sentence_endings.split(first_line)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) < 150:
                return self._clean_topic(first_sentence)

        # Last fallback: first N characters
        return self._clean_topic(chunk_text[:80] + "...")

    def _clean_topic(self, topic: str) -> str:
        """Clean up topic string."""
        # Remove markdown formatting
        topic = re.sub(r'#{1,6}\s*', '', topic)

        # Remove numbering
        topic = re.sub(r'^\d+\.\d*\s*', '', topic)

        # Remove trailing punctuation
        topic = topic.rstrip('.:;,')

        # Truncate if too long
        if len(topic) > 100:
            topic = topic[:97] + "..."

        return topic.strip()


class SemanticChunker(TextChunker):
    """
    Enhanced chunker that tries to maintain semantic coherence.

    Groups related content together based on section structure.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_length: int = 100,
        respect_sections: bool = True
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_length)
        self.respect_sections = respect_sections

        # Section delimiter patterns
        self.section_patterns = [
            re.compile(r'\n\s*#{1,6}\s+.+\n', re.MULTILINE),
            re.compile(r'\n\s*\d+\.\d*\s+[A-Z].+\n', re.MULTILINE),
            re.compile(r'\n\s*[A-Z][^.!?]{3,50}:\s*\n', re.MULTILINE),
        ]

    def chunk_text(
        self,
        text: str,
        chapter_title: Optional[str] = None
    ) -> List[Dict]:
        """
        Split text into semantically coherent chunks.

        Respects section boundaries when possible.
        """
        if not self.respect_sections:
            return super().chunk_text(text, chapter_title)

        # Split by sections first
        sections = self._split_by_sections(text)

        all_chunks = []
        chunk_index = 0

        for section in sections:
            section_title = section.get("title", "")
            section_text = section.get("text", "")

            if not section_text.strip():
                continue

            # Chunk the section
            section_chunks = super().chunk_text(section_text, chapter_title)

            # Update indices and add section context
            for chunk in section_chunks:
                chunk["chunk_index"] = chunk_index
                if section_title and chunk.get("topic", "").startswith(section_text[:20]):
                    chunk["topic"] = section_title
                chunk["section"] = section_title
                all_chunks.append(chunk)
                chunk_index += 1

        return all_chunks

    def _split_by_sections(self, text: str) -> List[Dict]:
        """Split text into sections based on headers."""
        sections = []

        # Find all section markers
        markers = []
        for pattern in self.section_patterns:
            for match in pattern.finditer(text):
                markers.append({
                    "start": match.start(),
                    "end": match.end(),
                    "title": match.group().strip()
                })

        # Sort by position
        markers.sort(key=lambda x: x["start"])

        if not markers:
            return [{"title": "", "text": text}]

        # Extract sections
        prev_end = 0
        for i, marker in enumerate(markers):
            # Text before this marker (belongs to previous section or intro)
            if prev_end < marker["start"]:
                intro_text = text[prev_end:marker["start"]].strip()
                if intro_text:
                    if sections:
                        sections[-1]["text"] += "\n\n" + intro_text
                    else:
                        sections.append({"title": "Introduction", "text": intro_text})

            # Determine section end
            if i < len(markers) - 1:
                section_end = markers[i + 1]["start"]
            else:
                section_end = len(text)

            section_text = text[marker["end"]:section_end].strip()
            sections.append({
                "title": marker["title"],
                "text": section_text
            })

            prev_end = section_end

        return sections
