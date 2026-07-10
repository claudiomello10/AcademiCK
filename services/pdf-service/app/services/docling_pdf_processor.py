"""Docling-based PDF processor for structure-aware text extraction."""

import ast
import logging
import re
from typing import List, Dict, Set

from app.config import settings
from app.services.llm_client import build_chat_client

logger = logging.getLogger(__name__)


class DoclingPDFProcessor:
    """
    Uses Docling layout analysis to extract heading-based chapter/section
    structure from PDFs that lack a machine-readable TOC.

    Returns a list of sections, each with a chapter title, optional topic
    (sub-heading), and the raw body text. The caller is responsible for
    chunking the text via TextChunker.

    Processing order in tasks.py:
        1. DefaultPDFProcessor  (LLM-based, primary)
        2. DoclingPDFProcessor  (layout-based, this class — fallback)

    Returns [] when no headings are detected, which causes the job to fail.
    """

    def __init__(self, llm_model: str = settings.pdf_chapter_detection_model):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        self.model = llm_model

        # Use the standard pipeline (layout model enabled) so section headings are
        # correctly identified even in multi-column PDFs.
        # OCR is disabled since we only process text-native PDFs here — scanned PDFs
        # will return no sections and the job will fail with an error.
        # Table structure is disabled to reduce compute cost.
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def process(self, file_path: str, book_name: str) -> List[Dict]:
        """
        Convert a PDF and group its content under heading-based sections.

        Args:
            file_path: Absolute path to the PDF file.
            book_name: Display name of the book (used as fallback chapter title).

        Returns:
            List of section dicts:
                {
                    "chapter":             str,       # top-level heading text
                    "topic":               str,       # sub-heading text (empty string if none)
                    "text":                str,       # raw body text for this section
                    "is_first_in_chapter": bool,
                    "page":                int|None   # 1-based page number of the heading
                }
            Empty list if Docling finds no headings (causes the job to fail).
        """
        try:
            result = self._converter.convert(file_path)
        except Exception as e:
            logger.error(f"Docling conversion failed for {file_path}: {e}")
            return []

        doc = result.document
        raw_sections = self._extract_sections(doc)

        if not raw_sections:
            logger.info(f"Docling found no headings in {file_path}, processing will fail")
            return []

        # Try LLM-based hierarchy classification; fall back to flat if it fails
        headings = [s["heading"] for s in raw_sections if s["heading"]]
        chapter_headings = self._classify_headings_with_llm(headings, book_name)

        sections = self._build_output(raw_sections, book_name, chapter_headings)
        logger.info(f"Docling extracted {len(sections)} sections from {file_path}")
        return sections

    def _extract_sections(self, doc) -> List[Dict]:
        """
        Walk the Docling document and collect heading + paragraph sequences.

        Returns a flat list of raw section dicts:
            {"heading": str, "level": int, "paragraphs": [str]}
        """
        try:
            from docling_core.types.doc import DocItemLabel
        except ImportError:
            from docling.datamodel.base_models import DocItemLabel  # older versions

        sections = []
        current: Dict | None = None

        for item, level in doc.iterate_items():
            label = item.label

            if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
                text = (item.text or "").strip()
                if not text:
                    continue
                # Save previous section
                if current is not None:
                    sections.append(current)
                page_no = item.prov[0].page_no if item.prov else None
                current = {"heading": text, "level": level, "paragraphs": [], "page": page_no}

            elif label in (DocItemLabel.PARAGRAPH, DocItemLabel.TEXT, DocItemLabel.LIST_ITEM):
                text = (item.text or "").strip()
                if text:
                    if current is None:
                        # Text before any heading — create an implicit intro section
                        current = {"heading": "", "level": 1, "paragraphs": []}
                    current["paragraphs"].append(text)

        # Flush last section
        if current is not None:
            sections.append(current)

        return sections

    def _classify_headings_with_llm(self, headings: List[str], book_name: str) -> Set[str]:
        """
        Ask the LLM to identify which headings are top-level chapters.

        Returns a set of heading strings that should be treated as chapters.
        On any failure (no API key, network error, bad response), returns
        the full set of headings so every heading becomes a chapter (flat fallback).
        """
        try:
            client, model = build_chat_client(self.model)

            headings_text = "\n".join(f"- {h}" for h in headings)
            prompt = (
                f"The following is a list of headings extracted from an academic document titled '{book_name}'.\n"
                f"Identify which headings are TOP-LEVEL chapters or major sections (not sub-sections).\n"
                f"Sub-sections are headings that belong under a chapter, like '3.1 Architecture' under '3. Method'.\n"
                f"Return ONLY a Python list of the exact heading strings that are top-level chapters.\n"
                f"The response must be a valid Python list and nothing else. Example: ['Abstract', '1. Introduction']\n\n"
                f"Headings:\n{headings_text}"
            )

            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = completion.choices[0].message.content.strip()

            # Parse the response
            try:
                chapter_list = ast.literal_eval(answer)
            except (ValueError, SyntaxError):
                match = re.search(r'\[.*?\]', answer, re.DOTALL)
                if match:
                    chapter_list = ast.literal_eval(match.group())
                else:
                    raise ValueError(f"Could not parse LLM response: {answer!r}")

            if not isinstance(chapter_list, list):
                raise ValueError("LLM response was not a list")

            chapter_set = set(chapter_list)
            logger.info(f"LLM classified {len(chapter_set)} of {len(headings)} headings as chapters")
            return chapter_set

        except Exception as e:
            logger.warning(f"Heading classification failed, using flat structure: {e}")
            # Flat fallback: every heading is a chapter
            return set(headings)

    def _build_output(
        self, raw_sections: List[Dict], book_name: str, chapter_headings: Set[str]
    ) -> List[Dict]:
        """
        Convert raw sections into the output format consumed by tasks.py.

        Headings in chapter_headings → top-level chapters.
        All other headings → topics within the current chapter.
        Sections with no heading use book_name as chapter title.
        """
        if not raw_sections:
            return []

        output = []
        current_chapter = book_name
        current_chapter_page = None
        chapter_section_index = 0

        for section in raw_sections:
            heading = section["heading"]
            page = section.get("page")
            body = "\n\n".join(section["paragraphs"]).strip()

            if not body:
                # No text content — update chapter tracking but emit nothing
                if heading and heading in chapter_headings:
                    current_chapter = heading
                    current_chapter_page = page
                    chapter_section_index = 0
                continue

            if not heading:
                # Pre-heading text: assign to current chapter with no topic
                output.append({
                    "chapter": current_chapter,
                    "topic": "",
                    "text": body,
                    "page": current_chapter_page,
                    "is_first_in_chapter": chapter_section_index == 0,
                })
                chapter_section_index += 1

            elif heading in chapter_headings:
                # New top-level chapter
                current_chapter = heading
                current_chapter_page = page
                chapter_section_index = 0
                output.append({
                    "chapter": current_chapter,
                    "topic": "",
                    "text": body,
                    "page": page,
                    "is_first_in_chapter": True,
                })
                chapter_section_index += 1

            else:
                # Sub-heading → topic within the current chapter
                output.append({
                    "chapter": current_chapter,
                    "topic": heading,
                    "text": body,
                    "page": page,
                    "is_first_in_chapter": chapter_section_index == 0,
                })
                chapter_section_index += 1

        return output
