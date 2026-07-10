"""Shared fuzzy matching utilities."""

from difflib import SequenceMatcher
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def match_book_name(
    book_name: str,
    available_books: List[str],
    threshold: float = 0.6
) -> Optional[str]:
    """
    Match an LLM-produced book name to the closest available book
    using character-level similarity.

    Args:
        book_name: The book name produced by the LLM
        available_books: List of actual book names in Qdrant
        threshold: Minimum similarity ratio (0-1) to accept a match

    Returns:
        The best matching book name, or None if no match above threshold
    """
    if not book_name or not available_books:
        return None

    # Exact match first
    if book_name in available_books:
        return book_name

    # Character-level similarity matching
    book_lower = book_name.lower()
    best_match = None
    best_ratio = 0.0

    for book in available_books:
        ratio = SequenceMatcher(None, book_lower, book.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = book

    if best_ratio >= threshold:
        logger.info(
            f"Fuzzy matched book '{book_name}' -> '{best_match}' "
            f"(similarity: {best_ratio:.2f})"
        )
        return best_match

    logger.warning(
        f"No book match found for '{book_name}' "
        f"(best candidate: '{best_match}', similarity: {best_ratio:.2f})"
    )
    return None
