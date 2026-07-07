"""Qdrant Vector Database Client."""

from typing import List, Dict, Optional, Any
from datetime import datetime
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    SparseVector,
    VectorParams, Distance, SparseVectorParams, SparseIndexParams,
    models
)
import logging

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(self, host: str, port: int, collection: str):
        self.host = host
        self.port = port
        self.collection = collection
        self.client = AsyncQdrantClient(host=host, port=port, timeout=60)

    async def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        if await self.client.collection_exists(self.collection):
            logger.info(f"Collection '{self.collection}' already exists")
            return

        logger.info(f"Creating collection '{self.collection}'...")
        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000)
        )

        for field in ["book_name", "chapter_title", "topic"]:
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        await self.client.create_payload_index(
            collection_name=self.collection,
            field_name="is_introduction",
            field_schema=models.PayloadSchemaType.BOOL
        )
        logger.info(f"Collection '{self.collection}' created successfully")

    @staticmethod
    def _build_book_filter(book_filter: Optional[str]) -> Optional[Filter]:
        if not book_filter:
            return None
        return Filter(
            must=[FieldCondition(key="book_name", match=MatchValue(value=book_filter))]
        )

    @staticmethod
    def _format_point(point, score: float) -> Dict[str, Any]:
        payload = point.payload or {}
        return {
            "id": str(point.id),
            "score": score,
            "text": payload.get("text", ""),
            "book_name": payload.get("book_name", ""),
            "chapter_title": payload.get("chapter_title", ""),
            "chapter_id": payload.get("chapter_id", ""),
            "topic": payload.get("topic", ""),
            "is_introduction": payload.get("is_introduction", False),
            "chunk_id": payload.get("chunk_id", ""),
            "chunk_index": payload.get("chunk_index"),
            "page_number": payload.get("page_number"),
        }

    @staticmethod
    def _normalize_scores(points) -> Dict[str, float]:
        """Min-max normalize point scores to [0, 1], keyed by point id."""
        if not points:
            return {}
        scores = [p.score for p in points]
        lo, hi = min(scores), max(scores)
        rng = hi - lo
        return {
            str(p.id): ((p.score - lo) / rng if rng > 0 else 1.0)
            for p in points
        }

    async def search_hybrid(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        limit: int = 10,
        book_filter: Optional[str] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Hybrid search with weighted score fusion.

        Dense and sparse are queried separately and their scores min-max
        normalized (cosine and sparse-dot scores aren't on the same scale),
        then combined as dense_weight*dense + sparse_weight*sparse.
        """
        try:
            query_filter = self._build_book_filter(book_filter)
            oversample = limit * 3

            dense_points = (await self.client.query_points(
                collection_name=self.collection,
                query=dense_vector,
                using="dense",
                limit=oversample,
                query_filter=query_filter,
                with_payload=True,
            )).points

            sparse_points = []
            if sparse_vector:
                sparse_points = (await self.client.query_points(
                    collection_name=self.collection,
                    query=SparseVector(
                        indices=list(sparse_vector.keys()),
                        values=list(sparse_vector.values()),
                    ),
                    using="sparse",
                    limit=oversample,
                    query_filter=query_filter,
                    with_payload=True,
                )).points

            dense_norm = self._normalize_scores(dense_points)
            sparse_norm = self._normalize_scores(sparse_points)

            points_by_id = {str(p.id): p for p in [*dense_points, *sparse_points]}
            fused = [
                (
                    dense_weight * dense_norm.get(pid, 0.0)
                    + sparse_weight * sparse_norm.get(pid, 0.0),
                    point,
                )
                for pid, point in points_by_id.items()
            ]
            fused.sort(key=lambda item: item[0], reverse=True)

            return [self._format_point(point, score) for score, point in fused[:limit]]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    async def search_dense(
        self,
        vector: List[float],
        limit: int = 10,
        book_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform dense-only vector search.
        """
        try:
            results = (await self.client.query_points(
                collection_name=self.collection,
                query=vector,
                using="dense",
                limit=limit,
                query_filter=self._build_book_filter(book_filter),
                with_payload=True
            )).points
            return [self._format_point(point, point.score) for point in results]

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            raise

    async def get_books(self) -> List[str]:
        """Get list of unique book names in the collection."""
        try:
            # Scroll through all points to get unique book names
            books = set()
            offset = None

            while True:
                result = await self.client.scroll(
                    collection_name=self.collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["book_name"]
                )

                points, next_offset = result

                for point in points:
                    if point.payload and "book_name" in point.payload:
                        books.add(point.payload["book_name"])

                if next_offset is None:
                    break
                offset = next_offset

            return sorted(list(books))

        except Exception as e:
            logger.error(f"Failed to get books: {e}")
            return []

    async def get_books_with_chapters(self) -> List[Dict[str, Any]]:
        """Get list of books with their chapters from Qdrant payloads."""
        try:
            # Scroll through all points to collect book/chapter info
            book_chapters: Dict[str, set] = {}
            offset = None

            while True:
                result = await self.client.scroll(
                    collection_name=self.collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["book_name", "chapter_title"]
                )

                points, next_offset = result

                for point in points:
                    if point.payload:
                        book_name = point.payload.get("book_name", "")
                        chapter_title = point.payload.get("chapter_title", "")

                        if book_name:
                            if book_name not in book_chapters:
                                book_chapters[book_name] = set()
                            if chapter_title:
                                book_chapters[book_name].add(chapter_title)

                if next_offset is None:
                    break
                offset = next_offset

            # Format result
            books = []
            for book_name in sorted(book_chapters.keys()):
                chapters = sorted(list(book_chapters[book_name]))
                books.append({
                    "name": book_name,
                    "chapters": [{"title": ch} for ch in chapters]
                })

            return books

        except Exception as e:
            logger.error(f"Failed to get books with chapters: {e}")
            return []

    async def get_library_map(self) -> Dict[str, Dict[str, List[str]]]:
        """One-scroll catalogue: book -> chapter -> sorted distinct topics.

        Backs list_chapters and list_topics with no extra DB calls.
        """
        try:
            tree: Dict[str, Dict[str, set]] = {}
            offset = None

            while True:
                points, offset = await self.client.scroll(
                    collection_name=self.collection,
                    limit=1000,
                    offset=offset,
                    with_payload=["book_name", "chapter_title", "topic"],
                )

                for point in points:
                    payload = point.payload or {}
                    book = payload.get("book_name", "")
                    if not book:
                        continue
                    chapter = payload.get("chapter_title", "")
                    topic = payload.get("topic", "")
                    chapters = tree.setdefault(book, {})
                    topics = chapters.setdefault(chapter, set())
                    if topic:
                        topics.add(topic)

                if offset is None:
                    break

            return {
                book: {ch: sorted(tps) for ch, tps in chapters.items()}
                for book, chapters in tree.items()
            }

        except Exception as e:
            logger.error(f"Failed to build library map: {e}")
            return {}

    async def get_chapter_chunks(
        self, book_name: str, chapter_title: str, intro_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch a chapter's chunks, ordered, for read_chapter.

        For intro_only, returns the is_introduction chunk(s); when none are
        tagged (older ingests), falls back to the chapter's first two chunks.
        """
        try:
            all_chunks = self._sort_chunks(await self._scroll_chapter(book_name, chapter_title))
            if not intro_only:
                return all_chunks

            intro = [c for c in all_chunks if c.get("is_introduction")]
            if intro:
                return intro
            return all_chunks[:2]

        except Exception as e:
            logger.error(f"Failed to get chapter chunks: {e}")
            return []

    async def _scroll_chapter(self, book_name: str, chapter_title: str) -> List[Dict[str, Any]]:
        """Scroll all chunks of one chapter."""
        chunks: List[Dict[str, Any]] = []
        offset = None
        while True:
            points, offset = await self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="book_name", match=MatchValue(value=book_name)),
                        FieldCondition(key="chapter_title", match=MatchValue(value=chapter_title)),
                    ]
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
            )
            chunks.extend(self._format_point(p, 1.0) for p in points)
            if offset is None:
                break
        return chunks

    async def get_adjacent_chunks(
        self, chapter_id: str, chunk_index: Optional[int], window: int = 1
    ) -> List[Dict[str, Any]]:
        """Fetch chunks adjacent to a hit within the same chapter, for expand_context.

        Uses chunk_index when present; falls back to page_number proximity for
        books indexed before chunk_index was added to the payload.
        """
        if not chapter_id:
            return []
        try:
            chunks: List[Dict[str, Any]] = []
            offset = None
            while True:
                points, offset = await self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="chapter_id", match=MatchValue(value=chapter_id))]
                    ),
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                )
                chunks.extend(self._format_point(p, 1.0) for p in points)
                if offset is None:
                    break

            chunks = self._sort_chunks(chunks)

            if chunk_index is not None and any(c.get("chunk_index") is not None for c in chunks):
                lo, hi = chunk_index - window, chunk_index + window
                return [
                    c for c in chunks
                    if c.get("chunk_index") is not None and lo <= c["chunk_index"] <= hi
                ]

            # Fallback: nearest neighbours by list position around the same page.
            return chunks

        except Exception as e:
            logger.error(f"Failed to get adjacent chunks: {e}")
            return []

    @staticmethod
    def _sort_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order chunks by chunk_index, then page_number."""
        return sorted(
            chunks,
            key=lambda c: (
                c.get("chunk_index") if c.get("chunk_index") is not None else 1_000_000,
                c.get("page_number") if c.get("page_number") is not None else 1_000_000,
            ),
        )

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = await self.client.get_collection(self.collection)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def delete_by_book_name(self, book_name: str) -> int:
        """
        Delete all points belonging to a specific book.

        Args:
            book_name: The name of the book to delete

        Returns:
            Number of points deleted
        """
        try:
            # First, count how many points will be deleted
            count_result = await self.client.count(
                collection_name=self.collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="book_name",
                            match=MatchValue(value=book_name)
                        )
                    ]
                )
            )
            points_to_delete = count_result.count

            if points_to_delete == 0:
                return 0

            # Delete all points matching the book name
            await self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="book_name",
                            match=MatchValue(value=book_name)
                        )
                    ]
                )
            )

            logger.info(f"Deleted {points_to_delete} points for book: {book_name}")
            return points_to_delete

        except Exception as e:
            logger.error(f"Failed to delete book {book_name}: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy using collection existence check."""
        try:
            # Use collection_exists which is lighter than get_collections
            # and only checks our specific collection
            return await self.client.collection_exists(self.collection)
        except Exception:
            return False

    # Snapshot Management Methods

    async def create_snapshot(self) -> dict:
        """Create a snapshot of the collection."""
        try:
            result = await self.client.create_snapshot(collection_name=self.collection)
            return {
                "success": True,
                "snapshot_name": result.name,
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    async def list_snapshots(self) -> list:
        """List all available snapshots."""
        try:
            snapshots = await self.client.list_snapshots(collection_name=self.collection)
            return [
                {
                    "name": snap.name,
                    "size": snap.size,
                    "created_at": snap.creation_time
                }
                for snap in snapshots
            ]
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            raise

    async def restore_snapshot(self, snapshot_name: str) -> dict:
        """Restore collection from a snapshot."""
        try:
            # Qdrant expects a full URL to download the snapshot from
            # We use the snapshot download URL from Qdrant itself
            snapshot_url = f"http://{self.host}:{self.port}/collections/{self.collection}/snapshots/{snapshot_name}"

            await self.client.recover_snapshot(
                collection_name=self.collection,
                location=snapshot_url
            )
            return {
                "success": True,
                "message": f"Restored from snapshot: {snapshot_name}"
            }
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            raise

    async def delete_snapshot(self, snapshot_name: str) -> dict:
        """Delete a snapshot."""
        try:
            await self.client.delete_snapshot(
                collection_name=self.collection,
                snapshot_name=snapshot_name
            )
            return {
                "success": True,
                "message": f"Deleted snapshot: {snapshot_name}"
            }
        except Exception as e:
            logger.error(f"Failed to delete snapshot: {e}")
            raise

    def get_snapshot_url(self, snapshot_name: str) -> str:
        """Get the download URL for a snapshot."""
        # Qdrant snapshot download URL format
        return f"http://{self.host}:{self.port}/collections/{self.collection}/snapshots/{snapshot_name}"
