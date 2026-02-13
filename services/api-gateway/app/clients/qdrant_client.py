"""Qdrant Vector Database Client."""

from typing import List, Dict, Optional, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    SparseVector, NamedSparseVector,
    SearchRequest, NamedVector,
    Prefetch, FusionQuery, Fusion,
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
        self.client = QdrantClient(host=host, port=port, timeout=60)

    def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        if self.client.collection_exists(self.collection):
            logger.info(f"Collection '{self.collection}' already exists")
            return

        logger.info(f"Creating collection '{self.collection}'...")
        self.client.create_collection(
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
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="is_introduction",
            field_schema=models.PayloadSchemaType.BOOL
        )
        logger.info(f"Collection '{self.collection}' created successfully")

    async def search_hybrid(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        limit: int = 10,
        book_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse vectors.

        Uses Reciprocal Rank Fusion (RRF) to combine results.
        """
        try:
            # Build filter if specified
            query_filter = None
            if book_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="book_name",
                            match=MatchValue(value=book_filter)
                        )
                    ]
                )

            # Build prefetch queries
            prefetch = [
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=limit * 3
                )
            ]

            # Add sparse prefetch if available
            if sparse_vector:
                sparse_indices = list(sparse_vector.keys())
                sparse_values = list(sparse_vector.values())
                prefetch.append(
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        using="sparse",
                        limit=limit * 3
                    )
                )

            # Execute hybrid search with RRF fusion
            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                query_filter=query_filter,
                with_payload=True
            )

            # Transform results
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "book_name": point.payload.get("book_name", ""),
                    "chapter_title": point.payload.get("chapter_title", ""),
                    "topic": point.payload.get("topic", ""),
                    "is_introduction": point.payload.get("is_introduction", False),
                    "chunk_id": point.payload.get("chunk_id", "")
                }
                for point in results.points
            ]

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
            # Build filter if specified
            query_filter = None
            if book_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="book_name",
                            match=MatchValue(value=book_filter)
                        )
                    ]
                )

            results = self.client.search(
                collection_name=self.collection,
                query_vector=NamedVector(name="dense", vector=vector),
                limit=limit,
                query_filter=query_filter,
                with_payload=True
            )

            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "book_name": point.payload.get("book_name", ""),
                    "chapter_title": point.payload.get("chapter_title", ""),
                    "topic": point.payload.get("topic", ""),
                    "is_introduction": point.payload.get("is_introduction", False),
                    "chunk_id": point.payload.get("chunk_id", "")
                }
                for point in results
            ]

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
                result = self.client.scroll(
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
                result = self.client.scroll(
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

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection)
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
            count_result = self.client.count(
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
            self.client.delete(
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

    def health_check(self) -> bool:
        """Check if Qdrant is healthy using collection existence check."""
        try:
            # Use collection_exists which is lighter than get_collections
            # and only checks our specific collection
            return self.client.collection_exists(self.collection)
        except Exception:
            return False

    # Snapshot Management Methods

    def create_snapshot(self) -> dict:
        """Create a snapshot of the collection."""
        try:
            result = self.client.create_snapshot(collection_name=self.collection)
            return {
                "success": True,
                "snapshot_name": result.name,
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise

    def list_snapshots(self) -> list:
        """List all available snapshots."""
        try:
            snapshots = self.client.list_snapshots(collection_name=self.collection)
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

    def restore_snapshot(self, snapshot_name: str) -> dict:
        """Restore collection from a snapshot."""
        try:
            # Qdrant expects a full URL to download the snapshot from
            # We use the snapshot download URL from Qdrant itself
            snapshot_url = f"http://{self.host}:{self.port}/collections/{self.collection}/snapshots/{snapshot_name}"

            self.client.recover_snapshot(
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

    def delete_snapshot(self, snapshot_name: str) -> dict:
        """Delete a snapshot."""
        try:
            self.client.delete_snapshot(
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
