"""Celery tasks for PDF processing."""

import asyncio
from uuid import uuid4
from datetime import datetime
import logging
import httpx
import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, SparseVector,
    VectorParams, Distance, SparseVectorParams, SparseIndexParams,
    models
)

from app.workers.celery_app import celery_app
from app.config import settings
from app.services.chunker import TextChunker

logger = logging.getLogger(__name__)

# Try to import NLTK for default processor
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download failed: {e}")


def clean_text_for_postgres(text: str) -> str:
    """Remove null bytes and other problematic characters for PostgreSQL UTF-8."""
    if not text:
        return ""
    # Remove null bytes (0x00) which PostgreSQL rejects
    cleaned = text.replace('\x00', '')
    # Also remove other control characters except newline, tab, carriage return
    cleaned = ''.join(char for char in cleaned if char >= ' ' or char in '\n\t\r')
    return cleaned


def run_async(coro):
    """Helper to run async code in Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True)
def process_pdf_task(self, file_path: str, book_name: str):
    """
    Process a PDF file: extract text, chunk, generate embeddings,
    store in Qdrant and PostgreSQL.
    """
    return run_async(_process_pdf_async(self, file_path, book_name))


async def _process_docling_sections(
    sections, book_name, book_id, pool, chunker, task, fallback_reason
):
    """
    Insert chapter records and chunk text for each Docling-extracted section.

    Args:
        sections: Output of DoclingPDFProcessor.process() — list of
                  {chapter, topic, text, is_first_in_chapter, page} dicts.
        book_name, book_id, pool, chunker, task, fallback_reason: as in caller.

    Returns:
        (all_chunks, chapter_ids, chapters_info) matching the existing fallback format.
    """

    # Group sections by chapter to create one DB chapter record per chapter title
    chapter_order = []
    chapters_by_title = {}
    for section in sections:
        title = section["chapter"]
        if title not in chapters_by_title:
            chapter_order.append(title)
            chapters_by_title[title] = []
        chapters_by_title[title].append(section)

    total_chapters = len(chapter_order)
    all_chunks = []
    chapter_ids = {}
    chapters_info = []

    async with pool.acquire() as conn:
        for i, chapter_title in enumerate(chapter_order):
            chapter_id = str(uuid4())
            chapter_ids[chapter_title] = chapter_id
            chapter_sections = chapters_by_title[chapter_title]
            start_page = chapter_sections[0].get("page") if chapter_sections else None

            await conn.execute("""
                INSERT INTO chapters (id, book_id, title, chapter_number, start_page)
                VALUES ($1, $2, $3, $4, $5)
            """, chapter_id, book_id, chapter_title, i + 1, start_page)
            chapter_chunk_index = 0

            for section in chapter_sections:
                body_text = section["text"]
                topic_override = section.get("topic", "")

                raw_chunks = chunker.chunk_text(body_text, chapter_title)

                for chunk in raw_chunks:
                    # Use the Docling sub-heading as topic when available
                    if topic_override:
                        chunk["topic"] = topic_override
                    chunk["chapter"] = chapter_title
                    chunk["chapter_id"] = chapter_id
                    chunk["book_id"] = book_id
                    chunk["book_name"] = book_name
                    chunk["page"] = section.get("page")
                    # Mark first chunk of the whole chapter as introduction
                    chunk["is_introduction"] = chapter_chunk_index == 0
                    chunk["chunk_index"] = chapter_chunk_index
                    all_chunks.append(chunk)
                    chapter_chunk_index += 1

            chapters_info.append({"title": chapter_title, "chapter_id": chapter_id})

            progress = 10 + (i / max(total_chapters, 1)) * 30
            task.update_state(state="PROCESSING", meta={
                "progress": progress,
                "stage": "chunking_docling",
                "warning": f"Using Docling fallback: {fallback_reason}",
                "chapters_total": total_chapters,
                "chapters_processed": i,
                "current_chapter": chapter_title
            })

    return all_chunks, chapter_ids, chapters_info


async def _process_pdf_async(task, file_path: str, book_name: str):
    """Async implementation of PDF processing with two-method support.

    Tries the default LLM-based processor first, falls back to Docling
    if that fails. If both fail, the job fails with an error.
    """
    # Initialize clients
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Ensure collection exists (needed on fresh installs)
    if not qdrant.collection_exists(settings.qdrant_collection):
        logger.info(f"Creating Qdrant collection '{settings.qdrant_collection}'...")
        qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000)
        )
        for field in ["book_name", "chapter_title", "topic"]:
            qdrant.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        qdrant.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="is_introduction",
            field_schema=models.PayloadSchemaType.BOOL
        )
        logger.info(f"Collection '{settings.qdrant_collection}' created successfully")

    http_client = httpx.AsyncClient(timeout=120.0)

    # Track fallback status
    use_fallback = False
    docling_succeeded = False
    fallback_reason = None

    try:
        # Create book record
        task.update_state(state="PROCESSING", meta={
            "progress": 0,
            "stage": "initializing",
            "chapters_total": 0,
            "chapters_processed": 0
        })

        async with pool.acquire() as conn:
            book_id = str(uuid4())
            await conn.execute("""
                INSERT INTO books (id, name, file_path, processing_status, created_at)
                VALUES ($1, $2, $3, 'processing', $4)
                ON CONFLICT (name) DO UPDATE SET
                    processing_status = 'processing',
                    updated_at = $4
                RETURNING id
            """, book_id, book_name, file_path, datetime.utcnow())

            # Also get existing book_id if it was updated
            existing = await conn.fetchval(
                "SELECT id FROM books WHERE name = $1", book_name
            )
            if existing:
                book_id = str(existing)

        # Try default processing first (LLM-based chapter identification)
        all_chunks = []
        chapter_ids = {}
        chapters_info = []

        try:
            from app.services.default_pdf_processor import DefaultPDFProcessor

            # Check if OpenAI API key is configured
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")

            default_processor = DefaultPDFProcessor()

            task.update_state(state="PROCESSING", meta={
                "progress": 5,
                "stage": "analyzing_chapters_llm",
                "chapters_total": 0,
                "chapters_processed": 0
            })

            # Get chapter structure using LLM
            summary_list = default_processor.get_summary_list_from_PDF(file_path, book_name)
            if not summary_list:
                raise ValueError("No chapters identified by LLM")

            total_chapters = len(summary_list)
            logger.info(f"Default processor identified {total_chapters} chapters")

            task.update_state(state="PROCESSING", meta={
                "progress": 10,
                "stage": "processing_chapters",
                "chapters_total": total_chapters,
                "chapters_processed": 0
            })

            # Process each chapter with progress updates
            from pypdf import PdfReader
            from langchain.text_splitter import NLTKTextSplitter

            reader = PdfReader(file_path)
            text_splitter = NLTKTextSplitter(
                chunk_size=settings.chunk_size,
                separator="",
                chunk_overlap=settings.chunk_overlap,
                add_start_index=True,
                use_span_tokenize=True
            )

            async with pool.acquire() as conn:
                for idx, chapter in enumerate(summary_list):
                    chapter_title = chapter["Title"]
                    chapter_id = str(uuid4())
                    chapter_ids[chapter_title] = chapter_id

                    # Create chapter record
                    await conn.execute("""
                        INSERT INTO chapters (id, book_id, title, chapter_number, start_page)
                        VALUES ($1, $2, $3, $4, $5)
                    """, chapter_id, book_id, chapter_title, idx + 1, chapter["Page"])

                    # Update progress
                    progress = 10 + (idx / total_chapters) * 30
                    task.update_state(state="PROCESSING", meta={
                        "progress": progress,
                        "stage": f"processing_chapter",
                        "chapters_total": total_chapters,
                        "chapters_processed": idx,
                        "current_chapter": chapter_title
                    })

                    # Process chapter using default processor's method
                    chapter_chunks = default_processor.process_chapter(
                        reader, chapter, idx, summary_list, book_name, text_splitter
                    )

                    for chunk in chapter_chunks:
                        chunk["chapter_id"] = chapter_id
                        chunk["book_id"] = book_id
                        all_chunks.append(chunk)

                    chapters_info.append({
                        "title": chapter_title,
                        "chapter_id": chapter_id
                    })

            logger.info(f"Default processor extracted {len(all_chunks)} chunks from {total_chapters} chapters")

        except Exception as e:
            use_fallback = True
            fallback_reason = str(e)
            logger.warning(f"Default processing failed, using fallback: {e}")

            chunker = TextChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                min_chunk_length=settings.min_chunk_length
            )

            # ---------------------------------------------------------------
            # Fallback 1: Docling layout-based processor
            # ---------------------------------------------------------------
            try:
                from app.services.docling_pdf_processor import DoclingPDFProcessor

                task.update_state(state="PROCESSING", meta={
                    "progress": 5,
                    "stage": "fallback_docling",
                    "warning": f"Using Docling fallback: {fallback_reason}",
                    "chapters_total": 0,
                    "chapters_processed": 0
                })

                docling_processor = DoclingPDFProcessor()
                raw_sections = docling_processor.process(file_path, book_name)

                if raw_sections:
                    all_chunks, chapter_ids, chapters_info = await _process_docling_sections(
                        raw_sections, book_name, book_id, pool, chunker, task, fallback_reason
                    )
                    docling_succeeded = True
                    logger.info(f"Docling fallback extracted {len(all_chunks)} chunks")

            except Exception as de:
                logger.error(f"Docling fallback also failed: {de}")

            if not docling_succeeded:
                raise RuntimeError(
                    f"All processing methods failed. Default: {fallback_reason}. "
                    f"Docling also failed to extract structure."
                )

        # Determine processing method and warning
        if not use_fallback:
            processing_method = "default"
            warning_msg = None
        else:
            processing_method = "docling"
            warning_msg = f"Using Docling fallback: {fallback_reason}"

        logger.info(f"Total chunks to embed: {len(all_chunks)}")

        # Generate embeddings in batches
        total_chapters = len(chapters_info)

        task.update_state(state="PROCESSING", meta={
            "progress": 40,
            "stage": "embedding",
            "chapters_total": total_chapters,
            "chapters_processed": total_chapters,
            "warning": warning_msg
        })
        batch_size = 16
        qdrant_points = []
        chunks_for_db = []

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]

            # Get embeddings from embedding service
            try:
                response = await http_client.post(
                    f"{settings.embedding_service_url}/embed",
                    json={"texts": texts, "return_sparse": True, "return_colbert": False}
                )
                response.raise_for_status()
                embeddings_data = response.json()
            except Exception as e:
                logger.error(f"Embedding service error: {e}")
                raise

            dense_embeddings = embeddings_data.get("dense_embeddings", [])
            sparse_embeddings = embeddings_data.get("sparse_embeddings", [])

            for j, chunk in enumerate(batch):
                if j >= len(dense_embeddings):
                    continue

                chunk_id = str(uuid4())
                qdrant_point_id = str(uuid4())

                # Prepare Qdrant point
                point_vectors = {"dense": dense_embeddings[j]}

                if j < len(sparse_embeddings) and sparse_embeddings[j]:
                    sparse_dict = sparse_embeddings[j]
                    indices = [int(k) for k in sparse_dict.keys()]
                    values = [float(v) for v in sparse_dict.values()]
                    point_vectors["sparse"] = SparseVector(indices=indices, values=values)

                qdrant_points.append(PointStruct(
                    id=qdrant_point_id,
                    vector=point_vectors,
                    payload={
                        "chunk_id": chunk_id,
                        "book_id": chunk["book_id"],
                        "book_name": chunk.get("book_name", book_name),
                        "chapter_id": chunk["chapter_id"],
                        "chapter_title": chunk.get("chapter", chunk.get("chapter_title", "")),
                        "topic": chunk.get("topic", ""),
                        "text": chunk["text"],
                        "is_introduction": chunk.get("is_introduction", False),
                        "page_number": chunk.get("page"),
                        "created_at": datetime.utcnow().isoformat()
                    }
                ))

                chunks_for_db.append({
                    "id": chunk_id,
                    "book_id": chunk["book_id"],
                    "chapter_id": chunk["chapter_id"],
                    "qdrant_point_id": qdrant_point_id,
                    "text": chunk["text"],
                    "topic": chunk.get("topic", ""),
                    "is_introduction": chunk.get("is_introduction", False),
                    "page_number": chunk.get("page"),
                    "char_count": len(chunk["text"])
                })

            progress = 40 + ((i + len(batch)) / max(len(all_chunks), 1)) * 40
            task.update_state(state="PROCESSING", meta={
                "progress": progress,
                "stage": "embedding",
                "chapters_total": total_chapters,
                "chapters_processed": total_chapters,
                "warning": warning_msg
            })

        # Store in Qdrant
        task.update_state(state="PROCESSING", meta={
            "progress": 80,
            "stage": "storing_vectors",
            "chapters_total": total_chapters,
            "chapters_processed": total_chapters,
            "warning": warning_msg
        })

        # Batch upsert to Qdrant
        batch_size = 100
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i:i + batch_size]
            qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=batch
            )

        # Store metadata in PostgreSQL
        task.update_state(state="PROCESSING", meta={
            "progress": 90,
            "stage": "storing_metadata",
            "chapters_total": total_chapters,
            "chapters_processed": total_chapters,
            "warning": warning_msg
        })

        async with pool.acquire() as conn:
            for chunk in chunks_for_db:
                # Clean text to remove null bytes and problematic characters
                clean_text = clean_text_for_postgres(chunk["text"])
                clean_topic = clean_text_for_postgres(chunk["topic"])
                await conn.execute("""
                    INSERT INTO chunks
                    (id, book_id, chapter_id, qdrant_point_id, text, topic,
                     is_introduction, page_number, char_count, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, chunk["id"], chunk["book_id"], chunk["chapter_id"],
                   chunk["qdrant_point_id"], clean_text, clean_topic,
                   chunk["is_introduction"], chunk.get("page_number"),
                   len(clean_text), datetime.utcnow())

            # Update book status
            await conn.execute("""
                UPDATE books
                SET processing_status = 'completed',
                    total_chunks = $1,
                    processed_at = $2,
                    updated_at = $2,
                    processing_method = $4
                WHERE id = $3
            """, len(chunks_for_db), datetime.utcnow(), book_id, processing_method)

            # Update chapter chunk counts
            for chapter_title, chapter_id in chapter_ids.items():
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM chunks WHERE chapter_id = $1",
                    chapter_id
                )
                await conn.execute(
                    "UPDATE chapters SET chunk_count = $1 WHERE id = $2",
                    count, chapter_id
                )

        task.update_state(state="PROCESSING", meta={
            "progress": 100,
            "stage": "complete",
            "chapters_total": total_chapters,
            "chapters_processed": total_chapters,
            "warning": warning_msg
        })

        return {
            "success": True,
            "book_id": book_id,
            "book_name": book_name,
            "chunks_processed": len(chunks_for_db),
            "chapters_processed": len(chapters_info),
            "processing_method": processing_method,
            "used_fallback": use_fallback,
            "fallback_reason": fallback_reason
        }

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")

        # Update book status to failed
        try:
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE books
                    SET processing_status = 'failed', error_message = $1, updated_at = $2
                    WHERE name = $3
                """, str(e), datetime.utcnow(), book_name)
        except Exception:
            pass

        raise

    finally:
        await pool.close()
        await http_client.aclose()
