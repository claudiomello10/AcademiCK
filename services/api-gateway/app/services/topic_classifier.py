"""Async query-to-topic classification via embedding similarity.

Runs after the chat response is already on its way to the student —
a failure here logs and drops the data point, never breaks chat.
"""

import json
import logging
import math
from uuid import UUID

from app.config import settings

logger = logging.getLogger(__name__)


def embedding_text(name: str, description: str | None) -> str:
    """The text embedded for a topic; queries are matched against it."""
    return f"{name} — {description}" if description else name


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def classify_query(
    db_pool, embedding_client, class_id: str, message_id: str, text: str
) -> None:
    """Assign the best-matching class topic to a stored user message.

    Below-threshold or no-topic classes record a NULL topic (unclassified),
    so 'students asking off-topic questions' is itself visible in analytics.
    """
    try:
        if not settings.topic_classification_enabled:
            return
        if not (class_id and message_id and text):
            return

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, embedding FROM class_topics
                WHERE class_id = $1 AND embedding IS NOT NULL
                """,
                UUID(class_id),
            )

        topic_id = None
        similarity = None

        if rows:
            result = await embedding_client.embed_batch([text], return_sparse=False)
            query_vector = result["dense_embeddings"][0]

            best_id, best_sim = None, -1.0
            for row in rows:
                vector = row["embedding"]
                if isinstance(vector, str):
                    vector = json.loads(vector)
                sim = cosine(query_vector, vector)
                if sim > best_sim:
                    best_id, best_sim = row["id"], sim

            similarity = best_sim
            if best_sim >= settings.topic_similarity_threshold:
                topic_id = best_id

        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO message_topics (message_id, class_id, topic_id, similarity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (message_id) DO NOTHING
                """,
                UUID(message_id), UUID(class_id), topic_id, similarity,
            )

    except Exception as e:
        logger.warning(f"Topic classification failed for message {message_id}: {e}")
