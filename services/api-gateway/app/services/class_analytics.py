"""Per-class topic analytics, shared by the professor and manager routers."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import HTTPException

UNCLASSIFIED = "unclassified"


def _parse_range(date_from: Optional[str], date_to: Optional[str]):
    """ISO dates → [start, end) datetimes; `to` is inclusive (whole day)."""
    try:
        start = datetime.fromisoformat(date_from) if date_from else None
        end = datetime.fromisoformat(date_to) + timedelta(days=1) if date_to else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be ISO format (YYYY-MM-DD)")
    return start, end


def _range_clause(start, end, params: list, column: str = "mt.created_at") -> str:
    clause = ""
    if start:
        params.append(start)
        clause += f" AND {column} >= ${len(params)}"
    if end:
        params.append(end)
        clause += f" AND {column} < ${len(params)}"
    return clause


async def topic_ranking(
    db_pool, class_id: str,
    date_from: Optional[str] = None, date_to: Optional[str] = None,
) -> dict:
    """Query counts per topic/subtopic (subtopics rolled up into parents),
    plus the unclassified bucket. Topics with zero queries are included."""
    start, end = _parse_range(date_from, date_to)

    params: list = [UUID(class_id)]
    range_sql = _range_clause(start, end, params)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT t.id, t.name, t.parent_topic_id, t.position,
                   COUNT(mt.message_id) AS query_count
            FROM class_topics t
            LEFT JOIN message_topics mt
                ON mt.topic_id = t.id {range_sql}
            WHERE t.class_id = $1
            GROUP BY t.id
            ORDER BY t.position, t.name
            """,
            *params,
        )

        unclassified_params: list = [UUID(class_id)]
        unclassified_sql = _range_clause(start, end, unclassified_params)
        unclassified = await conn.fetchval(
            f"""
            SELECT COUNT(*) FROM message_topics mt
            WHERE mt.class_id = $1 AND mt.topic_id IS NULL {unclassified_sql}
            """,
            *unclassified_params,
        )

    topics = [
        {
            "id": str(r["id"]),
            "name": r["name"],
            "count": r["query_count"],
            "subtopics": [],
        }
        for r in rows if r["parent_topic_id"] is None
    ]
    by_id = {t["id"]: t for t in topics}
    for r in rows:
        if r["parent_topic_id"] is not None:
            parent = by_id.get(str(r["parent_topic_id"]))
            if parent:
                parent["subtopics"].append({
                    "id": str(r["id"]),
                    "name": r["name"],
                    "count": r["query_count"],
                })
                parent["count"] += r["query_count"]

    topics.sort(key=lambda t: t["count"], reverse=True)
    total = sum(t["count"] for t in topics) + unclassified

    return {
        "topics": topics,
        "unclassified": unclassified,
        "total": total,
        "from": date_from,
        "to": date_to,
    }


async def topic_queries(
    db_pool, class_id: str, topic_id: str,
    date_from: Optional[str] = None, date_to: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """The raw student queries assigned to one topic (or 'unclassified').

    A parent topic includes its subtopics' queries.
    """
    start, end = _parse_range(date_from, date_to)
    params: list = [UUID(class_id)]

    if topic_id == UNCLASSIFIED:
        topic_sql = "AND mt.topic_id IS NULL"
        name = UNCLASSIFIED
    else:
        try:
            topic_uuid = UUID(topic_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=404, detail="Topic not found")
        async with db_pool.acquire() as conn:
            name = await conn.fetchval(
                "SELECT name FROM class_topics WHERE id = $1 AND class_id = $2",
                topic_uuid, UUID(class_id),
            )
        if not name:
            raise HTTPException(status_code=404, detail="Topic not found")
        params.append(topic_uuid)
        topic_sql = (
            f"AND (mt.topic_id = ${len(params)} OR mt.topic_id IN "
            f"(SELECT id FROM class_topics WHERE parent_topic_id = ${len(params)}))"
        )

    range_sql = _range_clause(start, end, params)
    params.append(limit)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT m.content, m.created_at, mt.similarity
            FROM message_topics mt
            JOIN messages m ON m.id = mt.message_id
            WHERE mt.class_id = $1 {topic_sql} {range_sql}
            ORDER BY m.created_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )

    return {
        "topic": name,
        "queries": [
            {
                "content": r["content"],
                "created_at": r["created_at"].isoformat(),
                "similarity": r["similarity"],
            }
            for r in rows
        ],
    }
