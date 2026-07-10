"""On-demand LLM digest of a class's topic analytics (Portuguese).

Gated by ANALYTICS_SUMMARY_ENABLED; the date range is chosen by the
professor per request — nothing here is scheduled.
"""

import logging
from typing import Optional

from pydantic_ai import Agent

from app.config import settings
from app.services import class_analytics
from app.services.agent_models import build_model

logger = logging.getLogger(__name__)

SAMPLE_QUERIES_PER_TOPIC = 8

SYSTEM_PROMPT = (
    "Você é um assistente pedagógico. A partir das estatísticas de perguntas "
    "dos alunos de uma turma, escreva um resumo curto em português para o "
    "professor: onde os alunos estão com mais dificuldade, possíveis "
    "equívocos recorrentes visíveis nas perguntas, e sugestões práticas de "
    "reforço. Seja específico e baseie-se apenas nos dados fornecidos. "
    "Use no máximo 300 palavras."
)


async def generate_summary(
    db_pool, class_id: str, subject: str,
    date_from: Optional[str] = None, date_to: Optional[str] = None,
) -> dict:
    ranking = await class_analytics.topic_ranking(db_pool, class_id, date_from, date_to)

    sections = [f"Disciplina: {subject}", f"Total de perguntas no período: {ranking['total']}"]
    for topic in ranking["topics"]:
        sections.append(f"\n## {topic['name']} — {topic['count']} pergunta(s)")
        for sub in topic["subtopics"]:
            sections.append(f"  - {sub['name']}: {sub['count']}")
        queries = await class_analytics.topic_queries(
            db_pool, class_id, topic["id"], date_from, date_to,
            limit=SAMPLE_QUERIES_PER_TOPIC,
        )
        for q in queries["queries"]:
            sections.append(f'  Pergunta: "{q["content"]}"')

    unclassified = await class_analytics.topic_queries(
        db_pool, class_id, class_analytics.UNCLASSIFIED, date_from, date_to,
        limit=SAMPLE_QUERIES_PER_TOPIC,
    )
    sections.append(f"\n## Não classificadas — {ranking['unclassified']} pergunta(s)")
    for q in unclassified["queries"]:
        sections.append(f'  Pergunta: "{q["content"]}"')

    agent = Agent(
        model=build_model(settings.analytics_summary_model),
        output_type=str,
        system_prompt=SYSTEM_PROMPT,
    )
    result = await agent.run("\n".join(sections))

    return {
        "summary": result.output,
        "from": date_from,
        "to": date_to,
        "total_queries": ranking["total"],
        "model": settings.analytics_summary_model,
    }
