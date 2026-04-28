from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.models.dashboard import DashboardMode, DataSource
from app.services.fetchers import NewsArticle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RAW_EXAMPLES_PATH = PROJECT_ROOT / "data" / "raw" / "news_examples.jsonl"


def save_news_examples(
    query: str,
    detected_mode: DashboardMode,
    articles: list[NewsArticle],
    data_source: DataSource,
    output_path: Path = DEFAULT_RAW_EXAMPLES_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()

    with output_path.open("a", encoding="utf-8") as file:
        for article in articles:
            file.write(
                json.dumps(
                    _example_row(
                        query=query,
                        detected_mode=detected_mode,
                        article=article,
                        data_source=data_source,
                        created_at=created_at,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )


def _example_row(
    query: str,
    detected_mode: DashboardMode,
    article: NewsArticle,
    data_source: DataSource,
    created_at: str,
) -> dict[str, object]:
    return {
        "query": query,
        "detected_mode": detected_mode.value,
        "title": article.title,
        "snippet": article.snippet,
        "source": article.source,
        "published_at": (
            article.published_at.isoformat() if article.published_at else None
        ),
        "url": article.url,
        "data_source": data_source.value,
        "created_at": created_at,
        "sentiment_label": "",
        "event_tags_label": [],
        "importance_label": "",
        "impact_label": "",
    }
