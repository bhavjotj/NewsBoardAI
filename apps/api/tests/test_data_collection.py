import json
from datetime import datetime, timezone

from app.models.dashboard import DashboardMode, DataSource
from app.services.data_collection import save_news_examples
from app.services.fetchers import NewsArticle


def test_save_news_examples_writes_jsonl_rows(tmp_path) -> None:
    output_path = tmp_path / "raw" / "news_examples.jsonl"
    articles = [
        NewsArticle(
            title="Netflix shares rise after earnings",
            source="Example News",
            published_at=datetime(2026, 4, 28, 12, 30, tzinfo=timezone.utc),
            snippet="Revenue growth and pricing updates drew investor attention.",
            url="https://example.com/netflix",
        )
    ]

    save_news_examples(
        query="netflix",
        detected_mode=DashboardMode.BUSINESS,
        articles=articles,
        data_source=DataSource.GOOGLE_NEWS_RSS,
        output_path=output_path,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 1
    assert rows[0]["query"] == "netflix"
    assert rows[0]["detected_mode"] == "business"
    assert rows[0]["title"] == "Netflix shares rise after earnings"
    assert rows[0]["snippet"] == "Revenue growth and pricing updates drew investor attention."
    assert rows[0]["source"] == "Example News"
    assert rows[0]["published_at"] == "2026-04-28T12:30:00+00:00"
    assert rows[0]["url"] == "https://example.com/netflix"
    assert rows[0]["data_source"] == "google_news_rss"
    assert rows[0]["created_at"]
    assert rows[0]["sentiment_label"] == ""
    assert rows[0]["event_tags_label"] == []
    assert rows[0]["importance_label"] == ""
    assert rows[0]["impact_label"] == ""
