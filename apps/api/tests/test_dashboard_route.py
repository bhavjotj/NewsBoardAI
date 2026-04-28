from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.main import app
from app.models.dashboard import DashboardRequest, DataSource
from app.routes.news import create_dashboard
from app.services.fetchers import NewsArticle


def test_dashboard_route_returns_compact_response() -> None:
    response = create_dashboard(
        DashboardRequest(query="Tesla", max_results=3, mode="business")
    )

    body = response.model_dump()
    assert body["topic"] == "Tesla"
    assert body["data_source"] == DataSource.MOCK
    assert body["time_window"] == "Recent news"
    assert body["brief"]
    assert "source cards" not in body["brief"]
    assert body["sentiment"]["label"] in {"positive", "neutral", "negative", "mixed"}
    assert "market" in body["event_tags"]
    assert len(body["sources"]) == 3
    assert body["confidence"] in {"low", "medium", "high"}
    assert body["possible_impact"]


def test_dashboard_route_uses_request_defaults() -> None:
    response = create_dashboard(DashboardRequest(query="Apple"))

    body = response.model_dump()
    assert len(body["sources"]) == 5
    assert body["data_source"] == DataSource.MOCK


def test_dashboard_route_uses_mocked_google_news(monkeypatch) -> None:
    def fake_fetch_google_news_rss(query: str, max_results: int):
        return [
            NewsArticle(
                title=f"{query} real headline",
                source="Example Publisher",
                published_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
                snippet="Real RSS result mentions launch and growth.",
                url="https://example.com/real-news",
            )
        ][:max_results]

    monkeypatch.setattr(
        "app.routes.news.fetch_google_news_rss", fake_fetch_google_news_rss
    )

    response = create_dashboard(
        DashboardRequest(query="Nintendo", max_results=1, use_real_news=True)
    )

    assert response.data_source == DataSource.GOOGLE_NEWS_RSS
    assert len(response.sources) == 1
    assert response.sources[0].source == "Example Publisher"
    assert response.sources[0].url.unicode_string() == "https://example.com/real-news"


def test_dashboard_route_falls_back_when_google_news_fails(monkeypatch) -> None:
    def failing_fetch_google_news_rss(query: str, max_results: int):
        raise TimeoutError("rss unavailable")

    monkeypatch.setattr(
        "app.routes.news.fetch_google_news_rss", failing_fetch_google_news_rss
    )

    response = create_dashboard(
        DashboardRequest(query="PlayStation", max_results=2, use_real_news=True)
    )

    assert response.data_source == DataSource.FALLBACK_MOCK
    assert len(response.sources) == 2
    assert all(source.source.startswith("Mock") for source in response.sources)


def test_dashboard_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        DashboardRequest(query="   ")


def test_dashboard_post_route_is_registered() -> None:
    matching_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/api/news/dashboard"
    ]

    assert matching_routes
    assert "POST" in matching_routes[0].methods
