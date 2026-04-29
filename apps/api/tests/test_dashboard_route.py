from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.main import app
from app.models.dashboard import (
    AnalysisSource,
    DashboardMode,
    DashboardRequest,
    DataSource,
)
from app.routes.news import create_dashboard
from app.services.analyzer import DashboardAnalysis
from app.services.fetchers import NewsArticle
from app.services.hybrid_analyzer import HybridAnalysisResult


def test_dashboard_route_returns_compact_response() -> None:
    response = create_dashboard(
        DashboardRequest(
            query="Tesla",
            max_results=3,
            mode="business",
            use_real_news=False,
            use_ml=False,
        )
    )

    body = response.model_dump()
    assert body["topic"] == "Tesla"
    assert body["data_source"] == DataSource.MOCK
    assert body["analysis_source"] == AnalysisSource.RULE_BASED
    assert body["detected_mode"] == DashboardMode.BUSINESS
    assert body["time_window"] == "Recent news"
    assert body["brief"]
    assert "source cards" not in body["brief"]
    assert body["sentiment"]["label"] in {"positive", "neutral", "negative", "mixed"}
    assert "market" in body["event_tags"]
    assert len(body["sources"]) == 3
    assert body["confidence"] in {"low", "medium", "high"}
    assert body["possible_impact"]


def test_dashboard_route_uses_real_news_by_default(monkeypatch) -> None:
    def fake_fetch_google_news_rss(query: str, max_results: int):
        return [
            NewsArticle(
                title=f"{query} shares rise after earnings",
                source="Example Publisher",
                published_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
                snippet="Revenue growth and analyst upgrades are drawing investor attention.",
                url="https://example.com/default-real-news",
            )
        ][:max_results]

    monkeypatch.setattr(
        "app.routes.news.fetch_google_news_rss", fake_fetch_google_news_rss
    )

    response = create_dashboard(DashboardRequest(query="Apple", use_ml=False))

    body = response.model_dump()
    assert len(body["sources"]) == 1
    assert body["data_source"] == DataSource.GOOGLE_NEWS_RSS
    assert body["analysis_source"] == AnalysisSource.RULE_BASED
    assert body["detected_mode"] == DashboardMode.BUSINESS


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
        DashboardRequest(query="Nintendo", max_results=1, use_ml=False)
    )

    assert response.data_source == DataSource.GOOGLE_NEWS_RSS
    assert response.analysis_source == AnalysisSource.RULE_BASED
    assert response.detected_mode == DashboardMode.GAMING
    assert len(response.sources) == 1
    assert response.sources[0].source == "Example Publisher"
    assert response.sources[0].url.unicode_string() == "https://example.com/real-news"


def test_dashboard_route_saves_real_news_examples(monkeypatch) -> None:
    saved_calls = []

    def fake_fetch_google_news_rss(query: str, max_results: int):
        return [
            NewsArticle(
                title="Netflix shares rise after earnings",
                source="Example Publisher",
                published_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
                snippet="Revenue growth drew investor attention.",
                url="https://example.com/netflix",
            )
        ][:max_results]

    def fake_save_news_examples(**kwargs):
        saved_calls.append(kwargs)

    monkeypatch.setattr(
        "app.routes.news.fetch_google_news_rss", fake_fetch_google_news_rss
    )
    monkeypatch.setattr("app.routes.news.save_news_examples", fake_save_news_examples)

    response = create_dashboard(
        DashboardRequest(
            query="Netflix",
            max_results=1,
            save_examples=True,
            use_ml=False,
        )
    )

    assert response.data_source == DataSource.GOOGLE_NEWS_RSS
    assert response.analysis_source == AnalysisSource.RULE_BASED
    assert len(saved_calls) == 1
    assert saved_calls[0]["query"] == "Netflix"
    assert saved_calls[0]["detected_mode"] == DashboardMode.BUSINESS
    assert saved_calls[0]["data_source"] == DataSource.GOOGLE_NEWS_RSS


def test_dashboard_route_falls_back_when_google_news_fails(monkeypatch) -> None:
    def failing_fetch_google_news_rss(query: str, max_results: int):
        raise TimeoutError("rss unavailable")

    monkeypatch.setattr(
        "app.routes.news.fetch_google_news_rss", failing_fetch_google_news_rss
    )

    response = create_dashboard(
        DashboardRequest(query="PlayStation", max_results=2, use_ml=False)
    )

    assert response.data_source == DataSource.FALLBACK_MOCK
    assert response.analysis_source == AnalysisSource.RULE_BASED
    assert response.detected_mode == DashboardMode.GAMING
    assert len(response.sources) == 2
    assert all(source.source.startswith("Mock") for source in response.sources)


def test_dashboard_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        DashboardRequest(query="   ")


def test_dashboard_route_uses_hybrid_ml_when_available(monkeypatch) -> None:
    seen_kwargs = {}

    def fake_analyze_with_hybrid_ml(**kwargs):
        seen_kwargs.update(kwargs)
        return HybridAnalysisResult(
            analysis=DashboardAnalysis(
                sentiment_label="positive",
                sentiment_score=0.74,
                overall_signal="positive",
                event_tags=["gaming", "launch"],
                confidence="medium",
                possible_impact="Possible positive signal for review, release, or player interest.",
            ),
            detected_mode=DashboardMode.GAMING,
            analysis_source=AnalysisSource.HYBRID_ML,
        )

    monkeypatch.setattr(
        "app.routes.news.analyze_with_hybrid_ml", fake_analyze_with_hybrid_ml
    )

    response = create_dashboard(
        DashboardRequest(query="Nintendo Switch 2", max_results=2, use_real_news=False)
    )

    assert response.analysis_source == AnalysisSource.HYBRID_ML
    assert response.detected_mode == DashboardMode.GAMING
    assert response.sentiment.label == "positive"
    assert response.event_tags == ["gaming", "launch"]
    assert seen_kwargs["use_torch"] is True


def test_use_torch_false_is_passed_to_hybrid_analyzer(monkeypatch) -> None:
    seen_kwargs = {}

    def fake_analyze_with_hybrid_ml(**kwargs):
        seen_kwargs.update(kwargs)
        return HybridAnalysisResult(
            analysis=DashboardAnalysis(
                sentiment_label="neutral",
                sentiment_score=0.0,
                overall_signal="unclear",
                event_tags=["business"],
                confidence="medium",
                possible_impact="Impact is unclear.",
            ),
            detected_mode=DashboardMode.BUSINESS,
            analysis_source=AnalysisSource.HYBRID_ML,
            torch_used=False,
            torch_available=False,
        )

    monkeypatch.setattr(
        "app.routes.news.analyze_with_hybrid_ml", fake_analyze_with_hybrid_ml
    )

    response = create_dashboard(
        DashboardRequest(
            query="Microsoft stock",
            max_results=2,
            use_real_news=False,
            use_torch=False,
        )
    )

    body = response.model_dump()
    assert seen_kwargs["use_torch"] is False
    assert body["torch_used"] is False
    assert body["torch_available"] is False


def test_dashboard_route_includes_debug_when_requested(monkeypatch) -> None:
    def fake_analyze_with_hybrid_ml(**kwargs):
        return HybridAnalysisResult(
            analysis=DashboardAnalysis(
                sentiment_label="neutral",
                sentiment_score=0.0,
                overall_signal="unclear",
                event_tags=["sports"],
                confidence="medium",
                possible_impact="Impact is unclear.",
            ),
            detected_mode=DashboardMode.SPORTS,
            analysis_source=AnalysisSource.HYBRID_ML,
            debug={"aggregation_notes": ["debug enabled"]},
        )

    monkeypatch.setattr(
        "app.routes.news.analyze_with_hybrid_ml", fake_analyze_with_hybrid_ml
    )

    response = create_dashboard(
        DashboardRequest(
            query="NHL",
            max_results=2,
            use_real_news=False,
            debug_analysis=True,
        )
    )

    assert response.analysis_debug == {"aggregation_notes": ["debug enabled"]}


def test_dashboard_request_uses_torch_by_default() -> None:
    request = DashboardRequest(query="Nvidia stock")

    assert request.use_ml is True
    assert request.use_torch is True


def test_dashboard_route_falls_back_when_models_are_missing(monkeypatch) -> None:
    class EmptyPredictor:
        models = {}

    class MissingTorchService:
        available = False

    monkeypatch.setattr(
        "app.services.hybrid_analyzer.BaselinePredictor",
        lambda: EmptyPredictor(),
    )
    monkeypatch.setattr(
        "app.services.hybrid_analyzer.get_torch_topic_service",
        lambda: MissingTorchService(),
    )

    response = create_dashboard(
        DashboardRequest(query="Netflix", max_results=2, use_real_news=False)
    )

    assert response.analysis_source == AnalysisSource.RULE_BASED
    assert response.sentiment.label in {"positive", "neutral", "negative", "mixed"}


def test_use_ml_false_uses_rule_based_path(monkeypatch) -> None:
    def failing_hybrid_analyzer(articles, fallback_mode):
        raise AssertionError("hybrid analyzer should not be called")

    monkeypatch.setattr(
        "app.routes.news.analyze_with_hybrid_ml", failing_hybrid_analyzer
    )

    response = create_dashboard(
        DashboardRequest(query="Tesla", max_results=2, use_real_news=False, use_ml=False)
    )

    assert response.analysis_source == AnalysisSource.RULE_BASED


def test_dashboard_post_route_is_registered() -> None:
    matching_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/api/news/dashboard"
    ]

    assert matching_routes
    assert "POST" in matching_routes[0].methods
