from app.models.dashboard import DashboardMode
from app.services.analyzer import analyze_articles
from app.services.fetchers import fetch_mock_news


def test_analyzer_returns_dashboard_signals() -> None:
    articles = fetch_mock_news("Nintendo", max_results=3, mode=DashboardMode.GAMING)

    analysis = analyze_articles(articles, DashboardMode.GAMING)

    assert analysis.sentiment_label in {"positive", "neutral", "negative", "mixed"}
    assert analysis.confidence == "medium"
    assert "gaming" in analysis.event_tags
    assert analysis.possible_impact


def test_analyzer_handles_no_articles() -> None:
    analysis = analyze_articles([], DashboardMode.GENERAL)

    assert analysis.sentiment_label == "neutral"
    assert analysis.overall_signal == "unclear"
    assert analysis.confidence == "low"
    assert analysis.event_tags == ["general"]
