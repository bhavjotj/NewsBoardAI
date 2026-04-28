from datetime import datetime, timezone

from app.models.dashboard import DashboardMode
from app.services.analyzer import analyze_articles
from app.services.fetchers import NewsArticle, fetch_mock_news


def article(title: str, snippet: str, source: str = "Test Source") -> NewsArticle:
    return NewsArticle(
        title=title,
        source=source,
        published_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
        snippet=snippet,
        url="https://example.com/test",
    )


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


def test_nintendo_switch_gaming_result_does_not_get_sports_tag() -> None:
    articles = [
        article(
            "Nintendo Switch 2 game lineup adds launch titles",
            "The console release includes new trailers, pricing details, and previews.",
        )
    ]

    analysis = analyze_articles(articles, DashboardMode.GAMING)

    assert "gaming" in analysis.event_tags
    assert "sports" not in analysis.event_tags


def test_mixed_positive_and_negative_items_return_cautious_signal() -> None:
    articles = [
        article(
            "Tesla shares rally after strong delivery update",
            "Analysts describe growth and better than expected demand.",
        ),
        article(
            "Tesla faces delay risk in new product rollout",
            "Coverage points to weak timing, investigation concerns, and possible cuts.",
        ),
    ]

    analysis = analyze_articles(articles, DashboardMode.BUSINESS)

    assert analysis.sentiment_label == "mixed"
    assert analysis.overall_signal == "mixed"
    assert "Mixed signal" in analysis.possible_impact


def test_duplicate_like_sources_do_not_automatically_raise_confidence() -> None:
    articles = [
        article(
            "Company launches record product update",
            "Strong growth and launch momentum continue across the market.",
            source=f"Source {index}",
        )
        for index in range(5)
    ]

    analysis = analyze_articles(articles, DashboardMode.BUSINESS)

    assert analysis.confidence != "high"


def test_business_mode_uses_market_aware_possible_impact() -> None:
    articles = [
        article(
            "Company beats expectations as shares rally",
            "Revenue growth and profit were better than expected.",
        ),
        article(
            "Analysts upgrade company stock after earnings",
            "Investors focused on strong quarterly results.",
        ),
    ]

    analysis = analyze_articles(articles, DashboardMode.BUSINESS)

    assert "market and business attention" in analysis.possible_impact
