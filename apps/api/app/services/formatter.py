from app.models.dashboard import (
    DataSource,
    DashboardRequest,
    DashboardResponse,
    SentimentSummary,
    SourceCard,
)
from app.services.analyzer import DashboardAnalysis
from app.services.fetchers import NewsArticle


def format_dashboard_response(
    request: DashboardRequest,
    articles: list[NewsArticle],
    analysis: DashboardAnalysis,
    data_source: DataSource,
) -> DashboardResponse:
    return DashboardResponse(
        topic=request.query,
        data_source=data_source,
        time_window="Recent news",
        overall_signal=analysis.overall_signal,
        brief=_brief(request.query, articles, analysis),
        sentiment=SentimentSummary(
            label=analysis.sentiment_label,
            score=analysis.sentiment_score,
        ),
        event_tags=analysis.event_tags,
        sources=[
            SourceCard(
                title=article.title,
                source=article.source,
                published_at=article.published_at,
                snippet=article.snippet,
                url=article.url,
            )
            for article in articles
        ],
        confidence=analysis.confidence,
        possible_impact=analysis.possible_impact,
    )


def _brief(
    topic: str, articles: list[NewsArticle], analysis: DashboardAnalysis
) -> str:
    if not articles:
        return f"No recent source cards are available for {topic}."

    top_source = articles[0].source
    return (
        f"{topic} has a {analysis.overall_signal} signal across "
        f"{len(articles)} recent source cards, led by {top_source}."
    )
