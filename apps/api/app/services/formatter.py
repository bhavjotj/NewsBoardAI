from app.models.dashboard import (
    DataSource,
    DashboardMode,
    DashboardRequest,
    DashboardResponse,
    SentimentSummary,
    SourceCard,
)
from app.services.analyzer import DashboardAnalysis
from app.services.fetchers import NewsArticle
from app.utils.text import clean_snippet


def format_dashboard_response(
    request: DashboardRequest,
    articles: list[NewsArticle],
    analysis: DashboardAnalysis,
    data_source: DataSource,
    detected_mode: DashboardMode,
) -> DashboardResponse:
    return DashboardResponse(
        topic=request.query,
        data_source=data_source,
        detected_mode=detected_mode,
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
                snippet=clean_snippet(article.snippet, article.source),
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
        return f"No recent coverage is available for {topic}."

    tags = _tag_summary(analysis.event_tags)
    top_source = articles[0].source
    top_title = articles[0].title
    return (
        f"{topic} is showing a {analysis.overall_signal} signal around {tags}. "
        f"The clearest recent item is from {top_source}: {top_title}."
    )


def _tag_summary(tags: list[str]) -> str:
    useful_tags = [tag for tag in tags if tag != "general"][:3]
    if not useful_tags:
        return "general coverage"
    if len(useful_tags) == 1:
        return useful_tags[0]
    return ", ".join(useful_tags[:-1]) + f", and {useful_tags[-1]}"
