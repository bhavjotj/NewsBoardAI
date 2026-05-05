from __future__ import annotations

from app.models.dashboard import (
    AnalysisSource,
    DataSource,
    DashboardMode,
    DashboardRequest,
    DashboardResponse,
    SentimentSummary,
    SourceCard,
)
from app.services.analyzer import DashboardAnalysis
from app.services.brief_generator import generate_dashboard_brief
from app.services.fetchers import NewsArticle
from app.utils.text import clean_snippet


def format_dashboard_response(
    request: DashboardRequest,
    articles: list[NewsArticle],
    analysis: DashboardAnalysis,
    data_source: DataSource,
    detected_mode: DashboardMode,
    analysis_source: AnalysisSource,
    analysis_debug: dict | None = None,
    torch_used: bool | None = None,
    torch_available: bool | None = None,
) -> DashboardResponse:
    template_brief = _brief(request.query, articles, analysis)
    brief_result = generate_dashboard_brief(
        query=request.query,
        detected_mode=detected_mode,
        overall_signal=analysis.overall_signal,
        sentiment_label=analysis.sentiment_label,
        sentiment_score=analysis.sentiment_score,
        event_tags=analysis.event_tags,
        confidence_label=analysis.confidence,
        possible_impact=analysis.possible_impact,
        articles=articles,
        template_brief=template_brief,
        template_possible_impact=analysis.possible_impact,
        use_llm=request.use_llm_brief,
        model=request.ollama_model,
    )
    return DashboardResponse(
        topic=request.query,
        data_source=data_source,
        analysis_source=analysis_source,
        detected_mode=detected_mode,
        time_window="Recent news",
        overall_signal=analysis.overall_signal,
        brief=brief_result.brief,
        brief_source=brief_result.brief_source,
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
        possible_impact=brief_result.possible_impact,
        possible_impact_source=brief_result.possible_impact_source,
        llm_available=brief_result.llm_available,
        torch_used=torch_used,
        torch_available=torch_available,
        analysis_debug=analysis_debug,
    )


def _brief(
    topic: str, articles: list[NewsArticle], analysis: DashboardAnalysis
) -> str:
    if not articles:
        return f"No recent coverage is available for {topic}."
    if analysis.brief:
        return analysis.brief

    tags = _tag_summary(analysis.event_tags)
    top_source = articles[0].source
    return (
        f"{topic} is showing a {analysis.overall_signal} signal around {tags}. "
        f"Recent coverage led by {top_source} points to a possible shift worth watching."
    )


def _tag_summary(tags: list[str]) -> str:
    useful_tags = [tag for tag in tags if tag != "general"][:3]
    if not useful_tags:
        return "general coverage"
    if len(useful_tags) == 1:
        return useful_tags[0]
    return ", ".join(useful_tags[:-1]) + f", and {useful_tags[-1]}"
