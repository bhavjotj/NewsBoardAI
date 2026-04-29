from fastapi import APIRouter

from app.models.dashboard import (
    AnalysisSource,
    DashboardMode,
    DashboardRequest,
    DashboardResponse,
    DataSource,
)
from app.services.analyzer import analyze_articles, detect_mode
from app.services.data_collection import save_news_examples
from app.services.fetchers import NewsArticle, fetch_google_news_rss, fetch_mock_news
from app.services.formatter import format_dashboard_response
from app.services.hybrid_analyzer import analyze_with_hybrid_ml

router = APIRouter()


@router.post("/dashboard", response_model=DashboardResponse)
def create_dashboard(request: DashboardRequest) -> DashboardResponse:
    articles, data_source = _fetch_articles(request)
    fallback_mode = request.mode or detect_mode(request.query, articles)
    analysis, detected_mode, analysis_source = _analyze_articles(
        request,
        articles,
        fallback_mode,
    )
    if request.save_examples and data_source == DataSource.GOOGLE_NEWS_RSS:
        save_news_examples(
            query=request.query,
            detected_mode=detected_mode,
            articles=articles,
            data_source=data_source,
        )
    return format_dashboard_response(
        request,
        articles,
        analysis,
        data_source,
        detected_mode,
        analysis_source,
    )


def _fetch_articles(request: DashboardRequest) -> tuple[list[NewsArticle], DataSource]:
    if not request.use_real_news:
        mock_mode = request.mode or detect_mode(request.query, [])
        return (
            fetch_mock_news(request.query, request.max_results, mock_mode),
            DataSource.MOCK,
        )

    try:
        articles = fetch_google_news_rss(request.query, request.max_results)
        if articles:
            return articles, DataSource.GOOGLE_NEWS_RSS
    except Exception:
        pass

    return (
        fetch_mock_news(
            request.query,
            request.max_results,
            request.mode or detect_mode(request.query, []),
        ),
        DataSource.FALLBACK_MOCK,
    )


def _analyze_articles(
    request: DashboardRequest,
    articles: list[NewsArticle],
    fallback_mode: DashboardMode,
):
    if request.use_ml:
        return analyze_with_hybrid_ml(articles, fallback_mode)
    return (
        analyze_articles(articles, fallback_mode),
        fallback_mode,
        AnalysisSource.RULE_BASED,
    )
