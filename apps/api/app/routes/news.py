# Purpose: Defines the routes for the news dashboard, including the dashboard endpoint and the helper functions. Orchestrates the flow of data between the frontend and the backend.
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
from app.services.hybrid_analyzer import HybridAnalysisResult, analyze_with_hybrid_ml

# The router for the news dashboard
router = APIRouter()

# The dashboard endpoint, where we create the dashboard
@router.post(
    "/dashboard",
    response_model=DashboardResponse,
    response_model_exclude_none=True,
)
def create_dashboard(request: DashboardRequest) -> DashboardResponse:
    # Keep the route as orchestration; services own fetching, analysis, and shape.
    # Fetch the articles from the data source
    articles, data_source = _fetch_articles(request)
    fallback_mode = request.mode or detect_mode(request.query, articles)
    analysis_result = _analyze_articles(
        request,
        articles,
        fallback_mode,
    ) # Analyze the articles
    if request.save_examples and data_source == DataSource.GOOGLE_NEWS_RSS:
        save_news_examples(
            query=request.query,
            detected_mode=analysis_result.detected_mode,
            articles=articles,
            data_source=data_source,
        ) # Save the examples
    return format_dashboard_response( # Format the response
        request,
        articles,
        analysis_result.analysis,
        data_source,
        analysis_result.detected_mode,
        analysis_result.analysis_source,
        analysis_result.debug if request.debug_analysis else None,
        analysis_result.torch_used,
        analysis_result.torch_available,
    )

# Helper function to fetch the articles from the data source, main API endpoint
def _fetch_articles(request: DashboardRequest) -> tuple[list[NewsArticle], DataSource]:
    if not request.use_real_news: # If we are not using real news, use the mock data
        mock_mode = request.mode or detect_mode(request.query, [])
        return (
            fetch_mock_news(request.query, request.max_results, mock_mode),
            DataSource.MOCK,
        )
    # Try to fetch the articles from the RSS feed
    try:
        articles = fetch_google_news_rss(request.query, request.max_results)
        if articles:
            return articles, DataSource.GOOGLE_NEWS_RSS
    except Exception:
        # RSS is optional for local use; mock data keeps the dashboard usable.
        pass

    return (
        fetch_mock_news(
            request.query,
            request.max_results,
            request.mode or detect_mode(request.query, []),
        ),
        DataSource.FALLBACK_MOCK,
    )

# Helper function to analyze the articles
def _analyze_articles(
    request: DashboardRequest,
    articles: list[NewsArticle],
    fallback_mode: DashboardMode,
): # Use the hybrid ML model if enabled
    if request.use_ml:
        return analyze_with_hybrid_ml(
            query=request.query,
            articles=articles,
            fallback_mode=fallback_mode,
            include_debug=request.debug_analysis,
            use_torch=request.use_torch,
        )
    return HybridAnalysisResult( # Use the rule-based model if the hybrid ML model is not enabled
        analysis=analyze_articles(articles, fallback_mode),
        detected_mode=fallback_mode,
        analysis_source=AnalysisSource.RULE_BASED,
        debug=None,
    )
