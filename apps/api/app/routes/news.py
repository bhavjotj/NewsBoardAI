from fastapi import APIRouter

from app.models.dashboard import DashboardRequest, DashboardResponse, DataSource
from app.services.analyzer import analyze_articles
from app.services.fetchers import fetch_google_news_rss, fetch_mock_news
from app.services.formatter import format_dashboard_response

router = APIRouter()


@router.post("/dashboard", response_model=DashboardResponse)
def create_dashboard(request: DashboardRequest) -> DashboardResponse:
    articles, data_source = _fetch_articles(request)
    analysis = analyze_articles(articles, request.mode)
    return format_dashboard_response(request, articles, analysis, data_source)


def _fetch_articles(request: DashboardRequest):
    if not request.use_real_news:
        return (
            fetch_mock_news(request.query, request.max_results, request.mode),
            DataSource.MOCK,
        )

    try:
        articles = fetch_google_news_rss(request.query, request.max_results)
        if articles:
            return articles, DataSource.GOOGLE_NEWS_RSS
    except Exception:
        pass

    return (
        fetch_mock_news(request.query, request.max_results, request.mode),
        DataSource.FALLBACK_MOCK,
    )
