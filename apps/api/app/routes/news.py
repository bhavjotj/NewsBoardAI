from fastapi import APIRouter

from app.models.dashboard import DashboardRequest, DashboardResponse
from app.services.analyzer import analyze_articles
from app.services.fetchers import fetch_mock_news
from app.services.formatter import format_dashboard_response

router = APIRouter()


@router.post("/dashboard", response_model=DashboardResponse)
def create_dashboard(request: DashboardRequest) -> DashboardResponse:
    articles = fetch_mock_news(request.query, request.max_results, request.mode)
    analysis = analyze_articles(articles, request.mode)
    return format_dashboard_response(request, articles, analysis)
