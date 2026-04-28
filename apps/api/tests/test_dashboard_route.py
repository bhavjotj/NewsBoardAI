from app.main import app
from app.models.dashboard import DashboardRequest
from app.routes.news import create_dashboard


def test_dashboard_route_returns_compact_response() -> None:
    response = create_dashboard(
        DashboardRequest(query="Tesla", max_results=3, mode="business")
    )

    body = response.model_dump()
    assert body["topic"] == "Tesla"
    assert body["time_window"] == "Recent news"
    assert body["brief"]
    assert body["sentiment"]["label"] in {"positive", "neutral", "negative", "mixed"}
    assert "business" in body["event_tags"]
    assert len(body["sources"]) == 3
    assert body["confidence"] in {"low", "medium", "high"}
    assert body["possible_impact"]


def test_dashboard_route_uses_request_defaults() -> None:
    response = create_dashboard(DashboardRequest(query="Apple"))

    body = response.model_dump()
    assert len(body["sources"]) == 5


def test_dashboard_post_route_is_registered() -> None:
    matching_routes = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/api/news/dashboard"
    ]

    assert matching_routes
    assert "POST" in matching_routes[0].methods
