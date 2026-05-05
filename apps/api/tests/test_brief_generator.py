from datetime import datetime, timezone

from app.models.dashboard import BriefSource, DashboardMode
from app.services.brief_generator import (
    OLLAMA_TIMEOUT_SECONDS,
    build_brief_prompt,
    call_ollama_generate,
    generate_dashboard_brief,
)
from app.services.fetchers import NewsArticle


def article(title: str, snippet: str = "") -> NewsArticle:
    return NewsArticle(
        title=title,
        source="Example",
        published_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
        snippet=snippet,
        url="https://example.com/not-sent-to-ollama",
    )


def base_kwargs():
    return {
        "query": "Tesla",
        "detected_mode": DashboardMode.BUSINESS,
        "overall_signal": "mixed",
        "sentiment_label": "mixed",
        "sentiment_score": 0.0,
        "event_tags": ["business", "risk"],
        "possible_impact": "Mixed signal for market attention.",
        "articles": [
            article(
                "Tesla shares rise after earnings",
                "Investors watch revenue while risks remain.",
            )
        ],
        "template_brief": "Template brief.",
        "model": "llama3.2",
    }


def test_successful_ollama_brief_generation() -> None:
    captured = {}

    def fake_generate(model: str, prompt: str) -> str:
        captured["model"] = model
        captured["prompt"] = prompt
        return "Tesla coverage is focused on earnings and risk, creating a mixed business signal."

    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=fake_generate,
    )

    assert result.brief_source == BriefSource.OLLAMA
    assert result.llm_available is True
    assert "mixed business signal" in result.brief
    assert captured["model"] == "llama3.2"
    assert "not-sent-to-ollama" not in captured["prompt"]


def test_ollama_failure_falls_back_to_template() -> None:
    def failing_generate(model: str, prompt: str) -> str:
        raise TimeoutError("ollama timed out")

    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=failing_generate,
    )

    assert result.brief == "Template brief."
    assert result.brief_source == BriefSource.OLLAMA_FALLBACK
    assert result.llm_available is False


def test_ollama_http_call_uses_ten_second_timeout(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout: int):
        captured["timeout"] = timeout
        raise TimeoutError("ollama timed out")

    monkeypatch.setattr("app.services.brief_generator.urlopen", fake_urlopen)

    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=call_ollama_generate,
    )

    assert captured["timeout"] == OLLAMA_TIMEOUT_SECONDS == 10
    assert result.brief == "Template brief."
    assert result.brief_source == BriefSource.OLLAMA_FALLBACK
    assert result.llm_available is False


def test_empty_or_too_long_ollama_response_falls_back() -> None:
    for response in ("", "Too long. " * 80):
        result = generate_dashboard_brief(
            **base_kwargs(),
            use_llm=True,
            ollama_generate=lambda model, prompt, value=response: value,
        )

        assert result.brief == "Template brief."
        assert result.brief_source == BriefSource.OLLAMA_FALLBACK
        assert result.llm_available is True


def test_use_llm_false_uses_template_without_calling_ollama() -> None:
    def failing_generate(model: str, prompt: str) -> str:
        raise AssertionError("Ollama should not be called")

    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=False,
        ollama_generate=failing_generate,
    )

    assert result.brief == "Template brief."
    assert result.brief_source == BriefSource.TEMPLATE
    assert result.llm_available is None


def test_prompt_is_grounded_and_excludes_urls() -> None:
    prompt = build_brief_prompt(
        query="Nintendo Switch 2",
        detected_mode=DashboardMode.GAMING,
        overall_signal="positive",
        sentiment_label="positive",
        sentiment_score=0.4,
        event_tags=["gaming", "launch"],
        possible_impact="Possible player interest.",
        articles=[
            article(
                "Nintendo Switch 2 launch details emerge",
                "Console previews mention games and release timing.",
            )
        ],
    )

    assert "Use only the provided source titles and snippets" in prompt
    assert "Do not add facts" in prompt
    assert "https://example.com" not in prompt
