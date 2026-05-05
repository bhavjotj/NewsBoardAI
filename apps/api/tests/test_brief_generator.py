from datetime import datetime, timezone
import json

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
        "confidence_label": "medium",
        "possible_impact": "Mixed signal for market attention.",
        "articles": [
            article(
                "Tesla shares rise after earnings",
                "Investors watch revenue while risks remain.",
            )
        ],
        "template_brief": "Template brief.",
        "template_possible_impact": "Template impact.",
        "model": "llama3.2",
    }


def test_successful_ollama_text_generation_updates_both_fields() -> None:
    captured = {}

    def fake_generate(model: str, prompt: str) -> str:
        captured["model"] = model
        captured["prompt"] = prompt
        return json.dumps(
            {
                "brief": "Tesla coverage is focused on earnings and risk, creating a mixed business signal.",
                "possible_impact": "Market attention could stay cautious as investors weigh growth against risk.",
            }
        )

    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=fake_generate,
    )

    assert result.brief_source == BriefSource.OLLAMA
    assert result.possible_impact_source == BriefSource.OLLAMA
    assert result.llm_available is True
    assert "mixed business signal" in result.brief
    assert "Market attention" in result.possible_impact
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
    assert result.possible_impact == "Template impact."
    assert result.brief_source == BriefSource.OLLAMA_FALLBACK
    assert result.possible_impact_source == BriefSource.OLLAMA_FALLBACK
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
    assert result.possible_impact == "Template impact."
    assert result.brief_source == BriefSource.OLLAMA_FALLBACK
    assert result.possible_impact_source == BriefSource.OLLAMA_FALLBACK
    assert result.llm_available is False


def test_non_json_or_too_long_ollama_response_falls_back() -> None:
    long_response = json.dumps(
        {
            "brief": "Too long. " * 80,
            "possible_impact": "Also too long. " * 80,
        }
    )
    for response in ("", "This is not JSON.", long_response):
        result = generate_dashboard_brief(
            **base_kwargs(),
            use_llm=True,
            ollama_generate=lambda model, prompt, value=response: value,
        )

        assert result.brief == "Template brief."
        assert result.possible_impact == "Template impact."
        assert result.brief_source == BriefSource.OLLAMA_FALLBACK
        assert result.possible_impact_source == BriefSource.OLLAMA_FALLBACK
        assert result.llm_available is True


def test_invalid_brief_valid_impact_uses_partial_fallback() -> None:
    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=lambda model, prompt: json.dumps(
            {
                "brief": "Brief: Here is a markdown-heavy **dashboard brief** with a label.",
                "possible_impact": "Market attention could remain cautious as the coverage develops.",
            }
        ),
    )

    assert result.brief == "Template brief."
    assert result.possible_impact.startswith("Market attention")
    assert result.brief_source == BriefSource.OLLAMA_FALLBACK
    assert result.possible_impact_source == BriefSource.OLLAMA_PARTIAL


def test_valid_brief_invalid_impact_uses_partial_fallback() -> None:
    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=lambda model, prompt: json.dumps(
            {
                "brief": "Tesla coverage centers on earnings and risk, leaving the business signal mixed.",
                "possible_impact": "Possible Impact: Here is a certain prediction. It will happen.",
            }
        ),
    )

    assert result.brief.startswith("Tesla coverage")
    assert result.possible_impact == "Template impact."
    assert result.brief_source == BriefSource.OLLAMA_PARTIAL
    assert result.possible_impact_source == BriefSource.OLLAMA_FALLBACK


def test_json_wrapped_in_extra_text_is_parsed() -> None:
    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=lambda model, prompt: (
            'Sure. {"brief":"Tesla coverage centers on earnings and risk, leaving the business signal mixed.",'
            '"possible_impact":"Investor attention could remain cautious while the coverage develops."}'
        ),
    )

    assert result.brief_source == BriefSource.OLLAMA
    assert result.possible_impact_source == BriefSource.OLLAMA
    assert result.brief.startswith("Tesla coverage")
    assert result.possible_impact.startswith("Investor attention")


def test_markdown_heading_or_embedded_labels_are_rejected() -> None:
    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=True,
        ollama_generate=lambda model, prompt: json.dumps(
            {
                "brief": "### Main Theme: Tesla coverage is mixed.",
                "possible_impact": "The impact is cautious, but Brief: this includes a label.",
            }
        ),
    )

    assert result.brief == "Template brief."
    assert result.possible_impact == "Template impact."
    assert result.brief_source == BriefSource.OLLAMA_FALLBACK
    assert result.possible_impact_source == BriefSource.OLLAMA_FALLBACK


def test_use_llm_false_uses_template_without_calling_ollama() -> None:
    def failing_generate(model: str, prompt: str) -> str:
        raise AssertionError("Ollama should not be called")

    result = generate_dashboard_brief(
        **base_kwargs(),
        use_llm=False,
        ollama_generate=failing_generate,
    )

    assert result.brief == "Template brief."
    assert result.possible_impact == "Template impact."
    assert result.brief_source == BriefSource.TEMPLATE
    assert result.possible_impact_source == BriefSource.TEMPLATE
    assert result.llm_available is None


def test_prompt_is_grounded_and_excludes_urls() -> None:
    prompt = build_brief_prompt(
        query="Nintendo Switch 2",
        detected_mode=DashboardMode.GAMING,
        overall_signal="positive",
        sentiment_label="positive",
        sentiment_score=0.4,
        event_tags=["gaming", "launch"],
        confidence_label="medium",
        possible_impact="Possible player interest.",
        template_brief="Template brief.",
        template_possible_impact="Template impact.",
        articles=[
            article(
                "Nintendo Switch 2 launch details emerge",
                "Console previews mention games and release timing.",
            )
        ],
    )

    assert "Use only the provided source titles and snippets" in prompt
    assert "Do not add facts" in prompt
    assert '"brief":"...","possible_impact":"..."' in prompt
    assert "https://example.com" not in prompt
