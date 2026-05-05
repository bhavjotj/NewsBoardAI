from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from app.models.dashboard import BriefSource, DashboardMode
from app.services.fetchers import NewsArticle
from app.utils.text import clean_snippet, clean_text

OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TIMEOUT_SECONDS = 10
MAX_BRIEF_CHARS = 360
MAX_SOURCE_SNIPPET_CHARS = 220


@dataclass(frozen=True)
class BriefGenerationResult:
    brief: str
    brief_source: BriefSource
    llm_available: bool | None


def generate_dashboard_brief(
    query: str,
    detected_mode: DashboardMode,
    overall_signal: str,
    sentiment_label: str,
    sentiment_score: float,
    event_tags: list[str],
    possible_impact: str,
    articles: list[NewsArticle],
    template_brief: str,
    use_llm: bool,
    model: str,
    ollama_generate: Callable[[str, str], str] | None = None,
) -> BriefGenerationResult:
    if not use_llm:
        return BriefGenerationResult(template_brief, BriefSource.TEMPLATE, None)

    generate = ollama_generate or call_ollama_generate
    prompt = build_brief_prompt(
        query=query,
        detected_mode=detected_mode,
        overall_signal=overall_signal,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        event_tags=event_tags,
        possible_impact=possible_impact,
        articles=articles,
    )

    try:
        candidate = generate(model=model, prompt=prompt)
    except Exception:
        return BriefGenerationResult(
            template_brief,
            BriefSource.OLLAMA_FALLBACK,
            False,
        )

    brief = validate_brief(candidate)
    if brief is None:
        return BriefGenerationResult(
            template_brief,
            BriefSource.OLLAMA_FALLBACK,
            True,
        )
    return BriefGenerationResult(brief, BriefSource.OLLAMA, True)


def build_brief_prompt(
    query: str,
    detected_mode: DashboardMode,
    overall_signal: str,
    sentiment_label: str,
    sentiment_score: float,
    event_tags: list[str],
    possible_impact: str,
    articles: list[NewsArticle],
) -> str:
    sources = "\n".join(
        _source_lines(article, index)
        for index, article in enumerate(articles, start=1)
    )
    tags = ", ".join(event_tags[:4]) or "general"
    return (
        "Write a concise NewsBoardAI dashboard brief.\n"
        "Use only the provided source titles and snippets.\n"
        "Do not add facts, names, numbers, or claims not shown below.\n"
        "Do not give financial advice or certain predictions.\n"
        "Keep it 1 to 2 sentences.\n"
        "Mention the main theme and signal using cautious wording.\n\n"
        f"Topic: {query}\n"
        f"Detected mode: {detected_mode.value}\n"
        f"Overall signal: {overall_signal}\n"
        f"Sentiment: {sentiment_label} ({sentiment_score:.2f})\n"
        f"Event tags: {tags}\n"
        f"Possible impact: {possible_impact}\n\n"
        f"Sources:\n{sources}\n\n"
        "Brief:"
    )


def call_ollama_generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 90,
        },
    }
    request = Request(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama brief generation failed.") from error
    return str(body.get("response", ""))


def validate_brief(value: str) -> str | None:
    brief = " ".join(clean_text(value).split())
    brief = brief.strip("\"' ")
    if not brief:
        return None
    if len(brief) > MAX_BRIEF_CHARS:
        return None
    if _sentence_count(brief) > 2:
        return None
    if "\n" in brief or brief.startswith(("-", "*")):
        return None
    return brief


def _source_lines(article: NewsArticle, index: int) -> str:
    snippet = clean_snippet(
        article.snippet,
        article.source,
        max_length=MAX_SOURCE_SNIPPET_CHARS,
    )
    return (
        f"{index}. Title: {clean_text(article.title)}\n"
        f"   Snippet: {snippet or 'No snippet provided.'}"
    )


def _sentence_count(value: str) -> int:
    endings = sum(1 for character in value if character in ".!?")
    return max(1, endings)
