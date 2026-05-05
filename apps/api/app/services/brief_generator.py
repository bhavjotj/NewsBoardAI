from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from app.models.dashboard import BriefSource, DashboardMode
from app.services.fetchers import NewsArticle
from app.utils.text import clean_snippet, clean_text

OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TIMEOUT_SECONDS = 10
OLLAMA_NUM_PREDICT = 140
MAX_BRIEF_CHARS = 360
MAX_IMPACT_CHARS = 260
MAX_SOURCE_SNIPPET_CHARS = 220
LABEL_PATTERN = re.compile(
    r"\b(brief|possible impact|main theme|newsboardai dashboard brief)\s*:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BriefGenerationResult:
    brief: str
    possible_impact: str
    brief_source: BriefSource
    possible_impact_source: BriefSource
    llm_available: bool | None


def generate_dashboard_brief(
    query: str,
    detected_mode: DashboardMode,
    overall_signal: str,
    sentiment_label: str,
    sentiment_score: float,
    event_tags: list[str],
    confidence_label: str,
    possible_impact: str,
    articles: list[NewsArticle],
    template_brief: str,
    template_possible_impact: str,
    use_llm: bool,
    model: str,
    ollama_generate: Callable[[str, str], str] | None = None,
) -> BriefGenerationResult:
    if not use_llm:
        return BriefGenerationResult(
            brief=template_brief,
            possible_impact=template_possible_impact,
            brief_source=BriefSource.TEMPLATE,
            possible_impact_source=BriefSource.TEMPLATE,
            llm_available=None,
        )

    generate = ollama_generate or call_ollama_generate
    prompt = build_brief_prompt(
        query=query,
        detected_mode=detected_mode,
        overall_signal=overall_signal,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        event_tags=event_tags,
        possible_impact=possible_impact,
        confidence_label=confidence_label,
        template_brief=template_brief,
        template_possible_impact=template_possible_impact,
        articles=articles,
    )

    try:
        candidate = parse_ollama_json(generate(model=model, prompt=prompt))
    except Exception:
        return BriefGenerationResult(
            brief=template_brief,
            possible_impact=template_possible_impact,
            brief_source=BriefSource.OLLAMA_FALLBACK,
            possible_impact_source=BriefSource.OLLAMA_FALLBACK,
            llm_available=False,
        )

    brief = validate_brief(str(candidate.get("brief", "")) if candidate else "")
    impact = validate_possible_impact(
        str(candidate.get("possible_impact", "")) if candidate else ""
    )
    brief_valid = brief is not None
    impact_valid = impact is not None
    return BriefGenerationResult(
        brief=brief or template_brief,
        possible_impact=impact or template_possible_impact,
        brief_source=_field_source(brief_valid, impact_valid),
        possible_impact_source=_field_source(impact_valid, brief_valid),
        llm_available=True,
    )


def build_brief_prompt(
    query: str,
    detected_mode: DashboardMode,
    overall_signal: str,
    sentiment_label: str,
    sentiment_score: float,
    event_tags: list[str],
    possible_impact: str,
    confidence_label: str,
    template_brief: str,
    template_possible_impact: str,
    articles: list[NewsArticle],
) -> str:
    sources = "\n".join(
        _source_lines(article, index)
        for index, article in enumerate(articles, start=1)
    )
    tags = ", ".join(event_tags[:4]) or "general"
    return (
        "Rewrite NewsBoardAI presentation text as strict JSON only.\n"
        'Return exactly: {"brief":"...","possible_impact":"..."}\n'
        "Use only the provided source titles and snippets.\n"
        "Do not add facts, names, numbers, or claims not shown below.\n"
        "Do not give financial advice or certain predictions.\n"
        "Do not mention NewsBoardAI.\n"
        "Do not include markdown, headings, labels, bullets, or preambles.\n"
        "The brief must be 1 to 2 sentences.\n"
        "The possible_impact must be 1 sentence.\n"
        "Use cautious wording.\n\n"
        f"Topic: {query}\n"
        f"Detected mode: {detected_mode.value}\n"
        f"Overall signal: {overall_signal}\n"
        f"Sentiment: {sentiment_label} ({sentiment_score:.2f})\n"
        f"Event tags: {tags}\n"
        f"Confidence: {confidence_label}\n"
        f"Template brief: {template_brief}\n"
        f"Template possible impact: {template_possible_impact or possible_impact}\n\n"
        f"Sources:\n{sources}\n\n"
        "JSON:"
    )


def call_ollama_generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": OLLAMA_NUM_PREDICT,
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


def parse_ollama_json(value: str) -> dict[str, Any] | None:
    text = clean_text(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_text = _extract_json_object(text)
        if json_text is None:
            return None
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def validate_brief(value: str) -> str | None:
    return _validate_generated_text(
        value=value,
        max_chars=MAX_BRIEF_CHARS,
        max_sentences=2,
    )


def validate_possible_impact(value: str) -> str | None:
    return _validate_generated_text(
        value=value,
        max_chars=MAX_IMPACT_CHARS,
        max_sentences=1,
    )


def _validate_generated_text(
    value: str,
    max_chars: int,
    max_sentences: int,
) -> str | None:
    if LABEL_PATTERN.search(value):
        return None
    text = _clean_generated_text(value)
    if not text:
        return None
    if len(text) > max_chars:
        return None
    if _sentence_count(text) > max_sentences:
        return None
    if text.lower().startswith(("here is", "here's", "newsboardai")):
        return None
    return text


def _clean_generated_text(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text)
    text = re.sub(
        r"^(brief|possible impact|main theme|impact)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split()).strip("\"' ")


def _extract_json_object(value: str) -> str | None:
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return value[start : end + 1]


def _field_source(field_valid: bool, other_field_valid: bool) -> BriefSource:
    if field_valid and other_field_valid:
        return BriefSource.OLLAMA
    if field_valid:
        return BriefSource.OLLAMA_PARTIAL
    return BriefSource.OLLAMA_FALLBACK


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
