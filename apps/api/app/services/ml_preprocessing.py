# Purpose: Preprocesses the text for the machine learning models.
from __future__ import annotations

from app.utils.text import clean_text

MIN_SNIPPET_WORDS = 5
MAX_TITLE_OVERLAP = 0.75


def build_model_text(example: dict) -> str:
    # Avoid training on duplicate title/snippet text from RSS-style feeds.
    title = clean_text(str(example.get("title", "")))
    snippet = clean_text(str(example.get("snippet", "")))

    if not title:
        return snippet
    if not _snippet_adds_signal(title, snippet):
        return title
    return f"{title}. {snippet}"


def _snippet_adds_signal(title: str, snippet: str) -> bool:
    title_tokens = _tokens(title)
    snippet_tokens = _tokens(snippet)

    if len(snippet_tokens) < MIN_SNIPPET_WORDS:
        return False
    if not title_tokens:
        return True

    new_tokens = snippet_tokens - title_tokens
    if len(new_tokens) < 3:
        return False

    overlap = len(snippet_tokens & title_tokens) / len(snippet_tokens)
    return overlap <= MAX_TITLE_OVERLAP


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in clean_text(value).lower().replace("-", " ").split()
        if len(token) > 2
    }
