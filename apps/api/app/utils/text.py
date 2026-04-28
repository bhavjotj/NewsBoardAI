from html import unescape
from re import IGNORECASE, escape, sub


def clean_text(value: str) -> str:
    text = unescape(value)
    text = sub(r"<[^>]+>", " ", text)
    text = sub(r"\s+", " ", text)
    return text.strip()


def clean_snippet(value: str, source: str = "", max_length: int = 180) -> str:
    text = clean_text(value)
    if source:
        source_pattern = escape(source.strip())
        text = sub(rf"^\s*{source_pattern}\s*[-:|]\s*", "", text, flags=IGNORECASE)
        text = sub(rf"\s*[-:|]\s*{source_pattern}\s*$", "", text, flags=IGNORECASE)
        text = sub(rf"\b{source_pattern}\b\s*$", "", text, flags=IGNORECASE)
    text = sub(r"\s+", " ", text).strip(" -|:")
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "."
