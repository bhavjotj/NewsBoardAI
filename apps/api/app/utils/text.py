from html import unescape
from re import sub


def clean_text(value: str) -> str:
    text = unescape(value)
    text = sub(r"<[^>]+>", " ", text)
    text = sub(r"\s+", " ", text)
    return text.strip()
