from __future__ import annotations

import csv
import json
from pathlib import Path

from app.services.ml_preprocessing import build_model_text
from app.utils.text import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

TEXT_COLUMNS = ("sentence", "text")
SENTIMENT_COLUMNS = ("sentiment", "label")
TITLE_COLUMNS = ("title", "headline")
DESCRIPTION_COLUMNS = ("description", "text", "snippet")
CATEGORY_COLUMNS = ("category", "label", "class")

AG_NEWS_MODE_MAP = {
    "1": "politics",
    "2": "sports",
    "3": "business",
    "4": "general",
    "world": "politics",
    "sports": "sports",
    "business": "business",
    "sci/tech": "general",
    "sci-tech": "general",
    "science/technology": "general",
    "tech": "general",
}


def load_financial_phrasebank(path: Path, max_rows: int | None = None) -> list[dict]:
    if not path.exists():
        print_missing_dataset(
            path,
            "Financial PhraseBank CSV",
            "sentence or text; sentiment or label",
        )
        return []

    examples = []
    for row in read_csv_rows(path, max_rows):
        text = first_value(row, TEXT_COLUMNS)
        label = normalize_label(first_value(row, SENTIMENT_COLUMNS))
        if text and label in {"positive", "neutral", "negative"}:
            examples.append({"text": clean_text(text), "label": label})
    return examples


def load_ag_news(path: Path, max_rows: int | None = None) -> list[dict]:
    if not path.exists():
        print_missing_dataset(
            path,
            "AG News CSV",
            "title; description or text; category or label",
        )
        return []

    examples = []
    for row in read_csv_rows(path, max_rows):
        title = first_value(row, TITLE_COLUMNS)
        description = first_value(row, DESCRIPTION_COLUMNS)
        mode = normalize_ag_news_mode(first_value(row, CATEGORY_COLUMNS))
        text = clean_text(f"{title}. {description}".strip(" ."))
        if text and mode:
            examples.append({"text": text, "label": mode})
    return examples


def load_project_labeled_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print_missing_dataset(
            path,
            "project labeled JSONL",
            "title, snippet, sentiment_label, event_tags_label",
        )
        return []

    examples = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                {
                    "text": build_model_text(row),
                    "sentiment_label": str(row.get("sentiment_label", "")).strip(),
                    "event_label": primary_event_label(row),
                    "raw": row,
                }
            )
    return examples


def read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        raw_rows = list(csv.reader(file))
    if not raw_rows:
        return []

    first_row = [normalize_key(value) for value in raw_rows[0]]
    known_columns = set(
        TEXT_COLUMNS
        + SENTIMENT_COLUMNS
        + TITLE_COLUMNS
        + DESCRIPTION_COLUMNS
        + CATEGORY_COLUMNS
    )
    if any(column in known_columns for column in first_row):
        fieldnames = first_row
        data_rows = raw_rows[1:]
    else:
        fieldnames = default_fieldnames(path)
        data_rows = raw_rows

    rows = []
    for raw_row in data_rows:
        row = {
            fieldnames[index]: value
            for index, value in enumerate(raw_row)
            if index < len(fieldnames)
        }
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def first_value(row: dict, columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column)
        if value:
            return str(value).strip()
    return ""


def normalize_label(value: str) -> str:
    return clean_text(value).lower()


def normalize_ag_news_mode(value: str) -> str:
    normalized = clean_text(value).lower()
    return AG_NEWS_MODE_MAP.get(normalized, "")


def primary_event_label(row: dict) -> str:
    labels = row.get("event_tags_label", [])
    if isinstance(labels, list) and labels:
        return str(labels[0]).strip()
    if isinstance(labels, str):
        return labels.split(",", 1)[0].strip()
    return ""


def external_max_rows(value: int) -> int | None:
    if value <= 0:
        return None
    return value


def print_missing_dataset(path: Path, dataset_name: str, columns: str) -> None:
    print(
        f"Missing {dataset_name}: {path}\n"
        f"Place the local dataset under {DEFAULT_EXTERNAL_DIR} and include columns: {columns}."
    )


def normalize_key(value: str | None) -> str:
    return clean_text(value or "").lower().replace(" ", "_")


def default_fieldnames(path: Path) -> list[str]:
    name = path.name.lower()
    if "ag" in name:
        return ["label", "title", "description"]
    return ["label", "text"]
