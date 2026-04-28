import json

from app.services.datasets import (
    external_max_rows,
    load_ag_news,
    load_financial_phrasebank,
    load_project_labeled_jsonl,
)


def test_load_financial_phrasebank_style_csv(tmp_path) -> None:
    path = tmp_path / "financial_phrasebank.csv"
    path.write_text(
        "sentence,sentiment\n"
        "Profit rose after earnings,positive\n"
        "Revenue was flat,neutral\n",
        encoding="utf-8",
    )

    rows = load_financial_phrasebank(path)

    assert rows == [
        {"text": "Profit rose after earnings", "label": "positive"},
        {"text": "Revenue was flat", "label": "neutral"},
    ]


def test_load_ag_news_style_csv(tmp_path) -> None:
    path = tmp_path / "ag_news.csv"
    path.write_text(
        "category,title,description\n"
        "Sports,Raptors win,Team closes game strongly\n"
        "Business,Stocks rise,Investors watch earnings\n",
        encoding="utf-8",
    )

    rows = load_ag_news(path)

    assert rows == [
        {"text": "Raptors win. Team closes game strongly", "label": "sports"},
        {"text": "Stocks rise. Investors watch earnings", "label": "business"},
    ]


def test_load_project_labeled_jsonl(tmp_path) -> None:
    path = tmp_path / "news_labeled.jsonl"
    row = {
        "title": "Nintendo Switch 2 launch details emerge",
        "snippet": "Game trailers and console pricing drew attention.",
        "sentiment_label": "positive",
        "event_tags_label": ["gaming", "launch"],
        "importance_label": "medium",
        "impact_label": "positive",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows = load_project_labeled_jsonl(path)

    assert len(rows) == 1
    assert rows[0]["sentiment_label"] == "positive"
    assert rows[0]["event_label"] == "gaming"
    assert "Nintendo Switch 2" in rows[0]["text"]


def test_external_loader_applies_max_rows(tmp_path) -> None:
    path = tmp_path / "financial_phrasebank.csv"
    path.write_text(
        "sentence,sentiment\n"
        "First row,positive\n"
        "Second row,negative\n"
        "Third row,neutral\n",
        encoding="utf-8",
    )

    rows = load_financial_phrasebank(path, max_rows=external_max_rows(2))

    assert len(rows) == 2
