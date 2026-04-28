from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.ml_preprocessing import build_model_text

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "labeled" / "news_labeled.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
MIN_SPLIT_EXAMPLES = 30


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NewsBoardAI baselines.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        print(f"No labeled examples found at {args.input}")
        return

    args.model_dir.mkdir(parents=True, exist_ok=True)
    train_and_save(
        name="sentiment",
        rows=rows,
        label_getter=lambda row: str(row.get("sentiment_label", "")).strip(),
        output_path=args.model_dir / "sentiment_model.joblib",
    )
    train_and_save(
        name="event",
        rows=rows,
        label_getter=primary_event_label,
        output_path=args.model_dir / "event_model.joblib",
    )


def train_and_save(name, rows, label_getter, output_path: Path) -> None:
    texts = []
    labels = []
    for row in rows:
        label = label_getter(row)
        text = build_model_text(row)
        if label and text:
            texts.append(text)
            labels.append(label)

    counts = Counter(labels)
    print(f"\n{name.title()} label counts: {dict(sorted(counts.items()))}")
    if not labels:
        print(f"Warning: no labeled {name} examples available; skipping.")
        return
    if len(counts) < 2:
        print(f"Warning: {name} has one class only; skipping model training.")
        return

    model = make_pipeline()
    if can_split(labels):
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels,
        )
        model.fit(train_texts, train_labels)
        predictions = model.predict(test_texts)
        print(f"{name.title()} accuracy: {accuracy_score(test_labels, predictions):.2f}")
        print(classification_report(test_labels, predictions, zero_division=0))
    else:
        print(
            f"Warning: only {len(labels)} usable {name} examples. "
            "Training on all data without a reliable train/test split."
        )
        model.fit(texts, labels)

    artifact = {
        "model": model,
        "label_counts": dict(counts),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_examples": len(labels),
        "text_builder": "title_plus_nonduplicate_snippet",
    }
    joblib.dump(artifact, output_path)
    print(f"Saved {name} model to {output_path}")


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=5000,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def can_split(labels: list[str]) -> bool:
    counts = Counter(labels)
    return len(labels) >= MIN_SPLIT_EXAMPLES and min(counts.values()) >= 2


def primary_event_label(row: dict) -> str:
    labels = row.get("event_tags_label", [])
    if isinstance(labels, list) and labels:
        return str(labels[0]).strip()
    if isinstance(labels, str):
        return labels.split(",", 1)[0].strip()
    return ""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
