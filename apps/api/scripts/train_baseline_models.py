from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.services.datasets import (
    external_max_rows,
    load_ag_news,
    load_financial_phrasebank,
    load_project_labeled_jsonl,
)

DEFAULT_PROJECT_DATA = PROJECT_ROOT / "data" / "labeled" / "news_labeled.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
DEFAULT_MAX_ROWS = 10000
MIN_SPLIT_EXAMPLES = 30


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NewsBoardAI baselines.")
    parser.add_argument("--project-data", type=Path, default=DEFAULT_PROJECT_DATA)
    parser.add_argument("--sentiment-data", type=Path)
    parser.add_argument("--topic-data", type=Path)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    max_rows = external_max_rows(args.max_rows)
    project_examples = load_project_labeled_jsonl(args.project_data)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # External sentiment data wins when provided; project labels still train events.
    sentiment_examples = sentiment_training_examples(args, project_examples, max_rows)
    train_and_save(
        name="sentiment",
        examples=sentiment_examples,
        output_path=args.model_dir / "sentiment_model.joblib",
    )

    event_examples = [
        {"text": example["text"], "label": example["event_label"]}
        for example in project_examples
        if example.get("event_label")
    ]
    train_and_save(
        name="event",
        examples=event_examples,
        output_path=args.model_dir / "event_model.joblib",
    )

    if args.topic_data:
        topic_examples = load_ag_news(args.topic_data, max_rows=max_rows)
        train_and_save(
            name="topic",
            examples=topic_examples,
            output_path=args.model_dir / "topic_model.joblib",
        )


def sentiment_training_examples(args, project_examples, max_rows):
    if args.sentiment_data:
        return load_financial_phrasebank(args.sentiment_data, max_rows=max_rows)
    return [
        {"text": example["text"], "label": example["sentiment_label"]}
        for example in project_examples
        if example.get("sentiment_label")
    ]


def train_and_save(name: str, examples: list[dict], output_path: Path) -> None:
    texts = [str(example["text"]) for example in examples if example.get("label")]
    labels = [str(example["label"]) for example in examples if example.get("label")]
    counts = Counter(labels)

    print(f"\n{name.title()} label counts: {dict(sorted(counts.items()))}")
    if not labels:
        print(f"Warning: no labeled {name} examples available; skipping.")
        return
    if len(counts) < 2:
        print(f"Warning: {name} has one class only; skipping model training.")
        return

    model = make_pipeline()
    # Small local datasets are expected, so train without a split when needed.
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


if __name__ == "__main__":
    main()
