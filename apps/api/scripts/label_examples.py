from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "news_examples.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "labeled" / "news_labeled.jsonl"

SENTIMENT_CHOICES = {"positive", "neutral", "negative", "mixed"}
IMPORTANCE_CHOICES = {"low", "medium", "high"}
IMPACT_CHOICES = {"positive", "negative", "mixed", "unclear"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Label NewsBoardAI examples.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    examples = load_unlabeled_examples(args.input, args.output)
    if not examples:
        print("No unlabeled examples found.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for index, example in enumerate(examples, start=1):
            print(f"\nExample {index} of {len(examples)}")
            print(f"Query: {example.get('query', '')}")
            print(f"Mode: {example.get('detected_mode', '')}")
            print(f"Title: {example.get('title', '')}")
            print(f"Snippet: {example.get('snippet', '')}")
            print(f"Source: {example.get('source', '')}")

            labeled = dict(example)
            labeled["sentiment_label"] = prompt_choice(
                "Sentiment", SENTIMENT_CHOICES
            )
            labeled["event_tags_label"] = prompt_tags("Event tags")
            labeled["importance_label"] = prompt_choice(
                "Importance", IMPORTANCE_CHOICES
            )
            labeled["impact_label"] = prompt_choice("Impact", IMPACT_CHOICES)

            output_file.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            output_file.flush()

    print(f"\nSaved labels to {args.output}")


def load_unlabeled_examples(input_path: Path, output_path: Path) -> list[dict]:
    if not input_path.exists():
        return []

    labeled_keys = load_labeled_keys(output_path)
    examples = []
    for row in read_jsonl(input_path):
        if example_key(row) in labeled_keys:
            continue
        if is_unlabeled(row):
            examples.append(row)
    return examples


def load_labeled_keys(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    return {example_key(row) for row in read_jsonl(output_path)}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_unlabeled(row: dict) -> bool:
    return not (
        row.get("sentiment_label")
        or row.get("event_tags_label")
        or row.get("importance_label")
        or row.get("impact_label")
    )


def example_key(row: dict) -> tuple[str, str]:
    return (str(row.get("query", "")), str(row.get("url", "")))


def prompt_choice(label: str, choices: set[str]) -> str:
    options = ", ".join(sorted(choices))
    while True:
        value = input(f"{label} ({options}): ").strip().lower()
        if value in choices:
            return value
        print(f"Please enter one of: {options}")


def prompt_tags(label: str) -> list[str]:
    value = input(f"{label} comma-separated: ").strip()
    return [tag.strip().lower() for tag in value.split(",") if tag.strip()]


if __name__ == "__main__":
    main()
