from __future__ import annotations

import argparse
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))

from app.ml.torch_training import load_training_examples, train_and_save

DEFAULT_PROJECT_DATA = PROJECT_ROOT / "data" / "labeled" / "news_labeled.jsonl"
DEFAULT_TOPIC_DATA = PROJECT_ROOT / "data" / "external" / "ag_news.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "torch_event"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a local PyTorch topic/event classifier."
    )
    parser.add_argument("--project-data", type=Path, default=DEFAULT_PROJECT_DATA)
    parser.add_argument("--topic-data", type=Path, default=DEFAULT_TOPIC_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_training_examples(
        project_data=args.project_data,
        topic_data=args.topic_data,
        max_rows=args.max_rows,
    )
    train_and_save(
        examples=examples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
