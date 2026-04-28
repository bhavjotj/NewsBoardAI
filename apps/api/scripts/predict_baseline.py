from __future__ import annotations

import argparse
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))

from app.services.baseline_predictor import BaselinePredictor, LabelPrediction

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with baseline models.")
    parser.add_argument(
        "--title",
        default="Nintendo Switch 2 console launch details emerge",
    )
    parser.add_argument(
        "--snippet",
        default="New game trailers and hardware reports are drawing attention.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    result = BaselinePredictor(model_dir=args.model_dir).predict(
        title=args.title,
        snippet=args.snippet,
    )

    print(f"Text: {result.input_text}")
    print("\nRaw model predictions:")
    print_prediction("sentiment", result.raw_predictions["sentiment"])
    print_prediction("event tag", result.raw_predictions["event_tag"])
    print_prediction("topic mode", result.raw_predictions["topic_mode"])

    print("\nFinal predictions:")
    print_prediction("sentiment", result.sentiment)
    print_prediction("event tag", result.event_tag)
    print_prediction("topic mode", result.topic_mode)

    if result.adjustments:
        print("\nAdjustments:")
        for adjustment in result.adjustments:
            print(f"- {adjustment}")
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"- {note}")


def print_prediction(name: str, prediction: LabelPrediction) -> None:
    if prediction.label is None:
        return
    confidence = (
        "n/a" if prediction.confidence is None else f"{prediction.confidence:.2f}"
    )
    print(f"- {name}: {prediction.label} (confidence: {confidence})")


if __name__ == "__main__":
    main()
