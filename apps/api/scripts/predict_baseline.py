from __future__ import annotations

import argparse
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))

import joblib

from app.services.ml_preprocessing import build_model_text

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

    example = {"title": args.title, "snippet": args.snippet}
    text = build_model_text(example)

    print(f"Text: {text}")
    predict_if_available("sentiment", args.model_dir / "sentiment_model.joblib", text)
    predict_if_available("event tag", args.model_dir / "event_model.joblib", text)
    predict_if_available("topic mode", args.model_dir / "topic_model.joblib", text)


def predict_if_available(label: str, model_path: Path, text: str) -> None:
    if not model_path.exists():
        return
    prediction = predict_one(model_path, text)
    print(f"Predicted {label}: {prediction}")


def predict_one(model_path: Path, text: str) -> str:
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}. Train models first.")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    return str(model.predict([text])[0])


if __name__ == "__main__":
    main()
