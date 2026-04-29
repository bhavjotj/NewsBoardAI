from __future__ import annotations

import argparse
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))

from app.ml.torch_inference import TorchEventPredictor

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "torch_event"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a topic/event label with the local PyTorch model."
    )
    parser.add_argument(
        "--title",
        default="Nintendo Switch 2 review roundup draws player interest",
    )
    parser.add_argument(
        "--snippet",
        default="Console previews and game coverage highlight launch expectations.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    predictor = TorchEventPredictor.from_model_dir(args.model_dir)
    if predictor is None:
        print(
            f"Missing PyTorch model files in {args.model_dir}. "
            "Train first with apps/api/scripts/train_torch_event_model.py."
        )
        return

    prediction = predictor.predict(title=args.title, snippet=args.snippet)
    print(f"Text: {prediction.input_text}")
    print(f"Predicted label: {prediction.label}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print("Top labels:")
    for item in prediction.top_labels:
        print(f"- {item.label}: {item.probability:.3f}")


if __name__ == "__main__":
    main()
