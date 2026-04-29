from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TORCH_MODEL_DIR = PROJECT_ROOT / "models" / "torch_event"


@dataclass(frozen=True)
class TorchTopicLabel:
    label: str
    confidence: float


@dataclass(frozen=True)
class TorchTopicPrediction:
    label: str | None
    confidence: float | None
    top_labels: list[TorchTopicLabel] = field(default_factory=list)
    available: bool = False
    error: str | None = None


class TorchTopicService:
    def __init__(self, model_dir: Path = DEFAULT_TORCH_MODEL_DIR) -> None:
        self.model_dir = model_dir
        self._predictor = None
        self._load_error: str | None = None
        self._loaded = False

    @property
    def available(self) -> bool:
        return self._load_predictor() is not None

    def predict(self, title: str, snippet: str = "") -> TorchTopicPrediction:
        predictor = self._load_predictor()
        if predictor is None:
            return TorchTopicPrediction(
                label=None,
                confidence=None,
                available=False,
                error=self._load_error,
            )

        try:
            result = predictor.predict(title=title, snippet=snippet)
        except Exception as error:
            return TorchTopicPrediction(
                label=None,
                confidence=None,
                available=True,
                error=f"PyTorch prediction failed: {error}",
            )

        return TorchTopicPrediction(
            label=result.label,
            confidence=round(result.confidence, 3),
            top_labels=[
                TorchTopicLabel(
                    label=item.label,
                    confidence=round(item.probability, 3),
                )
                for item in result.top_labels
            ],
            available=True,
        )

    def _load_predictor(self):
        if self._loaded:
            return self._predictor

        self._loaded = True
        try:
            from app.ml.torch_inference import TorchEventPredictor

            self._predictor = TorchEventPredictor.from_model_dir(self.model_dir)
        except ModuleNotFoundError:
            self._load_error = "PyTorch is not installed."
            self._predictor = None
        except Exception as error:
            self._load_error = f"PyTorch model could not be loaded: {error}"
            self._predictor = None

        if self._predictor is None and self._load_error is None:
            self._load_error = f"No PyTorch model found in {self.model_dir}."
        return self._predictor


@lru_cache(maxsize=1)
def get_torch_topic_service() -> TorchTopicService:
    return TorchTopicService()
