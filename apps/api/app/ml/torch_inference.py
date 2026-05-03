from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from app.ml.torch_text_dataset import encode_text
from app.ml.torch_text_model import NeuralTextClassifier
from app.services.ml_preprocessing import build_model_text


@dataclass(frozen=True)
class TorchLabelProbability:
    label: str
    probability: float


@dataclass(frozen=True)
class TorchPrediction:
    input_text: str
    label: str
    confidence: float
    top_labels: list[TorchLabelProbability]


class TorchEventPredictor:
    def __init__(
        self,
        model: NeuralTextClassifier,
        vocab: dict[str, int],
        id_to_label: dict[int, str],
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.id_to_label = id_to_label
        self.model.eval()

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "TorchEventPredictor | None":
        # Return None instead of raising so backend inference can fail open.
        model_path = model_dir / "model.pt"
        metadata_path = model_dir / "metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vocab = {str(token): int(index) for token, index in metadata["vocab"].items()}
        id_to_label = {
            int(index): str(label) for index, label in metadata["id_to_label"].items()
        }
        model = NeuralTextClassifier(
            vocab_size=len(vocab),
            num_classes=len(id_to_label),
            embedding_dim=int(metadata.get("embedding_dim", 64)),
            hidden_dim=int(metadata.get("hidden_dim", 64)),
            dropout=float(metadata.get("dropout", 0.2)),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return cls(model=model, vocab=vocab, id_to_label=id_to_label)

    def predict(self, title: str, snippet: str = "") -> TorchPrediction:
        input_text = build_model_text({"title": title, "snippet": snippet})
        token_ids = encode_text(input_text, self.vocab)
        input_tensor = torch.tensor(token_ids, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)

        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_tensor, offsets), dim=1)[0]

        top_count = min(3, probabilities.numel())
        top_values, top_indexes = torch.topk(probabilities, k=top_count)
        top_labels = [
            TorchLabelProbability(
                label=self.id_to_label[int(index.item())],
                probability=float(value.item()),
            )
            for value, index in zip(top_values, top_indexes)
        ]
        best = top_labels[0]
        return TorchPrediction(
            input_text=input_text,
            label=best.label,
            confidence=best.probability,
            top_labels=top_labels,
        )
