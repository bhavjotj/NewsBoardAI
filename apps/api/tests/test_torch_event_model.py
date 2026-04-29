from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.ml.torch_inference import TorchEventPredictor
from app.ml.torch_text_dataset import (
    TextExample,
    TorchTextDataset,
    build_vocab,
    collate_text_batch,
    tokenize,
)
from app.ml.torch_text_model import NeuralTextClassifier


def test_tokenizer_and_vocab() -> None:
    tokens = tokenize("Nintendo Switch 2 review: console launch!")
    vocab = build_vocab(["Nintendo Switch 2 review", "Tesla stock rally"])

    assert tokens == ["nintendo", "switch", "2", "review", "console", "launch"]
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert "nintendo" in vocab


def test_dataset_encoding_and_collate() -> None:
    examples = [
        TextExample(text="Raptors playoff game", label="sports"),
        TextExample(text="Tesla earnings rise", label="business"),
    ]
    label_to_id = {"business": 0, "sports": 1}
    vocab = build_vocab(example.text for example in examples)
    dataset = TorchTextDataset(examples, vocab=vocab, label_to_id=label_to_id)

    token_ids, label_id = dataset[0]
    input_ids, offsets, labels = collate_text_batch([dataset[0], dataset[1]])

    assert token_ids
    assert label_id == 1
    assert input_ids.ndim == 1
    assert offsets.tolist() == [0, len(dataset[0][0])]
    assert labels.tolist() == [1, 0]


def test_model_forward_pass_shape() -> None:
    model = NeuralTextClassifier(vocab_size=20, num_classes=4)
    input_ids = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long)
    offsets = torch.tensor([0, 3], dtype=torch.long)

    logits = model(input_ids, offsets)

    assert tuple(logits.shape) == (2, 4)


def test_inference_missing_model_files_returns_none(tmp_path: Path) -> None:
    assert TorchEventPredictor.from_model_dir(tmp_path / "missing") is None
