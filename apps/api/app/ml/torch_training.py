from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from app.ml.torch_text_dataset import (
    TextExample,
    TorchTextDataset,
    build_vocab,
    collate_text_batch,
)
from app.ml.torch_text_model import NeuralTextClassifier
from app.services.datasets import (
    CATEGORY_COLUMNS,
    DESCRIPTION_COLUMNS,
    TITLE_COLUMNS,
    external_max_rows,
    first_value,
    load_project_labeled_jsonl,
    read_csv_rows,
)
from app.utils.text import clean_text

AG_NEWS_LABELS = {
    "1": "politics",
    "2": "sports",
    "3": "business",
    "4": "tech",
    "world": "politics",
    "sports": "sports",
    "business": "business",
    "sci/tech": "tech",
    "sci-tech": "tech",
    "science/technology": "tech",
    "tech": "tech",
}

PROJECT_EVENT_LABELS = {
    "business",
    "gaming",
    "general",
    "launch",
    "legal",
    "market",
    "politics",
    "product",
    "review",
    "risk",
    "sports",
    "tech",
    "workforce",
}

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.2
MIN_VALIDATION_EXAMPLES = 30


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_training_examples(
    project_data: Path | None,
    topic_data: Path | None,
    max_rows: int,
) -> list[TextExample]:
    examples: list[TextExample] = []
    max_external_rows = external_max_rows(max_rows)

    if topic_data:
        examples.extend(load_ag_news_for_torch(topic_data, max_rows=max_external_rows))

    if project_data:
        examples.extend(load_project_events_for_torch(project_data))

    return examples


def load_ag_news_for_torch(
    path: Path,
    max_rows: int | None = None,
) -> list[TextExample]:
    if not path.exists():
        print(
            f"Missing AG News CSV: {path}\n"
            "Place it under data/external/ with title, description/text, and category/label columns."
        )
        return []

    examples = []
    for row in read_csv_rows(path, max_rows=max_rows):
        title = first_value(row, TITLE_COLUMNS)
        description = first_value(row, DESCRIPTION_COLUMNS)
        label = normalize_ag_news_label(first_value(row, CATEGORY_COLUMNS))
        text = clean_text(f"{title}. {description}".strip(" ."))
        if text and label:
            examples.append(TextExample(text=text, label=label))
    return examples


def load_project_events_for_torch(path: Path) -> list[TextExample]:
    rows = load_project_labeled_jsonl(path)
    examples = []
    for row in rows:
        label = normalize_project_event_label(row.get("event_label", ""))
        if row.get("text") and label:
            examples.append(TextExample(text=row["text"], label=label))
    return examples


def normalize_ag_news_label(value: str) -> str:
    return AG_NEWS_LABELS.get(clean_text(value).lower(), "")


def normalize_project_event_label(value: str) -> str:
    label = clean_text(str(value)).lower().replace(" ", "_")
    return label if label in PROJECT_EVENT_LABELS else ""


def label_counts(examples: list[TextExample]) -> Counter[str]:
    return Counter(example.label for example in examples)


def can_validate(examples: list[TextExample]) -> bool:
    counts = label_counts(examples)
    return len(examples) >= MIN_VALIDATION_EXAMPLES and min(counts.values()) >= 2


def train_and_save(
    examples: list[TextExample],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    set_seed(seed)
    counts = label_counts(examples)
    print(f"Label counts: {dict(sorted(counts.items()))}")

    if not examples:
        print("Warning: no training examples found; skipping PyTorch training.")
        return
    if len(counts) < 2:
        print("Warning: at least two labels are required; skipping PyTorch training.")
        return
    if len(examples) < MIN_VALIDATION_EXAMPLES:
        print(
            f"Warning: only {len(examples)} examples found. "
            "Training will run, but validation and generalization are limited."
        )

    labels = sorted(counts)
    label_to_id = {label: index for index, label in enumerate(labels)}
    vocab = build_vocab(example.text for example in examples)
    dataset = TorchTextDataset(examples, vocab=vocab, label_to_id=label_to_id)

    train_dataset = dataset
    validation_dataset = None
    if can_validate(examples):
        validation_size = max(1, int(len(dataset) * 0.2))
        train_size = len(dataset) - validation_size
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_text_batch,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_text_batch,
        )
        if validation_dataset is not None
        else None
    )

    model = NeuralTextClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_to_id),
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        dropout=DEFAULT_DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        message = f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f}"
        if validation_loader is not None:
            message += f" - validation accuracy: {evaluate_accuracy(model, validation_loader):.3f}"
        print(message)

    save_artifacts(
        output_dir=output_dir,
        model=model,
        vocab=vocab,
        label_to_id=label_to_id,
        counts=counts,
        training_examples=len(examples),
        epochs=epochs,
    )


def train_epoch(
    model: NeuralTextClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for input_ids, offsets, labels in loader:
        optimizer.zero_grad()
        logits = model(input_ids, offsets)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
    return total_loss / max(total_examples, 1)


def evaluate_accuracy(model: NeuralTextClassifier, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, offsets, labels in loader:
            predictions = model(input_ids, offsets).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.size(0)
    return correct / max(total, 1)


def save_artifacts(
    output_dir: Path,
    model: NeuralTextClassifier,
    vocab: dict[str, int],
    label_to_id: dict[str, int],
    counts: Counter[str],
    training_examples: int,
    epochs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    metadata = {
        "model_type": "embedding_bag_mlp",
        "vocab": vocab,
        "label_to_id": label_to_id,
        "id_to_label": {str(index): label for label, index in label_to_id.items()},
        "label_counts": dict(counts),
        "training_examples": training_examples,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "text_builder": "title_plus_nonduplicate_snippet",
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "dropout": DEFAULT_DROPOUT,
        "epochs": epochs,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved PyTorch model to {output_dir}")
