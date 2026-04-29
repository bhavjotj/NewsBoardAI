from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class TextExample:
    text: str
    label: str


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(
    texts: Iterable[str],
    max_size: int = 20000,
    min_freq: int = 1,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, count in counts.most_common(max_size - len(vocab)):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(token, UNK_ID) for token in tokenize(text)]
    return ids or [UNK_ID]


class TorchTextDataset(Dataset):
    def __init__(
        self,
        examples: list[TextExample],
        vocab: dict[str, int],
        label_to_id: dict[str, int],
    ) -> None:
        self.examples = examples
        self.vocab = vocab
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        example = self.examples[index]
        return encode_text(example.text, self.vocab), self.label_to_id[example.label]


def collate_text_batch(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    offsets = []
    flattened_tokens = []

    for token_ids, _ in batch:
        offsets.append(len(flattened_tokens))
        flattened_tokens.extend(token_ids or [UNK_ID])

    return (
        torch.tensor(flattened_tokens, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        labels,
    )
