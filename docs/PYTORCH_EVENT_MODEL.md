# NewsBoardAI PyTorch Event Model

The PyTorch event model is a lightweight neural upgrade for NewsBoardAI's ML layer. It predicts broad topic or event labels from a news title and snippet, while staying local-first and CPU-friendly.

This model does not replace the existing scikit-learn baseline or the rule-based dashboard analyzer. It is a separate training and prediction path that can later be integrated into the dashboard when enough local data exists.

## Why It Exists

The scikit-learn baseline is fast and practical, but it relies on TF-IDF features and linear decision boundaries. The PyTorch model adds a small neural classifier that can learn word embeddings from local examples without downloading large pretrained models.

It is meant to improve the project story around:

- local neural text classification
- event/topic classification
- model checkpoints and metadata
- CPU-friendly training
- future dashboard inference upgrades

## Model Design

The model uses:

- simple lowercase tokenization
- a vocabulary built from local training data
- `EmbeddingBag` mean pooling
- one hidden layer with ReLU
- dropout
- a linear output layer
- class probabilities from softmax

It does not use torchtext, transformers, Hugging Face downloads, TensorFlow, or paid APIs.

## Training Data

The trainer can combine:

- AG News-style CSV data from `data/external/ag_news.csv`
- project labeled JSONL data from `data/labeled/news_labeled.jsonl`

AG News labels are mapped into broad app labels:

```text
World -> politics
Sports -> sports
Business -> business
Sci/Tech -> tech
```

Project labels use the first `event_tags_label` value when it matches known NewsBoardAI event labels such as `gaming`, `sports`, `business`, `market`, `product`, `launch`, `legal`, `risk`, `workforce`, `review`, `politics`, or `general`.

## Install

```bash
.venv/bin/python -m pip install -r apps/api/requirements.txt
```

This installs PyTorch along with the existing backend and baseline dependencies.

## Train

From the repository root:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_torch_event_model.py \
  --project-data data/labeled/news_labeled.jsonl \
  --topic-data data/external/ag_news.csv \
  --output-dir models/torch_event \
  --max-rows 10000 \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.001 \
  --seed 42
```

The script prints class counts, train loss, and validation accuracy when there are enough examples for a validation split.

Artifacts are saved locally:

```text
models/torch_event/model.pt
models/torch_event/metadata.json
```

## Predict

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/predict_torch_event.py \
  --title "Nintendo Switch 2 review roundup draws player interest" \
  --snippet "Console previews and game coverage highlight launch expectations." \
  --model-dir models/torch_event
```

The prediction script prints:

- input text
- predicted label
- confidence
- top 3 labels with probabilities

## Limitations

- The model learns only from local examples and does not use pretrained language knowledge.
- AG News provides broad topic categories, not detailed NewsBoardAI event labels.
- Project-labeled data is still small, so project-specific labels may be weak.
- Class imbalance can affect predictions.
- This model is not integrated into the live dashboard yet.
- Trained files under `models/` and datasets under `data/` are local artifacts and should not be committed.
