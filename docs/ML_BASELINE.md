# NewsBoardAI ML Baseline

This baseline trains small local classifiers from `data/labeled/news_labeled.jsonl`.
It is intentionally simple and does not use transformers, TensorFlow, PyTorch, or paid APIs.

## Setup

```bash
.venv/bin/python -m pip install -r apps/api/requirements.txt
```

## Train

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py
```

The script trains:

- `models/baseline/sentiment_model.joblib`
- `models/baseline/event_model.joblib`

Both `models/` and JSONL data files are ignored by git.

## Predict

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/predict_baseline.py \
  --title "Netflix shares rise after earnings" \
  --snippet "Revenue growth and pricing updates drew investor attention."
```

## Current Limitations

The current labeled set has only 24 examples, so the training script warns and trains on all available data instead of reporting a reliable train/test score. Treat predictions as a rough local baseline, not a production model.
