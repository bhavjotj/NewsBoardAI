# NewsBoardAI ML Baseline

This baseline trains small local classifiers for sentiment, event tags, and topic
mode. It stays lightweight: TF-IDF features, Logistic Regression, no neural
network dependencies, and no paid APIs.

External datasets help because live Google News RSS calls produce only a few
examples at a time. Larger local datasets can improve basic language coverage
without scraping more articles.

## Hybrid Predictor

`apps/api/app/services/baseline_predictor.py` wraps the trained baseline models
with a small domain-aware post-processing layer. It:

- loads any available baseline models from `models/baseline/`
- uses `predict_proba` when available and returns label confidence
- keeps raw model predictions separate from final adjusted predictions
- adds notes when confidence is low or a model is missing
- uses reusable domain lexicons for finance, risk, growth, sports, gaming,
  politics, and product launches

The post-processing layer is intentionally conservative. It adjusts predictions
mainly when confidence is low or when multiple related terms agree. This helps
with gaps in external datasets, especially AG News not having a gaming class.
For example, several terms such as `Nintendo`, `Switch`, `console`, `review`,
or `game` can move `topic_mode` toward `gaming` even if the topic model only
knows broad AG News classes.

## Setup

```bash
.venv/bin/python -m pip install -r apps/api/requirements.txt
```

Keep downloaded datasets under:

```text
data/external/
```

`data/` and `models/` are ignored by git.

## Expected Formats

Financial PhraseBank style CSV:

```csv
sentence,sentiment
"Profit rose after earnings","positive"
"Revenue was flat","neutral"
```

Supported text columns: `sentence`, `text`.
Supported label columns: `sentiment`, `label`.
Labels: `positive`, `neutral`, `negative`.

AG News style CSV:

```csv
category,title,description
Sports,"Raptors win","Team closes game strongly"
Business,"Stocks rise","Investors watch earnings"
```

Supported title columns: `title`, `headline`.
Supported body columns: `description`, `text`, `snippet`.
Supported category columns: `category`, `label`, `class`.
Mapped modes: `World -> politics`, `Sports -> sports`, `Business -> business`,
`Sci/Tech -> general`.

Project JSONL:

```json
{"title":"Nintendo Switch 2 launch details emerge","snippet":"Console pricing drew attention.","sentiment_label":"positive","event_tags_label":["gaming","launch"],"importance_label":"medium","impact_label":"positive"}
```

## Train With Project Data Only

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py \
  --project-data data/labeled/news_labeled.jsonl
```

This trains:

- `models/baseline/sentiment_model.joblib`
- `models/baseline/event_model.joblib`

## Train With Financial PhraseBank

Place a local CSV at `data/external/financial_phrasebank.csv`, then run:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py \
  --project-data data/labeled/news_labeled.jsonl \
  --sentiment-data data/external/financial_phrasebank.csv
```

When `--sentiment-data` is provided, the sentiment model uses that external
dataset instead of project labels. The event model still uses project labels.

## Train With AG News

Place a local CSV at `data/external/ag_news.csv`, then run:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py \
  --project-data data/labeled/news_labeled.jsonl \
  --topic-data data/external/ag_news.csv \
  --max-rows 10000
```

This adds:

- `models/baseline/topic_model.joblib`

Use `--max-rows 5000` or `--max-rows 10000` to keep local training fast. Use
`--max-rows 0` for unlimited external rows.

## Predict

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/predict_baseline.py \
  --title "Netflix shares rise after earnings" \
  --snippet "Revenue growth and pricing updates drew investor attention."
```

The prediction script prints raw model predictions, final adjusted predictions,
confidence values when available, and any post-processing notes.

## Current Limitations

- The project-labeled set is still tiny, so project-only models are rough.
- External labels are not identical to NewsBoardAI labels.
- AG News has broad categories, so topic mode is a coarse baseline.
- AG News does not include a gaming class, so gaming detection is currently a
  rule-assisted fallback.
- The event model still depends on project labels and predicts only the first
  event tag.
- Confidence values from small or mismatched datasets can be overconfident.
- Lexicon post-processing can improve obvious cases, but it is not a substitute
  for a larger labeled NewsBoardAI dataset.
- The scikit-learn baseline is integrated into the backend hybrid analyzer when
  local model files are available.
- This is still not the final neural network or transformer stage.
