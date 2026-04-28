# NewsBoardAI ML Baseline

This baseline trains small local classifiers for sentiment, event tags, and topic
mode. It stays lightweight: TF-IDF features, Logistic Regression, no neural
network dependencies, and no paid APIs.

External datasets help because live Google News RSS calls produce only a few
examples at a time. Larger local datasets can improve basic language coverage
without scraping more articles.

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

The prediction script prints whichever trained models exist: sentiment, event
tag, and topic mode.

## Current Limitations

- The project-labeled set is still tiny, so project-only models are rough.
- External labels are not identical to NewsBoardAI labels.
- AG News has broad categories, so topic mode is a coarse baseline.
- The event model still depends on project labels and predicts only the first
  event tag.
- These models are not integrated into backend inference yet.
