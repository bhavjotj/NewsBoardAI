# NewsBoardAI

NewsBoardAI is a local-first AI/ML news intelligence dashboard. It fetches a small number of recent Google News RSS results for a topic and turns them into a compact visual brief with signal, sentiment, event tags, confidence, possible impact, and source cards.

It runs as both a React web app and a local Chrome side-panel extension.

## Project Overview

Search for a company, stock, game, sports team, political issue, product, or trend. NewsBoardAI sends the query to a FastAPI backend, fetches recent headlines, analyzes the coverage with hybrid rule-based and local ML logic, and returns a dashboard-friendly response for the UI.

## Key Features

- Recent news retrieval from Google News RSS
- Compact React dashboard and Chrome side-panel extension
- Hybrid rule-based analysis with safe fallbacks
- scikit-learn TF-IDF baseline models for sentiment, event, and topic signals
- Optional PyTorch broad-topic classifier for business, sports, tech, and politics/general support
- Optional local Ollama brief/impact wording with template fallback
- Cautious dashboard fields: overall signal, sentiment, event tags, confidence, possible impact, and sources
- Local JSONL data collection, labeling, and training scripts

## Tech Stack

| Area | Tools |
| --- | --- |
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Frontend | React, TypeScript, Vite, Tailwind CSS, lucide-react |
| Extension | Chrome Manifest V3, Chrome Side Panel API |
| News/Data | Google News RSS, local JSONL, optional local CSV datasets |
| ML | scikit-learn, TF-IDF, Logistic Regression, PyTorch, EmbeddingBag, joblib |

## Architecture Flow

```text
Search query
-> FastAPI backend
-> Google News RSS
-> preprocessing
-> hybrid analyzer + local ML signals
-> structured dashboard response
-> React web app or Chrome side panel
```

Detailed docs:

- [ML baseline](docs/ML_BASELINE.md)
- [PyTorch event/topic model](docs/PYTORCH_EVENT_MODEL.md)
- [Ollama local brief generation](docs/OLLAMA_BRIEF.md)
- [Chrome extension](docs/CHROME_EXTENSION.md)

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r apps/api/requirements.txt
```

```bash
cd apps/web
npm install
```

## Run Backend

Run backend tests:

```bash
PYTHONPATH=apps/api .venv/bin/python -m pytest apps/api/tests
```

Start the API:

```bash
.venv/bin/uvicorn app.main:app --reload --app-dir apps/api
```

## Run Frontend

```bash
cd apps/web
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173` with the backend running.

## Build/Load Chrome Extension

```bash
cd apps/web
npm run build:extension
```

Then open `chrome://extensions`, enable Developer Mode, choose Load unpacked, and select `apps/web/dist-extension`.

## ML Training

Train the scikit-learn baseline:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py \
  --project-data data/labeled/news_labeled.jsonl
```

Train the optional PyTorch broad-topic model:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_torch_event_model.py \
  --project-data data/labeled/news_labeled.jsonl \
  --topic-data data/external/ag_news.csv \
  --output-dir models/torch_event \
  --max-rows 10000
```

Local data and model artifacts under `data/` and `models/` are ignored by git.

## Limitations

- The backend must run locally for the web app and extension.
- Google News RSS is used lightly for local MVP-style retrieval.
- Analysis is signal-based and can be wrong or uncertain.
- Financial output is not financial advice.
- The PyTorch model is a broad topic signal, not the final source of truth.
- Local datasets and trained models are not committed.

## Future Improvements

- Better duplicate story clustering
- More labeled NewsBoardAI examples
- FinBERT-style business sentiment
- Sentence-transformer similarity for source grouping
- Local LLM/Ollama-assisted brief and impact wording
- More polished Chrome extension interactions
