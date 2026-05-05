# NewsBoardAI

NewsBoardAI is a local-first ML news intelligence dashboard. It fetches a small number of recent Google News RSS results for a topic, analyzes the coverage with local ML models, and uses a local Ollama LLM to turn the result into a clean visual brief with signal, sentiment, event tags, confidence, possible impact, and source cards.

It runs as both a React web app and a local Chrome side-panel extension.

## Project Overview

Search for a company, stock, game, sports team, political issue, product, or trend. NewsBoardAI sends the query to a FastAPI backend, fetches recent headlines, analyzes the coverage with hybrid rule-based and local ML logic, then uses Ollama to generate clearer dashboard wording for the brief and possible impact.

The goal is not to replace full news reading or make guaranteed predictions. The app is designed to quickly turn noisy recent headlines into a short, visual, and cautious news signal.

## Key Features

- Recent news retrieval from Google News RSS
- Compact React dashboard and Chrome side-panel extension
- Hybrid rule-based analysis with safe fallbacks
- scikit-learn TF-IDF baseline models for sentiment, event, and topic signals
- PyTorch broad-topic classifier for business, sports, tech, and politics/general support
- Local Ollama-powered brief and impact generation with template fallback
- Cautious dashboard fields: overall signal, sentiment, event tags, confidence, possible impact, and sources
- Local JSONL data collection, labeling, and training scripts

## Tech Stack

| Area | Tools |
| --- | --- |
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Frontend | React, TypeScript, Vite, Tailwind CSS, lucide-react |
| Extension | Chrome Manifest V3, Chrome Side Panel API |
| News/Data | Google News RSS, local JSONL, local CSV datasets |
| ML | scikit-learn, TF-IDF, Logistic Regression, PyTorch, EmbeddingBag, joblib |
| Local LLM | Ollama, llama3.2 or compatible local model |

## Architecture Flow

```text
Search query
-> FastAPI backend
-> Google News RSS
-> preprocessing
-> hybrid analyzer + scikit-learn + PyTorch topic signal
-> Ollama brief/impact wording
-> structured dashboard response
-> React web app or Chrome side panel

Detailed docs:

* [ML baseline](docs/ML_BASELINE.md)
* [PyTorch event/topic model](docs/PYTORCH_EVENT_MODEL.md)
* [Ollama local brief generation](docs/OLLAMA_BRIEF.md)
* [Chrome extension](docs/CHROME_EXTENSION.md)

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

## Run Ollama

NewsBoardAI uses Ollama locally to generate clearer dashboard brief and impact wording.

```bash
ollama pull llama3.2
ollama serve
```

The app still falls back to template wording if Ollama is unavailable, but Ollama is the intended local LLM layer for the full dashboard experience.

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

Open `http://127.0.0.1:5173` with Ollama and the backend running.

## Build/Load Chrome Extension

```bash
cd apps/web
npm run build:extension
```

Then open `chrome://extensions`, enable Developer Mode, choose Load unpacked, and select `apps/web/dist-extension`.

Keep Ollama and the backend running locally while using the extension.

## ML Training

Train the scikit-learn baseline:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py \
  --project-data data/labeled/news_labeled.jsonl
```

Train the PyTorch broad-topic model:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_torch_event_model.py \
  --project-data data/labeled/news_labeled.jsonl \
  --topic-data data/external/ag_news.csv \
  --output-dir models/torch_event \
  --max-rows 10000
```

Local data and model artifacts under `data/` and `models/` are ignored by git.

## Limitations

* Ollama and the backend must run locally for the full app experience.
* Google News RSS is used lightly for local MVP-style retrieval.
* Analysis is signal-based and can be wrong or uncertain.
* Financial output is not financial advice.
* The PyTorch model is a broad topic signal, not the final source of truth.
* Ollama improves wording, but it does not change the underlying analysis fields.

## Future Improvements

* Better duplicate story clustering
* More labeled NewsBoardAI examples
* FinBERT-style business sentiment
* Sentence-transformer similarity for source grouping
* Stronger Ollama prompting and local model options
* More polished Chrome extension interactions