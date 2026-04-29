# NewsBoardAI

NewsBoardAI is a local-first AI/ML news intelligence tool that turns a small set of recent news results into a compact visual dashboard.

Search a topic, company, stock, game, sports team, political issue, or trend. NewsBoardAI fetches recent coverage, analyzes the signal, and returns a short dashboard with sentiment, event tags, source cards, possible impact, confidence, and detected topic mode.

This is not a full news website and it is not a prediction engine. It is a lightweight personal news intelligence panel that highlights possible signals cautiously.

## What It Does

1. Run the FastAPI backend locally.
2. Open the React web app or Chrome side-panel extension.
3. Search a topic such as `Tesla`, `Nintendo Switch 2`, `Toronto Raptors`, or `Bitcoin`.
4. Review a compact dashboard built from recent Google News RSS results.

The dashboard is designed for quick scanning rather than long-form summaries.

## Key Features

- Recent news retrieval from Google News RSS
- Compact React dashboard UI
- Chrome Manifest V3 side-panel extension
- Sentiment, event tag, and topic/mode detection
- Hybrid ML plus rule-based analysis with safe fallbacks
- Confidence level and cautious possible impact wording
- Source cards with title, publisher, date, snippet, and link
- Local JSONL data collection for future labeling
- Local labeling and baseline model training scripts
- Optional external datasets under `data/external/`

## Architecture

```text
Search query
-> FastAPI backend
-> Google News RSS fetcher
-> text preprocessing
-> hybrid analyzer
-> structured dashboard response
-> React web app or Chrome side panel
```

The backend keeps the response shape small and dashboard-friendly. If local ML models are unavailable, the app falls back to rule-based analysis instead of crashing.

## Tech Stack

| Area | Tools |
| --- | --- |
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Frontend | React, TypeScript, Vite, Tailwind CSS, lucide-react |
| Extension | Chrome Extension Manifest V3, Chrome Side Panel API |
| News/Data | Google News RSS, local JSONL examples, optional local CSV datasets |
| ML | scikit-learn, TF-IDF, Logistic Regression, PyTorch, EmbeddingBag, joblib |

## Project Structure

```text
apps/api/      FastAPI backend, analysis services, ML scripts, tests
apps/web/      React TypeScript frontend and Chrome extension build
docs/          Project, ML baseline, and extension documentation
data/          Local raw/labeled/external datasets ignored by git
models/        Local trained baseline models ignored by git
```

## Local Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:

```bash
python -m pip install -r apps/api/requirements.txt
```

Install frontend dependencies:

```bash
cd apps/web
npm install
```

## Run Backend

From the repository root, run tests:

```bash
PYTHONPATH=apps/api .venv/bin/python -m pytest apps/api/tests
```

Start the API:

```bash
.venv/bin/uvicorn app.main:app --reload --app-dir apps/api
```

The backend runs at:

```text
http://127.0.0.1:8000
```

## Run Frontend Web App

From the frontend folder:

```bash
cd apps/web
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

Keep the backend running while using the frontend.

## Chrome Side-Panel Extension

Build the local extension:

```bash
cd apps/web
npm run build:extension
```

Load it in Chrome:

1. Open `chrome://extensions`.
2. Enable Developer Mode.
3. Click Load unpacked.
4. Select `apps/web/dist-extension`.
5. Keep the FastAPI backend running locally.
6. Click the NewsBoardAI extension icon to open the side panel.

This is local-only and does not require a Chrome Web Store login.

## ML Baseline Usage

NewsBoardAI supports lightweight local baseline models. The scikit-learn baseline uses TF-IDF features with Logistic Regression classifiers for sentiment, event tags, and topic/mode prediction.

The project also includes an optional PyTorch neural broad-topic classifier. The live dashboard can use it as a supporting signal through `use_torch: true`, while the hybrid analyzer still handles specific event tags and rule-based fallbacks.

Models are trained locally and saved under:

```text
models/baseline/
```

Local data and model files are ignored by git, including `data/external/`, `data/raw/`, `data/labeled/`, and `models/`.

Train with project-labeled examples only:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py --project-data data/labeled/news_labeled.jsonl
```

Train with optional external datasets:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_baseline_models.py \
  --project-data data/labeled/news_labeled.jsonl \
  --sentiment-data data/external/financial_phrasebank.csv \
  --topic-data data/external/ag_news.csv \
  --max-rows 10000
```

Supported external datasets include Financial PhraseBank-style sentiment CSVs and AG News-style topic CSVs placed under `data/external/`.

Test a baseline prediction:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/predict_baseline.py \
  --title "Netflix shares rise after earnings" \
  --snippet "Revenue growth and pricing updates drew investor attention."
```

Train the optional PyTorch broad-topic model:

```bash
PYTHONPATH=apps/api .venv/bin/python apps/api/scripts/train_torch_event_model.py \
  --project-data data/labeled/news_labeled.jsonl \
  --topic-data data/external/ag_news.csv \
  --output-dir models/torch_event \
  --max-rows 10000
```

## Example API Request

```bash
curl -X POST "http://127.0.0.1:8000/api/news/dashboard" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tesla",
    "max_results": 3,
    "use_real_news": true,
    "use_ml": true,
    "use_torch": true
  }'
```

The response includes fields such as `topic`, `data_source`, `detected_mode`, `overall_signal`, `sentiment`, `event_tags`, `sources`, `confidence`, `possible_impact`, and `analysis_source`.

## Notes and Limitations

- The backend must be running locally for the web app and Chrome extension.
- Google News RSS is used lightly for local MVP news retrieval.
- Analysis is signal-based and not guaranteed.
- Financial dashboard output is not financial advice.
- Baseline ML and PyTorch models are not final transformer models.
- External datasets can improve coverage, but labels may not perfectly match NewsBoardAI's dashboard concepts.
- `data/external/`, `data/raw/`, `data/labeled/`, and `models/` are intentionally not committed.

## Future Improvements

- Better product and AI event aggregation
- FinBERT for business sentiment
- Sentence Transformers for duplicate story clustering
- Local LLM or Ollama-assisted summary generation
- More labeled NewsBoardAI examples
- Stronger importance scoring
- More polished Chrome extension interactions
