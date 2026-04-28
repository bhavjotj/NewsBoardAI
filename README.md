# NewsBoardAI

NewsBoardAI is an AI news intelligence tool that turns the top recent news results for a searched topic into a compact visual dashboard.

The goal is to summarize only a small number of recent news items, classify their sentiment and event type, and display a clean dashboard with key signals, source links, and possible short-term implications.

## Planned Features

- Search any topic, company, stock, game, sports team, or public issue
- Fetch the top 2 to 5 recent news results
- Classify sentiment, event type, and importance
- Generate a clean visual news brief
- Display results in a Chrome extension side panel
- Use local or free tools where possible

## Tech Stack

- Frontend: React, Tailwind CSS, Chrome Extension Side Panel
- Backend: FastAPI, Python
- ML: Hugging Face Transformers, TensorFlow or PyTorch
- Data: Google News RSS, GDELT, RSS feeds
- Storage: SQLite for local project data

## Project Status

Initial setup in progress.

## Backend MVP

Start the FastAPI backend:

```bash
PYTHONPATH=apps/api .venv/bin/uvicorn app.main:app --reload --app-dir apps/api
```

Dashboard request:

```bash
curl -X POST "http://127.0.0.1:8000/api/news/dashboard" \
  -H "Content-Type: application/json" \
  -d '{"query":"Tesla","max_results":5}'
```

Gaming topic request:

```bash
curl -X POST "http://127.0.0.1:8000/api/news/dashboard" \
  -H "Content-Type: application/json" \
  -d '{"query":"Nintendo Switch 2","max_results":5}'
```

The response includes `data_source`, which is `mock`, `google_news_rss`, or
`fallback_mock` if Google News RSS is unavailable. It also includes
`detected_mode`, which the backend infers from the query and recent headlines.
