# NewsBoardAI Agent Instructions

## Project Summary

NewsBoardAI is a lightweight personal AI news intelligence tool.

The user searches a topic, such as a company, stock, video game, sports team, athlete, political issue, product, or trend. The app fetches only a small number of recent news results, analyzes them, and displays a compact visual dashboard.

This is not meant to be a full news website. It should feel like a clean Chrome extension side panel or compact dashboard.

## Core Goal

Build a project that demonstrates practical AI/ML skills through:

- recent news retrieval
- text cleaning and preprocessing
- sentiment classification
- event type classification
- importance scoring
- compact visual dashboard output
- optional local ML models instead of paid APIs

## MVP Scope

The first working version should:

- Accept a user search query
- Fetch the top 3 to 5 recent news results
- Use a free or low-cost source, starting with Google News RSS or GDELT
- Extract title, source, date, snippet, and URL
- Analyze the small set of results
- Return a clean structured response for the frontend
- Display a compact dashboard with visual components

## Important Product Direction

The result should not be a long paragraph summary.

The result should be a dashboard-style brief with:

- topic name
- time window
- overall signal
- sentiment meter
- event tags
- top 2 to 5 source cards
- one short brief
- possible impact
- confidence level
- small timeline if useful

Keep results short, visual, and easy to scan.

## Preferred Tech Stack

Frontend:

- React
- TypeScript
- Tailwind CSS
- shadcn/ui where useful
- Recharts for small charts
- Chrome Extension Side Panel later

Backend:

- Python
- FastAPI
- Pydantic
- Uvicorn

ML / NLP:

- Hugging Face Transformers
- Sentence Transformers
- TensorFlow or PyTorch
- Start with lightweight local models where possible
- Prefer simple models and rule-based fallback before expensive LLM calls

Data:

- Google News RSS for MVP
- GDELT as a possible second source
- SQLite only if storage becomes necessary

## Cost Constraint

This project should be built to run at no cost or near-zero cost.

Avoid relying on paid APIs such as OpenAI, Anthropic, paid news APIs, or paid hosted databases unless they are optional.

If an API key is needed, hide it through environment variables and never commit secrets.

## Coding Style

Write clean, readable, modular code.

Prefer simple code over over-engineered abstractions.

Use clear file names and small functions.

Avoid creating large files that mix unrelated logic.

Add comments only when they explain non-obvious logic.

## Development Approach

Build incrementally.

Before coding a large feature, first inspect the existing project structure.

Make small focused changes.

Do not rewrite the whole project unless explicitly asked.

Do not introduce unnecessary dependencies.

If a dependency is added, explain why it is needed.

## Testing and Validation

For backend changes:

- Add simple testable functions where possible
- Use sample inputs and outputs
- Avoid requiring external paid APIs for tests

For frontend changes:

- Keep components reusable
- Handle loading, error, and empty states
- Keep UI compact and visually clean

## What Not To Build Yet

Do not build a full website with many pages.

Do not build user authentication yet.

Do not build a large database system yet.

Do not train a full summarization model from scratch.

Do not scrape full articles unless needed.

Do not make financial predictions look certain.

Use phrases like "possible impact", "signal", "confidence", and "unclear" instead of guaranteed predictions.

## Current Project Name

Use the name NewsBoardAI in code, documentation, and UI unless asked otherwise.