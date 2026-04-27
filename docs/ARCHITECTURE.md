# NewsBoardAI Architecture

## High-Level Flow

User search query
→ News fetcher
→ News cleaner
→ Duplicate grouping
→ Sentiment classifier
→ Event type classifier
→ Importance scorer
→ Dashboard response formatter
→ Chrome extension UI

## Main Components

### Chrome Extension

The extension will provide a compact side panel with:

- Search bar
- Loading state
- Visual dashboard cards
- Source links
- Saved topics later

### FastAPI Backend

The backend will handle:

- Search requests
- News fetching
- Text cleaning
- Model inference
- Response formatting

### ML Package

The ML package will include:

- Sentiment classification
- Event type classification
- Importance scoring
- Optional impact prediction

### Data Layer

The first version can avoid a database.

Later versions may use SQLite for:

- Saved searches
- Cached results
- Model training examples
- User-labeled examples
