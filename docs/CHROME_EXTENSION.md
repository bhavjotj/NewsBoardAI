# NewsBoardAI Chrome Extension

NewsBoardAI can run as a local Chrome Manifest V3 side panel. The extension uses the existing React app and calls the local FastAPI backend at `http://127.0.0.1:8000`.

This is local-only and does not require a Chrome Web Store login. The backend must be running on your machine before searches will work.

## Start the Backend

From the repository root:

```bash
uvicorn app.main:app --app-dir apps/api --reload
```

If you use a virtual environment, activate it first.

## Build the Extension

From the web app folder:

```bash
cd apps/web
npm run build:extension
```

This creates:

```text
apps/web/dist-extension
```

## Load in Chrome

1. Open `chrome://extensions`.
2. Turn on Developer Mode.
3. Click Load unpacked.
4. Select the `apps/web/dist-extension` folder.
5. Click the NewsBoardAI extension icon to open the side panel.

## Test Searches

With the backend running, try compact searches such as:

- `Tesla`
- `Netflix`
- `Nintendo Switch 2`
- `Toronto Raptors`
- `Bitcoin`

The extension sends `query`, `max_results`, `use_real_news: true`, `use_ml: true`, `use_torch: true`, and `use_llm_brief: true` to the local backend.

## Reload After Changes

After editing the frontend:

```bash
cd apps/web
npm run build:extension
```

Then go back to `chrome://extensions` and click the reload button on the NewsBoardAI extension card. Reopen the side panel to see the latest build.

## Regular Web App

The normal Vite workflow still works:

```bash
cd apps/web
npm run dev
```
