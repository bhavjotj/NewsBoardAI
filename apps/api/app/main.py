# Purpose: Main entry point for the API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.news import router as news_router

# Create the FastAPI app
app = FastAPI(title="NewsBoardAI API")

# Add CORS middleware, where we allow requests from the web app and extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint, where we check if the API is running
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

# Include the news router, where we handle the news dashboard requests
app.include_router(news_router, prefix="/api/news", tags=["news"])
