from fastapi import FastAPI

from app.routes.news import router as news_router

app = FastAPI(title="NewsBoardAI API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(news_router, prefix="/api/news", tags=["news"])
