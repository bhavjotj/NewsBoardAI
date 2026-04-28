from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.news import router as news_router

app = FastAPI(title="NewsBoardAI API")

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(news_router, prefix="/api/news", tags=["news"])
