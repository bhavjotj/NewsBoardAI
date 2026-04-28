from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class DashboardMode(str, Enum):
    GENERAL = "general"
    BUSINESS = "business"
    SPORTS = "sports"
    GAMING = "gaming"
    POLITICS = "politics"


class DataSource(str, Enum):
    MOCK = "mock"
    GOOGLE_NEWS_RSS = "google_news_rss"
    FALLBACK_MOCK = "fallback_mock"


class DashboardRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=5)
    mode: Optional[DashboardMode] = None
    use_real_news: bool = True
    save_examples: bool = False

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class SentimentSummary(BaseModel):
    label: str
    score: float = Field(..., ge=-1.0, le=1.0)


class SourceCard(BaseModel):
    title: str
    source: str
    published_at: Optional[datetime] = None
    snippet: str
    url: HttpUrl


class DashboardResponse(BaseModel):
    topic: str
    data_source: DataSource
    detected_mode: DashboardMode
    time_window: str
    overall_signal: str
    brief: str
    sentiment: SentimentSummary
    event_tags: list[str]
    sources: list[SourceCard]
    confidence: str
    possible_impact: str
