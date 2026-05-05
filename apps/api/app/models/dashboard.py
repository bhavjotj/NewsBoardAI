# Purpose: Defines the backend API contract: what the frontend sends, what the backend returns, and what values are allowed.
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator # uses pydantic for validation


# Allowed dashboard topic modes
class DashboardMode(str, Enum):
    GENERAL = "general"
    BUSINESS = "business"
    SPORTS = "sports"
    GAMING = "gaming"
    POLITICS = "politics"

# Tracks where the data comes from
class DataSource(str, Enum):
    MOCK = "mock"
    GOOGLE_NEWS_RSS = "google_news_rss"
    FALLBACK_MOCK = "fallback_mock"

# Tracks which model was used to generate the analysis
class AnalysisSource(str, Enum):
    RULE_BASED = "rule_based"
    HYBRID_ML = "hybrid_ml"
    HYBRID_ML_FALLBACK = "hybrid_ml_fallback"

# Tracks how the dashboard brief was written
class BriefSource(str, Enum):
    TEMPLATE = "template"
    OLLAMA = "ollama"
    OLLAMA_PARTIAL = "ollama_partial"
    OLLAMA_FALLBACK = "ollama_fallback"

# The request sent by the frontend
class DashboardRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=5)
    mode: Optional[DashboardMode] = None # Optional, for testing and debugging
    use_real_news: bool = True
    save_examples: bool = False
    use_ml: bool = True
    use_torch: bool = True
    use_llm_brief: bool = True
    ollama_model: str = "llama3.2"
    debug_analysis: bool = False

    # Validates the query, ensuring it is not empty
    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned

# The sentiment summary of the dashboard
class SentimentSummary(BaseModel):
    label: str
    score: float = Field(..., ge=-1.0, le=1.0)

# The source card of the dashboard
class SourceCard(BaseModel):
    title: str
    source: str
    published_at: Optional[datetime] = None
    snippet: str
    url: HttpUrl

# The response sent by the backend to the frontend
class DashboardResponse(BaseModel):
    topic: str
    data_source: DataSource
    analysis_source: AnalysisSource
    detected_mode: DashboardMode
    time_window: str
    overall_signal: str
    brief: str
    brief_source: BriefSource
    sentiment: SentimentSummary
    event_tags: list[str]
    sources: list[SourceCard]
    confidence: str
    possible_impact: str
    possible_impact_source: BriefSource
    llm_available: Optional[bool] = None
    torch_used: Optional[bool] = None
    torch_available: Optional[bool] = None
    analysis_debug: Optional[dict[str, Any]] = None
