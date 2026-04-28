from dataclasses import dataclass

from app.models.dashboard import DashboardMode
from app.services.fetchers import NewsArticle

POSITIVE_TERMS = {
    "gain",
    "gains",
    "growth",
    "launch",
    "launched",
    "record",
    "surge",
    "strong",
    "upgrade",
    "win",
    "wins",
}

NEGATIVE_TERMS = {
    "ban",
    "cuts",
    "decline",
    "delay",
    "down",
    "drop",
    "falls",
    "lawsuit",
    "loss",
    "risk",
    "slump",
    "weak",
}

EVENT_KEYWORDS = {
    "business": {"earnings", "revenue", "stock", "shares", "deal", "acquisition"},
    "sports": {"game", "match", "season", "coach", "player", "win", "loss"},
    "gaming": {"game", "console", "studio", "release", "patch", "trailer"},
    "politics": {"election", "policy", "bill", "vote", "court", "government"},
    "legal": {"lawsuit", "court", "regulator", "probe", "investigation"},
    "product": {"launch", "release", "update", "feature", "device"},
}


@dataclass(frozen=True)
class DashboardAnalysis:
    sentiment_label: str
    sentiment_score: float
    overall_signal: str
    event_tags: list[str]
    confidence: str
    possible_impact: str


def analyze_articles(
    articles: list[NewsArticle], mode: DashboardMode
) -> DashboardAnalysis:
    combined_text = " ".join(
        f"{article.title} {article.snippet}" for article in articles
    ).lower()
    sentiment_score = _sentiment_score(combined_text)
    sentiment_label = _sentiment_label(sentiment_score)
    event_tags = _event_tags(combined_text, mode)
    confidence = _confidence(len(articles), sentiment_label, event_tags)

    return DashboardAnalysis(
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        overall_signal=_overall_signal(sentiment_label, event_tags),
        event_tags=event_tags,
        confidence=confidence,
        possible_impact=_possible_impact(sentiment_label, mode),
    )


def _sentiment_score(text: str) -> float:
    positive_hits = sum(1 for term in POSITIVE_TERMS if term in text)
    negative_hits = sum(1 for term in NEGATIVE_TERMS if term in text)
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return 0.0
    return round((positive_hits - negative_hits) / total_hits, 2)


def _sentiment_label(score: float) -> str:
    if score >= 0.3:
        return "positive"
    if score <= -0.3:
        return "negative"
    if score != 0:
        return "mixed"
    return "neutral"


def _event_tags(text: str, mode: DashboardMode) -> list[str]:
    tags = [
        tag
        for tag, keywords in EVENT_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    if mode.value != "general" and mode.value not in tags:
        tags.insert(0, mode.value)
    return tags[:4] or ["general"]


def _confidence(article_count: int, sentiment_label: str, event_tags: list[str]) -> str:
    if article_count >= 4 and sentiment_label != "mixed" and event_tags != ["general"]:
        return "high"
    if article_count >= 2:
        return "medium"
    return "low"


def _overall_signal(sentiment_label: str, event_tags: list[str]) -> str:
    if sentiment_label == "neutral" and event_tags == ["general"]:
        return "unclear"
    return sentiment_label


def _possible_impact(sentiment_label: str, mode: DashboardMode) -> str:
    if sentiment_label == "positive":
        return f"Possible positive signal for near-term {mode.value} attention."
    if sentiment_label == "negative":
        return f"Possible caution signal; details remain unclear from limited {mode.value} coverage."
    if sentiment_label == "mixed":
        return "Mixed signal; source details should be compared before drawing conclusions."
    return "Impact is unclear from the current small source set."
