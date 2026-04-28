from dataclasses import dataclass
from re import escape, search, sub

from app.models.dashboard import DashboardMode
from app.services.fetchers import NewsArticle

POSITIVE_TERMS = {
    "beats expectations": 1.0,
    "better than expected": 0.9,
    "growth": 0.7,
    "strong": 0.6,
    "surge": 0.8,
    "rally": 0.7,
    "record": 0.6,
    "upgrade": 0.6,
    "profit": 0.5,
    "launch": 0.4,
    "partnership": 0.4,
    "win": 0.5,
}

NEGATIVE_TERMS = {
    "misses expectations": 1.0,
    "weaker than expected": 0.9,
    "lawsuit": 0.8,
    "investigation": 0.8,
    "delay": 0.7,
    "risk": 0.6,
    "warning": 0.6,
    "recall": 0.8,
    "slump": 0.7,
    "drop": 0.6,
    "falls": 0.6,
    "cuts": 0.5,
    "loss": 0.6,
    "weak": 0.5,
}

EVENT_KEYWORDS = {
    "earnings": {"earnings", "revenue", "profit", "quarterly results"},
    "product": {"product", "device", "hardware", "feature", "update"},
    "launch": {"launch", "launched", "release date", "preorder", "pre-order"},
    "review": {"review", "reviews", "hands-on", "hands on", "rated"},
    "pricing": {"price", "pricing", "cost", "discount", "subscription"},
    "legal": {"lawsuit", "court", "settlement", "sued"},
    "regulation": {"regulator", "regulation", "probe", "investigation", "ban"},
    "partnership": {"partnership", "partner", "deal", "alliance", "acquisition"},
    "leadership": {"ceo", "executive", "leadership", "resigns", "appointed"},
    "sports": {"team", "player", "coach", "league", "matchup", "tournament"},
    "gaming": {
        "game",
        "console",
        "studio",
        "patch",
        "trailer",
        "nintendo",
        "xbox",
        "playstation",
    },
    "politics": {"election", "policy", "bill", "vote", "government", "campaign"},
    "market": {"stock", "shares", "market", "investors", "analyst"},
    "risk": {"risk", "delay", "warning", "concern", "uncertain", "weak"},
}

MODE_KEYWORDS = {
    DashboardMode.BUSINESS: {
        "stock",
        "shares",
        "earnings",
        "revenue",
        "profit",
        "market",
        "investor",
        "analyst",
        "tesla",
        "apple",
        "nvidia",
    },
    DashboardMode.GAMING: {
        "nintendo",
        "switch",
        "playstation",
        "xbox",
        "console",
        "game",
        "gaming",
        "studio",
        "trailer",
    },
    DashboardMode.SPORTS: {
        "nba",
        "nfl",
        "nhl",
        "mlb",
        "team",
        "player",
        "coach",
        "match",
        "matchup",
        "playoffs",
        "score",
    },
    DashboardMode.POLITICS: {
        "election",
        "policy",
        "senate",
        "congress",
        "president",
        "minister",
        "vote",
        "bill",
        "government",
        "campaign",
    },
}

MODE_TAGS = {
    DashboardMode.BUSINESS: "market",
    DashboardMode.SPORTS: "sports",
    DashboardMode.GAMING: "gaming",
    DashboardMode.POLITICS: "politics",
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
    article_scores = [_article_sentiment_score(article) for article in articles]
    combined_text = " ".join(_article_text(article) for article in articles)
    sentiment_score = _sentiment_score(article_scores)
    sentiment_label = _sentiment_label(sentiment_score, article_scores)
    event_tags = _event_tags(combined_text, mode)
    confidence = _confidence(articles, article_scores, sentiment_label, event_tags)

    return DashboardAnalysis(
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        overall_signal=_overall_signal(sentiment_label, event_tags),
        event_tags=event_tags,
        confidence=confidence,
        possible_impact=_possible_impact(sentiment_label, mode),
    )


def detect_mode(query: str, articles: list[NewsArticle]) -> DashboardMode:
    text = " ".join([query.lower(), *[_article_text(article) for article in articles]])
    scores = {
        mode: _keyword_count(text, keywords)
        for mode, keywords in MODE_KEYWORDS.items()
    }
    best_mode, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return DashboardMode.GENERAL
    return best_mode


def _article_sentiment_score(article: NewsArticle) -> float:
    text = _article_text(article)
    positive_score = _weighted_keyword_score(text, POSITIVE_TERMS)
    negative_score = _weighted_keyword_score(text, NEGATIVE_TERMS)
    total_score = positive_score + negative_score
    if total_score == 0:
        return 0.0
    raw_score = (positive_score - negative_score) / total_score
    return round(max(-1.0, min(1.0, raw_score)), 2)


def _sentiment_score(article_scores: list[float]) -> float:
    if not article_scores:
        return 0.0
    return round(sum(article_scores) / len(article_scores), 2)


def _sentiment_label(score: float, article_scores: list[float]) -> str:
    positive_count = sum(1 for value in article_scores if value >= 0.25)
    negative_count = sum(1 for value in article_scores if value <= -0.25)
    if positive_count and negative_count:
        return "mixed"
    if score >= 0.3:
        return "positive"
    if score <= -0.3:
        return "negative"
    return "neutral"


def _event_tags(text: str, mode: DashboardMode) -> list[str]:
    tag_scores = [
        (tag, _keyword_count(text, keywords))
        for tag, keywords in EVENT_KEYWORDS.items()
    ]
    tags = [
        tag
        for tag, score in sorted(tag_scores, key=lambda item: item[1], reverse=True)
        if score > 0
    ]
    mode_tag = MODE_TAGS.get(mode)
    if mode_tag and mode_tag not in tags:
        tags.insert(0, mode_tag)
    return tags[:4] or ["general"]


def _confidence(
    articles: list[NewsArticle],
    article_scores: list[float],
    sentiment_label: str,
    event_tags: list[str],
) -> str:
    article_count = len(articles)
    duplicate_ratio = _duplicate_ratio(articles)
    weak_snippet_count = sum(1 for article in articles if len(article.snippet) < 45)
    clear_signal_count = sum(1 for score in article_scores if abs(score) >= 0.25)

    if article_count <= 1 or weak_snippet_count >= max(2, article_count):
        return "low"
    if duplicate_ratio >= 0.5:
        return "low"
    if sentiment_label == "mixed":
        return "medium" if article_count >= 3 else "low"
    if (
        article_count >= 4
        and clear_signal_count >= 3
        and event_tags != ["general"]
        and duplicate_ratio < 0.35
    ):
        return "high"
    if article_count >= 2:
        return "medium"
    return "low"


def _overall_signal(sentiment_label: str, event_tags: list[str]) -> str:
    if sentiment_label == "neutral" and event_tags == ["general"]:
        return "unclear"
    return sentiment_label


def _possible_impact(sentiment_label: str, mode: DashboardMode) -> str:
    contexts = {
        DashboardMode.BUSINESS: "market and business attention",
        DashboardMode.GAMING: (
            "consumer interest, game release timing, or review conversation"
        ),
        DashboardMode.SPORTS: "team, player, or matchup discussion",
        DashboardMode.POLITICS: "policy debate or public reaction",
        DashboardMode.GENERAL: "public attention",
    }
    context = contexts[mode]
    if sentiment_label == "positive":
        return (
            f"Possible positive signal for {context}, "
            "but the source set is still limited."
        )
    if sentiment_label == "negative":
        return f"Possible caution signal for {context}; details remain unclear from limited coverage."
    if sentiment_label == "mixed":
        return f"Mixed signal for {context}; compare source details before drawing conclusions."
    return f"Impact on {context} is unclear from the current small source set."


def _article_text(article: NewsArticle) -> str:
    return f"{article.title} {article.snippet}".lower()


def _weighted_keyword_score(text: str, weighted_terms: dict[str, float]) -> float:
    return round(
        sum(weight for term, weight in weighted_terms.items() if _has_term(text, term)),
        2,
    )


def _keyword_count(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if _has_term(text, term))


def _has_term(text: str, term: str) -> bool:
    return search(rf"\b{escape(term)}\b", text) is not None


def _duplicate_ratio(articles: list[NewsArticle]) -> float:
    if len(articles) < 2:
        return 0.0

    fingerprints = [_fingerprint(article.title) for article in articles]
    duplicate_count = len(fingerprints) - len(set(fingerprints))
    return duplicate_count / len(fingerprints)


def _fingerprint(value: str) -> str:
    words = sub(r"[^a-z0-9\s]", " ", value.lower()).split()
    useful_words = [
        word
        for word in words
        if len(word) > 3 and word not in {"with", "from", "that", "this", "after"}
    ]
    return " ".join(useful_words[:7])
