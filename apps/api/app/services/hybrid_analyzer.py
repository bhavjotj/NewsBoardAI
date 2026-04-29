from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from app.models.dashboard import AnalysisSource, DashboardMode
from app.services.analyzer import DashboardAnalysis, analyze_articles
from app.services.baseline_predictor import (
    BaselinePredictionResult,
    BaselinePredictor,
    DOMAIN_LEXICONS,
    normalized_tokens,
)
from app.services.fetchers import NewsArticle

SENTIMENT_SCORE = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "mixed": 0.0,
}
CONFIDENCE_THRESHOLD = 0.55

MODE_LEXICONS = {
    DashboardMode.BUSINESS: {
        *DOMAIN_LEXICONS["finance_market_terms"],
        "ai",
        "business",
        "shopify",
        "microsoft",
        "meta",
    },
    DashboardMode.SPORTS: DOMAIN_LEXICONS["sports_terms"],
    DashboardMode.GAMING: DOMAIN_LEXICONS["gaming_terms"],
    DashboardMode.POLITICS: DOMAIN_LEXICONS["politics_policy_terms"],
}

TAG_LEXICONS = {
    "sports": DOMAIN_LEXICONS["sports_terms"],
    "playoffs": {"playoff", "playoffs", "matchup", "series"},
    "betting": {"betting", "picks", "odds", "prediction", "predictions"},
    "business": DOMAIN_LEXICONS["finance_market_terms"],
    "workforce": {"layoffs", "layoff", "workforce", "workers", "employees", "jobs"},
    "risk": DOMAIN_LEXICONS["negative_risk_terms"],
    "product": {"product", "menu", "drink", "drinks", "food", "store", "device"},
    "launch": DOMAIN_LEXICONS["product_launch_terms"],
    "health": {"health", "healthy", "illness", "nutrition", "food"},
    "gaming": DOMAIN_LEXICONS["gaming_terms"],
    "politics": DOMAIN_LEXICONS["politics_policy_terms"],
}

MODE_TAGS = {
    DashboardMode.BUSINESS: "business",
    DashboardMode.SPORTS: "sports",
    DashboardMode.GAMING: "gaming",
    DashboardMode.POLITICS: "politics",
}


@dataclass(frozen=True)
class HybridAnalysisResult:
    analysis: DashboardAnalysis
    detected_mode: DashboardMode
    analysis_source: AnalysisSource
    debug: dict | None = None


def analyze_with_hybrid_ml(
    query: str,
    articles: list[NewsArticle],
    fallback_mode: DashboardMode,
    predictor: BaselinePredictor | None = None,
    include_debug: bool = False,
) -> HybridAnalysisResult:
    predictor = predictor or BaselinePredictor()
    if not predictor.models:
        return _fallback(articles, fallback_mode, AnalysisSource.RULE_BASED)

    try:
        predictions = [
            predictor.predict(title=article.title, snippet=article.snippet)
            for article in articles
        ]
    except Exception as error:
        result = _fallback(articles, fallback_mode, AnalysisSource.HYBRID_ML_FALLBACK)
        if include_debug:
            return HybridAnalysisResult(
                result.analysis,
                result.detected_mode,
                result.analysis_source,
                {"aggregation_notes": [f"Hybrid prediction failed: {error}"]},
            )
        return result

    if not predictions:
        return _fallback(articles, fallback_mode, AnalysisSource.HYBRID_ML_FALLBACK)

    context = _context(query, articles)
    mode_scores = _mode_scores(query, context, predictions, fallback_mode)
    detected_mode = max(mode_scores.items(), key=lambda item: item[1])[0]
    sentiment_label, sentiment_score, sentiment_notes = _sentiment(context, predictions)
    event_tags, dropped_tags, tag_scores = _event_tags(
        query, context, predictions, detected_mode
    )
    confidence = _confidence(predictions, detected_mode, event_tags, sentiment_label)
    analysis = DashboardAnalysis(
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        overall_signal=_overall_signal(sentiment_label),
        event_tags=event_tags,
        confidence=confidence,
        possible_impact=_possible_impact(sentiment_label, detected_mode, event_tags),
        brief=_brief(query, sentiment_label, detected_mode, event_tags),
    )

    if confidence == "low" and not _strong_domain_agreement(mode_scores):
        result = _fallback(articles, fallback_mode, AnalysisSource.HYBRID_ML_FALLBACK)
        if include_debug:
            return HybridAnalysisResult(
                result.analysis,
                result.detected_mode,
                result.analysis_source,
                _debug(predictions, dropped_tags, tag_scores, mode_scores, ["Low hybrid confidence; used rule fallback."]),
            )
        return result

    debug = None
    if include_debug:
        debug = _debug(predictions, dropped_tags, tag_scores, mode_scores, sentiment_notes)
    return HybridAnalysisResult(
        analysis=analysis,
        detected_mode=detected_mode,
        analysis_source=AnalysisSource.HYBRID_ML,
        debug=debug,
    )


def _fallback(
    articles: list[NewsArticle],
    fallback_mode: DashboardMode,
    analysis_source: AnalysisSource,
) -> HybridAnalysisResult:
    return HybridAnalysisResult(
        analysis=analyze_articles(articles, fallback_mode),
        detected_mode=fallback_mode,
        analysis_source=analysis_source,
        debug=None,
    )


def _mode_scores(
    query: str,
    context: str,
    predictions: list[BaselinePredictionResult],
    fallback_mode: DashboardMode,
) -> dict[DashboardMode, float]:
    query_tokens = normalized_tokens(query)
    context_tokens = normalized_tokens(context)
    scores = {mode: 0.0 for mode in DashboardMode}
    scores[fallback_mode] += 0.5

    for mode, terms in MODE_LEXICONS.items():
        scores[mode] += 3.0 * len(query_tokens & terms)
        scores[mode] += 1.0 * len(context_tokens & terms)

    for prediction in predictions:
        label = prediction.topic_mode.label
        if label in DashboardMode._value2member_map_:
            mode = DashboardMode(label)
            confidence = prediction.topic_mode.confidence or 0.4
            support = len(context_tokens & MODE_LEXICONS.get(mode, set()))
            if confidence < 0.45 and support == 0:
                continue
            scores[mode] += 1.5 * confidence
    return scores


def _sentiment(
    context: str,
    predictions: list[BaselinePredictionResult],
) -> tuple[str, float, list[str]]:
    positive = 0.0
    negative = 0.0
    labels = []
    notes = []

    for prediction in predictions:
        label = prediction.sentiment.label or "neutral"
        confidence = prediction.sentiment.confidence or 0.5
        labels.append(label)
        if label == "positive":
            positive += confidence
        elif label == "negative":
            negative += confidence
        elif label == "mixed":
            positive += 0.25 * confidence
            negative += 0.25 * confidence

    tokens = normalized_tokens(context)
    positive += 0.25 * len(tokens & DOMAIN_LEXICONS["positive_growth_terms"])
    negative += 0.25 * len(tokens & DOMAIN_LEXICONS["negative_risk_terms"])

    score = round((positive - negative) / max(positive + negative, 1.0), 2)
    has_positive = positive >= 0.8
    has_negative = negative >= 0.8
    if has_positive and has_negative and abs(positive - negative) < 0.9:
        return "mixed", score, ["Positive and negative signals both appear."]
    if score >= 0.2:
        return "positive", score, notes
    if score <= -0.2:
        return "negative", score, notes
    if "positive" in labels and "negative" in labels:
        return "mixed", score, ["Source sentiment conflicts."]
    return "neutral", score, notes


def _event_tags(
    query: str,
    context: str,
    predictions: list[BaselinePredictionResult],
    detected_mode: DashboardMode,
) -> tuple[list[str], list[str], dict[str, float]]:
    query_tokens = normalized_tokens(query)
    context_tokens = normalized_tokens(context)
    scores = defaultdict(float)
    dropped_tags = []

    for tag, terms in TAG_LEXICONS.items():
        scores[tag] += 2.0 * len(query_tokens & terms)
        scores[tag] += 0.8 * len(context_tokens & terms)

    for prediction in predictions:
        tag = prediction.event_tag.label
        if not tag or tag == "general":
            continue
        support = scores.get(tag, 0.0)
        confidence = prediction.event_tag.confidence or 0.4
        if confidence < 0.45 and support == 0 and tag != MODE_TAGS.get(detected_mode):
            dropped_tags.append(tag)
            continue
        scores[tag] += confidence

    mode_tag = MODE_TAGS.get(detected_mode)
    if mode_tag:
        scores[mode_tag] += 1.5

    sorted_tags = [
        tag
        for tag, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= 1.0 and _tag_allowed_for_mode(tag, detected_mode, scores)
    ]
    return sorted_tags[:4] or ["general"], dropped_tags, dict(scores)


def _tag_allowed_for_mode(
    tag: str,
    detected_mode: DashboardMode,
    scores: dict[str, float],
) -> bool:
    if tag in {"sports", "gaming", "business", "politics"}:
        return tag == detected_mode.value or scores[tag] >= 2.5
    if detected_mode == DashboardMode.SPORTS and tag in {"business", "product"}:
        return scores[tag] >= 2.5
    return True


def _confidence(
    predictions: list[BaselinePredictionResult],
    detected_mode: DashboardMode,
    event_tags: list[str],
    sentiment_label: str,
) -> str:
    source_count = len(predictions)
    topic_labels = [
        prediction.topic_mode.label
        for prediction in predictions
        if prediction.topic_mode.label
    ]
    sentiment_labels = [
        prediction.sentiment.label
        for prediction in predictions
        if prediction.sentiment.label
    ]
    confidences = _usable_confidences(predictions)
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    topic_agreement = _agreement(topic_labels, detected_mode.value)
    sentiment_agreement = _agreement(sentiment_labels, sentiment_label)

    score = 0
    if source_count >= 2:
        score += 1
    if source_count >= 4:
        score += 1
    if topic_agreement >= 0.5 or detected_mode != DashboardMode.GENERAL:
        score += 1
    if event_tags != ["general"]:
        score += 1
    if sentiment_label == "mixed" or sentiment_agreement >= 0.5:
        score += 1
    if average_confidence >= CONFIDENCE_THRESHOLD:
        score += 1

    if score >= 5 and average_confidence >= 0.62:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def _agreement(labels: list[str | None], target: str) -> float:
    usable = [label for label in labels if label]
    if not usable:
        return 0.0
    return usable.count(target) / len(usable)


def _usable_confidences(predictions: list[BaselinePredictionResult]) -> list[float]:
    confidences = []
    for prediction in predictions:
        for label_prediction in (
            prediction.sentiment,
            prediction.event_tag,
            prediction.topic_mode,
        ):
            if label_prediction.confidence is not None:
                confidences.append(label_prediction.confidence)
    return confidences


def _strong_domain_agreement(mode_scores: dict[DashboardMode, float]) -> bool:
    top_score = max(mode_scores.values())
    return top_score >= 3.0


def _overall_signal(sentiment_label: str) -> str:
    if sentiment_label == "neutral":
        return "unclear"
    return sentiment_label


def _brief(
    query: str,
    sentiment_label: str,
    detected_mode: DashboardMode,
    event_tags: list[str],
) -> str:
    theme = _tag_phrase(event_tags)
    if sentiment_label == "neutral":
        return f"{query} coverage is focused on {theme}, with no clear positive or negative sentiment."
    if sentiment_label == "negative":
        return f"{query} coverage is centered on {theme}, creating a cautious {detected_mode.value} signal."
    if sentiment_label == "positive":
        return f"{query} coverage is leaning positive around {theme}, but the signal is still based on recent headlines."
    return f"{query} coverage is mixed, with competing signals around {theme}."


def _possible_impact(
    sentiment_label: str,
    detected_mode: DashboardMode,
    event_tags: list[str],
) -> str:
    context = {
        DashboardMode.BUSINESS: "market, operations, workforce, earnings, or product attention",
        DashboardMode.GAMING: "release, review, or player interest",
        DashboardMode.SPORTS: "team, player, matchup, or playoff attention",
        DashboardMode.POLITICS: "policy debate or public reaction",
        DashboardMode.GENERAL: "public attention or a general trend",
    }[detected_mode]
    theme = _tag_phrase(event_tags)

    if sentiment_label == "positive":
        return f"Possible positive signal for {context}, especially around {theme}."
    if sentiment_label == "negative":
        return f"Possible caution signal for {context}, especially around {theme}."
    if sentiment_label == "mixed":
        return f"Mixed signal for {context}; compare source details around {theme}."
    return f"Impact on {context} is unclear from the current source set."


def _tag_phrase(tags: list[str]) -> str:
    useful_tags = [tag for tag in tags if tag != "general"][:3]
    if not useful_tags:
        return "general coverage"
    if len(useful_tags) == 1:
        return useful_tags[0]
    return ", ".join(useful_tags[:-1]) + f", and {useful_tags[-1]}"


def _context(query: str, articles: list[NewsArticle]) -> str:
    article_text = " ".join(f"{article.title} {article.snippet}" for article in articles)
    return f"{query} {article_text}".lower()


def _debug(
    predictions: list[BaselinePredictionResult],
    dropped_tags: list[str],
    tag_scores: dict[str, float],
    mode_scores: dict[DashboardMode, float],
    aggregation_notes: list[str],
) -> dict:
    return {
        "articles": [
            {
                "input_text": prediction.input_text,
                "sentiment": prediction.sentiment.__dict__,
                "event_tag": prediction.event_tag.__dict__,
                "topic_mode": prediction.topic_mode.__dict__,
                "notes": prediction.notes,
                "adjustments": prediction.adjustments,
            }
            for prediction in predictions
        ],
        "dropped_low_confidence_tags": sorted(set(dropped_tags)),
        "tag_scores": {key: round(value, 2) for key, value in tag_scores.items()},
        "mode_scores": {
            mode.value: round(score, 2) for mode, score in mode_scores.items()
        },
        "aggregation_notes": aggregation_notes,
    }
