from __future__ import annotations

from collections import Counter

from app.models.dashboard import AnalysisSource, DashboardMode
from app.services.analyzer import DashboardAnalysis, analyze_articles
from app.services.baseline_predictor import BaselinePredictionResult, BaselinePredictor
from app.services.fetchers import NewsArticle

SENTIMENT_SCORE = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "mixed": 0.0,
}
CONFIDENCE_THRESHOLD = 0.55


def analyze_with_hybrid_ml(
    articles: list[NewsArticle],
    fallback_mode: DashboardMode,
    predictor: BaselinePredictor | None = None,
) -> tuple[DashboardAnalysis, DashboardMode, AnalysisSource]:
    predictor = predictor or BaselinePredictor()
    if not predictor.models:
        return (
            analyze_articles(articles, fallback_mode),
            fallback_mode,
            AnalysisSource.RULE_BASED,
        )

    try:
        predictions = [
            predictor.predict(title=article.title, snippet=article.snippet)
            for article in articles
        ]
    except Exception:
        return (
            analyze_articles(articles, fallback_mode),
            fallback_mode,
            AnalysisSource.HYBRID_ML_FALLBACK,
        )

    if not predictions or _all_confidence_too_low(predictions):
        return (
            analyze_articles(articles, fallback_mode),
            fallback_mode,
            AnalysisSource.HYBRID_ML_FALLBACK,
        )

    detected_mode = _detected_mode(predictions, fallback_mode)
    sentiment_label, sentiment_score = _sentiment(predictions)
    event_tags = _event_tags(predictions, detected_mode)
    confidence = _confidence(predictions, sentiment_label)
    analysis = DashboardAnalysis(
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        overall_signal=_overall_signal(sentiment_label),
        event_tags=event_tags,
        confidence=confidence,
        possible_impact=_possible_impact(sentiment_label, detected_mode, event_tags),
    )
    return analysis, detected_mode, AnalysisSource.HYBRID_ML


def _sentiment(predictions: list[BaselinePredictionResult]) -> tuple[str, float]:
    labels = [
        prediction.sentiment.label
        for prediction in predictions
        if prediction.sentiment.label
    ]
    if not labels:
        return "neutral", 0.0

    scores = [
        SENTIMENT_SCORE.get(prediction.sentiment.label or "neutral", 0.0)
        * (prediction.sentiment.confidence or 0.5)
        for prediction in predictions
        if prediction.sentiment.label
    ]
    average_score = round(sum(scores) / len(scores), 2)
    has_positive = any(label == "positive" for label in labels)
    has_negative = any(label == "negative" for label in labels)

    if has_positive and has_negative:
        return "mixed", average_score
    if average_score >= 0.25:
        return "positive", average_score
    if average_score <= -0.25:
        return "negative", average_score
    dominant_label = Counter(labels).most_common(1)[0][0]
    if dominant_label in {"positive", "negative"}:
        return dominant_label, average_score
    return "neutral", average_score


def _event_tags(
    predictions: list[BaselinePredictionResult],
    detected_mode: DashboardMode,
) -> list[str]:
    labels = [
        prediction.event_tag.label
        for prediction in predictions
        if prediction.event_tag.label and prediction.event_tag.label != "general"
    ]
    tags = [label for label, _ in Counter(labels).most_common(4)]
    mode_tag = detected_mode.value
    if detected_mode != DashboardMode.GENERAL and mode_tag not in tags:
        tags.insert(0, mode_tag)
    return tags[:4] or ["general"]


def _detected_mode(
    predictions: list[BaselinePredictionResult],
    fallback_mode: DashboardMode,
) -> DashboardMode:
    labels = [
        prediction.topic_mode.label
        for prediction in predictions
        if prediction.topic_mode.label in DashboardMode._value2member_map_
    ]
    if not labels:
        return fallback_mode
    return DashboardMode(Counter(labels).most_common(1)[0][0])


def _confidence(
    predictions: list[BaselinePredictionResult],
    sentiment_label: str,
) -> str:
    confidences = _usable_confidences(predictions)
    if not confidences:
        return "low"
    average = sum(confidences) / len(confidences)
    if average >= 0.72 and sentiment_label != "mixed":
        return "high"
    if average >= CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _all_confidence_too_low(predictions: list[BaselinePredictionResult]) -> bool:
    confidences = _usable_confidences(predictions)
    return not confidences or max(confidences) < CONFIDENCE_THRESHOLD


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


def _overall_signal(sentiment_label: str) -> str:
    if sentiment_label == "neutral":
        return "unclear"
    return sentiment_label


def _possible_impact(
    sentiment_label: str,
    detected_mode: DashboardMode,
    event_tags: list[str],
) -> str:
    context = {
        DashboardMode.BUSINESS: "market and business attention",
        DashboardMode.GAMING: "review, release, or player interest",
        DashboardMode.SPORTS: "team, player, or matchup attention",
        DashboardMode.POLITICS: "policy debate or public reaction",
        DashboardMode.GENERAL: "public attention or a general trend",
    }[detected_mode]
    tags = ", ".join(tag for tag in event_tags if tag != "general") or "the topic"

    if sentiment_label == "positive":
        return f"Possible positive signal for {context}, especially around {tags}."
    if sentiment_label == "negative":
        return f"Possible caution signal for {context}, especially around {tags}."
    if sentiment_label == "mixed":
        return f"Mixed signal for {context}; compare sources around {tags} before drawing conclusions."
    return f"Impact on {context} is unclear from the current source set."
