from datetime import datetime, timezone

from app.models.dashboard import AnalysisSource, DashboardMode
from app.services.baseline_predictor import (
    BaselinePredictionResult,
    LabelPrediction,
)
from app.services.fetchers import NewsArticle
from app.services.hybrid_analyzer import analyze_with_hybrid_ml
from app.services.torch_topic_service import TorchTopicLabel, TorchTopicPrediction


class FakePredictor:
    models = {"sentiment": object()}

    def __init__(self, predictions):
        self.predictions = predictions
        self.index = 0

    def predict(self, title: str, snippet: str = ""):
        prediction = self.predictions[self.index]
        self.index += 1
        return prediction


class FakeTorchService:
    def __init__(self, predictions, available=True):
        self.predictions = predictions
        self.index = 0
        self.available = available

    def predict(self, title: str, snippet: str = ""):
        prediction = self.predictions[self.index]
        self.index += 1
        return prediction


def article(title: str, snippet: str = "") -> NewsArticle:
    return NewsArticle(
        title=title,
        source="Example",
        published_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        snippet=snippet,
        url=f"https://example.com/{abs(hash(title))}",
    )


def prediction(
    sentiment="neutral",
    sentiment_conf=0.45,
    event="business",
    event_conf=0.25,
    topic="business",
    topic_conf=0.35,
) -> BaselinePredictionResult:
    return BaselinePredictionResult(
        input_text="",
        sentiment=LabelPrediction(sentiment, sentiment_conf),
        event_tag=LabelPrediction(event, event_conf),
        topic_mode=LabelPrediction(topic, topic_conf),
        raw_predictions={},
        notes=[],
        adjustments=[],
    )


def torch_prediction(label: str, confidence: float = 0.85) -> TorchTopicPrediction:
    return TorchTopicPrediction(
        label=label,
        confidence=confidence,
        top_labels=[TorchTopicLabel(label=label, confidence=confidence)],
        available=True,
    )


def test_nhl_sources_produce_sports_mode_without_business_noise() -> None:
    articles = [
        article("NHL playoffs referees draw attention", "Officiating and game picks dominate coverage."),
        article("NHL betting picks for playoff matchup", "Teams face a close series with odds shifting."),
        article("Stanley Cup playoff game preview", "The matchup has player injury questions."),
    ]
    predictions = [
        prediction(event="business", topic="business"),
        prediction(event="sports", topic="general", event_conf=0.48),
        prediction(event="business", topic="business"),
    ]

    result = analyze_with_hybrid_ml(
        query="NHL",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
        include_debug=True,
    )

    assert result.detected_mode == DashboardMode.SPORTS
    assert "sports" in result.analysis.event_tags
    assert "business" not in result.analysis.event_tags
    assert result.analysis.confidence in {"medium", "high"}
    assert result.debug


def test_layoff_workforce_sources_produce_business_risk_signal() -> None:
    articles = [
        article("Meta layoffs hit AI workers", "Workforce cuts raise business risk."),
        article("Company trims employees in AI unit", "Workers face layoffs as operations shift."),
        article("Meta workforce changes continue", "Analysts watch company jobs and risk."),
    ]
    predictions = [
        prediction(sentiment="neutral", event="sports", topic="sports"),
        prediction(sentiment="negative", sentiment_conf=0.52, event="product", topic="general"),
        prediction(sentiment="neutral", event="sports", topic="sports"),
    ]

    result = analyze_with_hybrid_ml(
        query="meta layoffs",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
    )

    assert result.detected_mode == DashboardMode.BUSINESS
    assert {"workforce", "risk"} & set(result.analysis.event_tags)
    assert result.analysis.sentiment_label in {"negative", "mixed"}


def test_product_launch_sources_drop_unrelated_sports_gaming_tags() -> None:
    articles = [
        article("McDonald's launches new drinks menu", "New product rollout focuses on food and pricing."),
        article("Restaurant adds summer drink lineup", "Menu launch expands beverage options."),
    ]
    predictions = [
        prediction(event="sports", topic="sports"),
        prediction(event="gaming", topic="gaming"),
    ]

    result = analyze_with_hybrid_ml(
        query="mcdonalds",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
    )

    assert {"product", "launch"} & set(result.analysis.event_tags)
    assert "sports" not in result.analysis.event_tags
    assert "gaming" not in result.analysis.event_tags


def test_low_confidence_noisy_tags_are_filtered_when_unsupported() -> None:
    articles = [
        article("Microsoft stock rises after earnings", "Revenue growth drew investor attention."),
        article("Shopify shares gain after analyst upgrade", "Investors watched market demand."),
    ]
    predictions = [
        prediction(sentiment="positive", event="gaming", topic="business", event_conf=0.20),
        prediction(sentiment="positive", event="sports", topic="business", event_conf=0.20),
    ]

    result = analyze_with_hybrid_ml(
        query="microsoft stock",
        articles=articles,
        fallback_mode=DashboardMode.BUSINESS,
        predictor=FakePredictor(predictions),
        include_debug=True,
    )

    assert result.detected_mode == DashboardMode.BUSINESS
    assert "gaming" not in result.analysis.event_tags
    assert "sports" not in result.analysis.event_tags
    assert result.debug
    assert set(result.debug["dropped_low_confidence_tags"]) == {"gaming", "sports"}


def test_repeated_source_agreement_raises_confidence_to_medium() -> None:
    articles = [
        article("Raptors playoff matchup preview", "Team and player coverage focuses on the game."),
        article("Raptors coach discusses playoff game", "Players prepare for the matchup."),
        article("NBA playoff picks include Raptors", "Betting odds and team form shape predictions."),
    ]
    predictions = [
        prediction(event="sports", event_conf=0.48, topic="sports", topic_conf=0.48),
        prediction(event="sports", event_conf=0.48, topic="sports", topic_conf=0.48),
        prediction(event="sports", event_conf=0.48, topic="sports", topic_conf=0.48),
    ]

    result = analyze_with_hybrid_ml(
        query="Toronto Raptors",
        articles=articles,
        fallback_mode=DashboardMode.SPORTS,
        predictor=FakePredictor(predictions),
    )

    assert result.detected_mode == DashboardMode.SPORTS
    assert result.analysis.confidence in {"medium", "high"}


def test_use_torch_false_skips_torch_mode_signal() -> None:
    articles = [
        article("Neutral civic update", "Local coverage has few domain signals."),
        article("General public notice", "Officials shared a short update."),
    ]
    predictions = [
        prediction(event="general", topic="general"),
        prediction(event="general", topic="general"),
    ]

    result = analyze_with_hybrid_ml(
        query="local update",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
        torch_service=FakeTorchService([torch_prediction("sports"), torch_prediction("sports")]),
        use_torch=False,
    )

    assert result.detected_mode == DashboardMode.GENERAL
    assert not result.torch_used


def test_torch_sports_prediction_strengthens_sports_mode() -> None:
    articles = [
        article("Hockey matchup preview", "Players prepare for a close game."),
        article("Playoff picks shift", "Team form and odds drive coverage."),
    ]
    predictions = [
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
    ]

    result = analyze_with_hybrid_ml(
        query="hockey",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
        torch_service=FakeTorchService([torch_prediction("sports"), torch_prediction("sports")]),
    )

    assert result.detected_mode == DashboardMode.SPORTS
    assert "sports" in result.analysis.event_tags
    assert result.torch_used


def test_torch_business_prediction_strengthens_business_mode() -> None:
    articles = [
        article("Company shares rise", "Investors watch revenue and earnings."),
        article("Analysts discuss market demand", "Stock coverage remains active."),
    ]
    predictions = [
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
    ]

    result = analyze_with_hybrid_ml(
        query="company stock",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
        torch_service=FakeTorchService([torch_prediction("business"), torch_prediction("business")]),
    )

    assert result.detected_mode == DashboardMode.BUSINESS
    assert "business" in result.analysis.event_tags


def test_torch_tech_prediction_supports_product_ai_tags() -> None:
    articles = [
        article("Apple AI tools arrive in iOS update", "Software features expand across devices."),
        article("AirPods update highlights Apple Intelligence", "Product coverage focuses on AI tools."),
    ]
    predictions = [
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
    ]

    result = analyze_with_hybrid_ml(
        query="Apple AI tools",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
        torch_service=FakeTorchService([torch_prediction("tech"), torch_prediction("tech")]),
    )

    assert {"ai", "product", "tech"} & set(result.analysis.event_tags)
    assert result.analysis.event_tags != ["general"]


def test_gaming_domain_evidence_overrides_torch_sports_prediction() -> None:
    articles = [
        article("Nintendo Switch 2 review roundup", "Console previews and game coverage grow."),
        article("GTA trailer details spark player interest", "Gaming fans watch release timing."),
    ]
    predictions = [
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
        prediction(event="general", topic="general", event_conf=0.2, topic_conf=0.2),
    ]

    result = analyze_with_hybrid_ml(
        query="Nintendo Switch 2",
        articles=articles,
        fallback_mode=DashboardMode.GENERAL,
        predictor=FakePredictor(predictions),
        torch_service=FakeTorchService([torch_prediction("sports"), torch_prediction("sports")]),
    )

    assert result.detected_mode == DashboardMode.GAMING
    assert "gaming" in result.analysis.event_tags
