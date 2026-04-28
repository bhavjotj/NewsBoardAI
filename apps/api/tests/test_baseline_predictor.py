from app.services.baseline_predictor import BaselinePredictor


class FakeModel:
    def __init__(self, label: str, confidence: float, classes: list[str]) -> None:
        self.label = label
        self.confidence = confidence
        self.classes_ = classes

    def predict(self, texts):
        return [self.label for _ in texts]

    def predict_proba(self, texts):
        low_value = (1.0 - self.confidence) / (len(self.classes_) - 1)
        probabilities = [low_value for _ in self.classes_]
        probabilities[self.classes_.index(self.label)] = self.confidence
        return [probabilities for _ in texts]


def predictor(sentiment=None, event=None, topic=None) -> BaselinePredictor:
    return BaselinePredictor(
        models={
            "sentiment": sentiment,
            "event": event,
            "topic": topic,
        }
    )


def test_finance_positive_wording_leans_positive_business() -> None:
    result = predictor(
        sentiment=FakeModel("negative", 0.35, ["negative", "neutral", "positive"]),
        topic=FakeModel("general", 0.40, ["business", "general", "sports"]),
    ).predict(
        title="Netflix shares rise after earnings",
        snippet="Revenue growth and analyst upgrades drew investor attention.",
    )

    assert result.sentiment.label == "positive"
    assert result.topic_mode.label == "business"
    assert result.adjustments


def test_finance_negative_wording_leans_negative_business() -> None:
    result = predictor(
        sentiment=FakeModel("positive", 0.40, ["negative", "neutral", "positive"]),
        topic=FakeModel("general", 0.45, ["business", "general", "sports"]),
    ).predict(
        title="Apple stock falls after weaker demand",
        snippet="Shares drop as analysts warn about risk and lower revenue.",
    )

    assert result.sentiment.label == "negative"
    assert result.topic_mode.label == "business"


def test_sports_wording_leans_sports() -> None:
    result = predictor(
        topic=FakeModel("general", 0.40, ["business", "general", "sports"])
    ).predict(
        title="Toronto Raptors player returns for playoff matchup",
        snippet="The coach said the team is ready for the next NBA game.",
    )

    assert result.topic_mode.label == "sports"


def test_gaming_wording_leans_gaming_without_ag_news_class() -> None:
    result = predictor(
        topic=FakeModel("general", 0.42, ["business", "general", "sports"]),
        event=FakeModel("sports", 0.45, ["gaming", "launch", "sports"]),
    ).predict(
        title="Nintendo Switch 2 review roundup",
        snippet="Console game previews highlight release timing and hardware changes.",
    )

    assert result.topic_mode.label == "gaming"
    assert result.event_tag.label == "gaming"


def test_low_confidence_prediction_includes_note() -> None:
    result = predictor(
        sentiment=FakeModel("neutral", 0.41, ["negative", "neutral", "positive"])
    ).predict(title="Company update", snippet="Details remain limited.")

    assert "sentiment model confidence is low." in result.notes


def test_high_confidence_prediction_not_overwritten_by_single_weak_term() -> None:
    result = predictor(
        topic=FakeModel("business", 0.92, ["business", "general", "sports"])
    ).predict(
        title="Company earnings update mentions game plan",
        snippet="Shares rose after revenue growth.",
    )

    assert result.topic_mode.label == "business"
    assert not any("gaming" in adjustment for adjustment in result.adjustments)
