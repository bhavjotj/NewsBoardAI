from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib

from app.services.ml_preprocessing import build_model_text

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
LOW_CONFIDENCE = 0.55
HIGH_CONFIDENCE = 0.80

DOMAIN_LEXICONS = {
    "finance_market_terms": {
        "stock",
        "stocks",
        "shares",
        "earnings",
        "revenue",
        "profit",
        "market",
        "investor",
        "investors",
        "analyst",
        "demand",
        "company",
        "companies",
        "layoffs",
        "layoff",
        "workforce",
        "workers",
        "employees",
        "operations",
    },
    "negative_risk_terms": {
        "falls",
        "fall",
        "fell",
        "weaker",
        "weak",
        "decline",
        "drops",
        "drop",
        "loss",
        "lawsuit",
        "risk",
        "warning",
        "cuts",
        "misses",
        "down",
        "layoffs",
        "layoff",
        "fire",
        "closure",
        "closed",
        "cuts",
    },
    "positive_growth_terms": {
        "rise",
        "rises",
        "rose",
        "growth",
        "gain",
        "gains",
        "beats",
        "strong",
        "higher",
        "record",
        "upgrade",
    },
    "sports_terms": {
        "team",
        "player",
        "coach",
        "league",
        "match",
        "matchup",
        "season",
        "playoff",
        "playoffs",
        "score",
        "tournament",
        "raptors",
        "nba",
        "nfl",
        "nhl",
        "mlb",
        "fifa",
        "referee",
        "referees",
        "officiating",
        "betting",
        "picks",
        "odds",
    },
    "gaming_terms": {
        "nintendo",
        "switch",
        "playstation",
        "xbox",
        "console",
        "game",
        "gaming",
        "review",
        "trailer",
        "release",
        "studio",
        "patch",
        "ps5",
    },
    "politics_policy_terms": {
        "election",
        "policy",
        "bill",
        "vote",
        "government",
        "senate",
        "congress",
        "campaign",
        "regulation",
    },
    "product_launch_terms": {
        "launch",
        "launched",
        "release",
        "product",
        "device",
        "pricing",
        "preorder",
        "pre-order",
        "menu",
        "drink",
        "drinks",
    },
}


@dataclass(frozen=True)
class LabelPrediction:
    label: str | None
    confidence: float | None = None


@dataclass(frozen=True)
class BaselinePredictionResult:
    input_text: str
    sentiment: LabelPrediction
    event_tag: LabelPrediction
    topic_mode: LabelPrediction
    raw_predictions: dict[str, LabelPrediction]
    notes: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)


class BaselinePredictor:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        models: dict[str, object] | None = None,
    ) -> None:
        self.models = models if models is not None else load_models(model_dir)

    def predict(self, title: str, snippet: str = "") -> BaselinePredictionResult:
        input_text = build_model_text({"title": title, "snippet": snippet})
        raw_predictions = {
            "sentiment": predict_with_confidence(
                self.models.get("sentiment"), input_text
            ),
            "event_tag": predict_with_confidence(self.models.get("event"), input_text),
            "topic_mode": predict_with_confidence(self.models.get("topic"), input_text),
        }

        sentiment = raw_predictions["sentiment"]
        event_tag = raw_predictions["event_tag"]
        topic_mode = raw_predictions["topic_mode"]
        notes = low_confidence_notes(raw_predictions)
        adjustments = []

        sentiment, sentiment_adjustment = adjust_sentiment(input_text, sentiment)
        topic_mode, topic_adjustment = adjust_topic_mode(input_text, topic_mode)
        event_tag, event_adjustment = adjust_event_tag(input_text, event_tag)

        for adjustment in (sentiment_adjustment, topic_adjustment, event_adjustment):
            if adjustment:
                adjustments.append(adjustment)

        return BaselinePredictionResult(
            input_text=input_text,
            sentiment=sentiment,
            event_tag=event_tag,
            topic_mode=topic_mode,
            raw_predictions=raw_predictions,
            notes=notes,
            adjustments=adjustments,
        )


def load_models(model_dir: Path) -> dict[str, object]:
    model_files = {
        "sentiment": model_dir / "sentiment_model.joblib",
        "event": model_dir / "event_model.joblib",
        "topic": model_dir / "topic_model.joblib",
    }
    models = {}
    for name, path in model_files.items():
        if path.exists():
            artifact = joblib.load(path)
            models[name] = artifact.get("model", artifact)
    return models


def predict_with_confidence(model: object | None, input_text: str) -> LabelPrediction:
    if model is None:
        return LabelPrediction(label=None, confidence=None)

    label = str(model.predict([input_text])[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([input_text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(label)])
        return LabelPrediction(label=label, confidence=round(confidence, 3))
    return LabelPrediction(label=label, confidence=None)


def adjust_sentiment(
    input_text: str, prediction: LabelPrediction
) -> tuple[LabelPrediction, str | None]:
    positive_score = concept_score(input_text, "positive_growth_terms")
    negative_score = concept_score(input_text, "negative_risk_terms")
    finance_score = concept_score(input_text, "finance_market_terms")

    if negative_score >= 3 and not is_high_confidence(prediction):
        return LabelPrediction("negative", confidence=rule_confidence(prediction)), (
            "Adjusted sentiment to negative from strong risk-term agreement."
        )
    if positive_score >= 3 and not is_high_confidence(prediction):
        return LabelPrediction("positive", confidence=rule_confidence(prediction)), (
            "Adjusted sentiment to positive from strong growth-term agreement."
        )
    if negative_score >= 2 and should_adjust(prediction):
        return LabelPrediction("negative", confidence=rule_confidence(prediction)), (
            "Adjusted sentiment to negative from multiple risk terms."
        )
    if positive_score >= 2 and should_adjust(prediction):
        return LabelPrediction("positive", confidence=rule_confidence(prediction)), (
            "Adjusted sentiment to positive from multiple growth terms."
        )
    if (
        finance_score >= 2
        and positive_score > negative_score
        and is_low_confidence(prediction)
    ):
        return LabelPrediction("positive", confidence=rule_confidence(prediction)), (
            "Adjusted sentiment toward positive finance signal."
        )
    return prediction, None


def adjust_topic_mode(
    input_text: str, prediction: LabelPrediction
) -> tuple[LabelPrediction, str | None]:
    topic_scores = {
        "gaming": concept_score(input_text, "gaming_terms"),
        "sports": concept_score(input_text, "sports_terms"),
        "business": concept_score(input_text, "finance_market_terms"),
        "politics": concept_score(input_text, "politics_policy_terms"),
    }
    best_label, best_score = max(topic_scores.items(), key=lambda item: item[1])
    if best_score >= 2 and should_adjust_topic(prediction, best_label):
        return LabelPrediction(best_label, confidence=rule_confidence(prediction)), (
            f"Adjusted topic mode to {best_label} from domain terms."
        )
    return prediction, None


def adjust_event_tag(
    input_text: str, prediction: LabelPrediction
) -> tuple[LabelPrediction, str | None]:
    if concept_score(input_text, "gaming_terms") >= 2 and should_adjust(prediction):
        return LabelPrediction("gaming", confidence=rule_confidence(prediction)), (
            "Adjusted event tag to gaming from gaming terms."
        )
    if concept_score(input_text, "product_launch_terms") >= 2 and should_adjust(
        prediction
    ):
        return LabelPrediction("launch", confidence=rule_confidence(prediction)), (
            "Adjusted event tag to launch from product launch terms."
        )
    return prediction, None


def should_adjust(prediction: LabelPrediction) -> bool:
    return prediction.label is None or is_low_confidence(prediction)


def should_adjust_topic(prediction: LabelPrediction, new_label: str) -> bool:
    if prediction.label == new_label:
        return False
    if prediction.label in {None, "general"}:
        return True
    return is_low_confidence(prediction)


def is_low_confidence(prediction: LabelPrediction) -> bool:
    return prediction.confidence is None or prediction.confidence < LOW_CONFIDENCE


def is_high_confidence(prediction: LabelPrediction) -> bool:
    return prediction.confidence is not None and prediction.confidence >= HIGH_CONFIDENCE


def rule_confidence(prediction: LabelPrediction) -> float | None:
    if prediction.confidence is None:
        return 0.6
    return round(max(prediction.confidence, LOW_CONFIDENCE), 3)


def low_confidence_notes(
    predictions: dict[str, LabelPrediction],
) -> list[str]:
    notes = []
    for name, prediction in predictions.items():
        if prediction.label is None:
            notes.append(f"No {name} model available.")
        elif prediction.confidence is None:
            notes.append(f"{name} model does not expose probabilities.")
        elif prediction.confidence < LOW_CONFIDENCE:
            notes.append(f"{name} model confidence is low.")
    return notes


def concept_score(input_text: str, concept: str) -> int:
    tokens = normalized_tokens(input_text)
    return sum(1 for term in DOMAIN_LEXICONS[concept] if term in tokens)


def normalized_tokens(input_text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}'\"").lower()
        for token in input_text.replace("-", " ").split()
    }
