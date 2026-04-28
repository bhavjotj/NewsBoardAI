import json
from pathlib import Path

RAW_PATH = Path("data/raw/news_examples.jsonl")
LABELED_PATH = Path("data/labeled/news_labeled.jsonl")

LABELS_BY_TITLE_START = {
    "Tesla Rival BYD's Earnings Dive": ("mixed", ["earnings", "market", "risk"], "medium", "mixed"),
    "Tesla Officially Registers Elon Musk": ("neutral", ["market", "leadership"], "medium", "unclear"),
    "Drunk Driver Used Tesla Autopilot": ("negative", ["legal", "risk"], "medium", "negative"),

    "‘Running Point’ & ‘Beef’ Return": ("negative", ["entertainment", "review"], "low", "negative"),
    "Dave Chappelle Re-Teams": ("positive", ["entertainment", "partnership"], "medium", "positive"),
    "Dave Chappelle Sets Three Shows": ("positive", ["entertainment", "event"], "low", "positive"),

    "Elon Musk takes the stand": ("negative", ["legal", "leadership", "risk"], "high", "negative"),
    "OpenAI Misses Key Revenue": ("negative", ["earnings", "market", "risk"], "high", "negative"),
    "Oracle, AMD, and CoreWeave stocks sink": ("negative", ["market", "risk", "partnership"], "high", "negative"),

    "All The Evidence That GTA 6": ("positive", ["gaming", "launch", "marketing"], "medium", "positive"),
    "Rockstar Want PS5": ("positive", ["gaming", "product", "development"], "medium", "positive"),
    "GTA 6 Trailer 3": ("positive", ["gaming", "launch", "marketing"], "medium", "positive"),

    "Cleveland Cavaliers vs Toronto Raptors": ("neutral", ["sports", "game"], "medium", "unclear"),
    "'Ultimate competitor'": ("positive", ["sports", "player"], "medium", "positive"),
    "Raptors’ Collin Murray-Boyles": ("positive", ["sports", "player"], "medium", "positive"),

    "Apple Readies Photo-Editing": ("positive", ["product", "launch", "ai"], "high", "positive"),
    "iOS 27 could make your AirPods": ("positive", ["product", "launch"], "medium", "positive"),
    "Apple is gearing up": ("positive", ["product", "launch", "ai"], "high", "positive"),

    "PS5 Digital Games": ("negative", ["gaming", "policy", "risk"], "high", "negative"),
    "Path-Traced F1 25": ("positive", ["gaming", "product", "technology"], "medium", "positive"),
    "Wanted in Clarksville": ("negative", ["crime", "risk"], "low", "negative"),

    "Fifa approves red cards": ("negative", ["sports", "regulation", "policy"], "high", "negative"),
    "Iran absent as FIFA": ("neutral", ["sports", "politics", "event"], "medium", "unclear"),
    "adidas Expands Pet Collection": ("positive", ["sports", "product", "marketing"], "low", "positive"),
}

LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)

count = 0
with RAW_PATH.open("r", encoding="utf-8") as raw_file, LABELED_PATH.open("a", encoding="utf-8") as labeled_file:
    for line in raw_file:
        row = json.loads(line)
        title = row.get("title", "")

        for title_start, labels in LABELS_BY_TITLE_START.items():
            if title.startswith(title_start):
                sentiment, tags, importance, impact = labels
                row["sentiment_label"] = sentiment
                row["event_tags_label"] = tags
                row["importance_label"] = importance
                row["impact_label"] = impact
                labeled_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                break

print(f"Applied labels to {count} examples.")
