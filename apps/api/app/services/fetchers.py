# Purpose: Fetches the news articles from the data source
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from urllib.request import urlopen
from xml.etree import ElementTree

from app.models.dashboard import DashboardMode
from app.utils.text import clean_text

# The news article data class
@dataclass(frozen=True)
class NewsArticle:
    title: str
    source: str
    published_at: datetime | None
    snippet: str
    url: str

# Fetches the mock news articles. Mock news are created to test the dashboard locally.
def fetch_mock_news(
    query: str, max_results: int = 5, mode: DashboardMode = DashboardMode.GENERAL
) -> list[NewsArticle]:
    now = datetime.now(timezone.utc)
    topic = clean_text(query)
    mode_label = mode.value
    articles = [
        NewsArticle(
            title=f"{topic} sees new launch momentum",
            source="Mock Business Daily",
            published_at=now - timedelta(hours=1),
            snippet=f"Recent {mode_label} coverage points to a launch, growth, and strong user interest.",
            url="https://example.com/newsboardai/mock-launch",
        ),
        NewsArticle(
            title=f"Analysts weigh possible risks around {topic}",
            source="Mock Market Wire",
            published_at=now - timedelta(hours=3),
            snippet="Coverage also mentions risk, delay concerns, and an unclear near-term outlook.",
            url="https://example.com/newsboardai/mock-risk",
        ),
        NewsArticle(
            title=f"{topic} remains active across recent headlines",
            source="Mock News Desk",
            published_at=now - timedelta(hours=6),
            snippet="Multiple sources describe an update, wider attention, and mixed public reaction.",
            url="https://example.com/newsboardai/mock-update",
        ),
        NewsArticle(
            title=f"New details emerge in {topic} coverage",
            source="Mock Briefing",
            published_at=now - timedelta(hours=10),
            snippet="The story is still developing with limited confirmed details from available sources.",
            url="https://example.com/newsboardai/mock-briefing",
        ),
        NewsArticle(
            title=f"{topic} discussion continues online",
            source="Mock Trends",
            published_at=now - timedelta(hours=14),
            snippet="Attention remains steady, though the signal is not yet decisive.",
            url="https://example.com/newsboardai/mock-trends",
        ),
    ]
    return articles[:max_results]

# Fetches the news articles from the Google News RSS feed.
def fetch_google_news_rss(query: str, max_results: int = 5) -> list[NewsArticle]:
    encoded_query = quote_plus(query)
    url = ( # The URL of the Google News RSS feed
        "https://news.google.com/rss/search"
        f"?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )
    with urlopen(url, timeout=10) as response: # Open the URL and read the response
        xml_data = response.read()

    root = ElementTree.fromstring(xml_data) # Parse the XML data
    items = root.findall("./channel/item")[:max_results] # Find all the items in the RSS feed
    return [_parse_google_news_item(item) for item in items] # Parse the items

# Parses the Google News RSS feed item and returns a NewsArticle object.
def _parse_google_news_item(item: ElementTree.Element) -> NewsArticle:
    title = clean_text(item.findtext("title", default="Untitled"))
    link = item.findtext("link", default="https://news.google.com")
    published_at = _parse_pub_date(item.findtext("pubDate"))
    source = item.findtext("source", default="Google News")
    snippet = clean_text(item.findtext("description", default=""))

    return NewsArticle(
        title=title,
        source=clean_text(source),
        published_at=published_at,
        snippet=snippet,
        url=link,
    )

# Parses the publication date from the RSS feed item.
def _parse_pub_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
