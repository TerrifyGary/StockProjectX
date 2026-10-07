from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Any


LANGUAGES = ("en", "zh", "ja")
LANGUAGE_TERMS = {
    "en": ("en",),
    "zh": ("zh-Hant", "zh-Hans"),
    "ja": ("ja",),
}


def _query_term(term: str) -> str:
    cleaned = " ".join(str(term).replace('"', "").split())
    if not cleaned:
        return ""
    return f'"{cleaned}"' if " " in cleaned else cleaned


def _terms_for_language(
    language: str,
    companies: list[dict[str, Any]],
    topics: list[dict[str, Any]],
) -> list[str]:
    terms: list[str] = []
    for company in companies:
        terms.extend(company.get("aliases", []))
        terms.extend(company.get("tickers", []))
        terms.append(company.get("name", ""))
    for topic in topics:
        for topic_language in LANGUAGE_TERMS[language]:
            terms.extend(topic.get("terms", {}).get(topic_language, []))

    deduplicated: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = str(term).strip()
        folded = value.casefold()
        if value and folded not in seen:
            seen.add(folded)
            deduplicated.append(value)
        if len(deduplicated) >= 100:
            break
    return deduplicated


def build_mediacloud_feeds(
    config: dict[str, Any],
    companies: list[dict[str, Any]],
    topics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not config.get("enabled", False):
        return []

    collection_ids = [int(value) for value in config.get("collection_ids", [])]
    if not collection_ids:
        print(
            "[source disabled] Media Cloud is enabled but no collection_ids are set "
            "in config/sources.yaml. Choose the news collections to search first."
        )
        return []
    if not os.getenv("MEDIACLOUD_API_KEY", "").strip():
        print(
            "[source disabled] Media Cloud is enabled but MEDIACLOUD_API_KEY is not set. "
            "Add your API key to .env."
        )
        return []

    max_records = min(max(int(config.get("max_records", 100)), 1), 1000)
    feeds: list[dict[str, Any]] = []
    for language in LANGUAGES:
        query_terms = [
            _query_term(term)
            for term in _terms_for_language(language, companies, topics)
        ]
        query_terms = [term for term in query_terms if term]
        if not query_terms:
            continue
        query = f'language:{language} AND ({" OR ".join(query_terms)})'
        feeds.append(
            {
                "name": f"Media Cloud Search ({language})",
                "language": language,
                "markets": ["GLOBAL"],
                "max_articles": max_records,
                "source_type": "mediacloud_api",
                "discovered_by": "Media Cloud Search API",
                "query": query,
                "collection_ids": collection_ids,
            }
        )
    return feeds


def fetch_mediacloud_entries(
    feed: dict[str, Any], window_start: datetime, window_end: datetime
) -> list[dict[str, Any]]:
    """Search Media Cloud for story metadata; publishers remain the source of text."""
    try:
        import mediacloud.api
    except ImportError as error:
        raise RuntimeError(
            "Media Cloud support requires the mediacloud Python package. "
            "Rebuild the collector image after updating dependencies."
        ) from error

    api = mediacloud.api.SearchApi(os.environ["MEDIACLOUD_API_KEY"].strip())
    start_date = window_start.astimezone(timezone.utc).date()
    end_date = (window_end.astimezone(timezone.utc) - timedelta(microseconds=1)).date()
    stories, _ = api.story_list(
        feed["query"],
        start_date=start_date,
        end_date=end_date,
        collection_ids=feed["collection_ids"],
        page_size=feed["max_articles"],
    )

    entries: list[dict[str, Any]] = []
    for story in stories:
        published = story.get("publish_date")
        if isinstance(published, datetime):
            published = published.isoformat()
        elif isinstance(published, date):
            # Media Cloud provides a publication date without a time. Use a
            # stable daytime estimate so the collector does not drop every
            # story on the first calendar date of its 48-hour window.
            estimated = datetime.combine(published, time(hour=12), tzinfo=timezone.utc)
            if published == start_date:
                estimated = max(estimated, window_start.astimezone(timezone.utc))
            published = estimated.isoformat()
        language = str(story.get("language") or feed["language"])
        if language == "zh":
            language = "zh"
        entries.append(
            {
                "title": story.get("title", ""),
                "link": story.get("url", ""),
                "publish_date": published,
                "language": language,
                "summary": "",
            }
        )
    return entries
