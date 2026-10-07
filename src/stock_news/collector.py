from __future__ import annotations

import hashlib
import html
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx
import trafilatura


TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
}
_last_gdelt_request_at = 0.0


def build_gdelt_feeds(
    config: dict[str, Any],
    companies: list[dict[str, Any]],
    topics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not config.get("enabled", False):
        return []

    terms: set[str] = set()
    for company in companies:
        # Keep one canonical query term per company so the combined request
        # stays short; full alias matching still happens on returned stories.
        candidates = company.get("aliases", [])
        candidate = (
            candidates[0]
            if candidates
            else (company.get("tickers", [company["name"]])[0])
        )
        term = str(candidate).strip()
        if term and term.isascii():
            terms.add(term)
    for topic in topics or []:
        # One representative phrase per topic keeps the combined query short.
        english_terms = topic.get("terms", {}).get("en", [])
        representative = next(
            (str(term).strip() for term in english_terms if len(str(term).replace(" ", "")) >= 4),
            "",
        )
        if representative:
            terms.add(representative)
    if not terms:
        return []

    query_terms = " OR ".join(
        f'"{term.replace(chr(34), "")}"' if " " in term else term.replace(chr(34), "")
        for term in sorted(terms)
    )
    timespan = str(config.get("timespan", "1day"))
    max_records = min(max(int(config.get("max_records", 5)), 1), 250)
    feeds = []
    for language, gdelt_language in GDELT_LANGUAGES.items():
        query = f"({query_terms}) sourcelang:{gdelt_language}"
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(max_records),
            "timespan": timespan,
            "sort": "datedesc",
            "format": "rss",
        }
        feeds.append(
            {
                "name": f"GDELT {gdelt_language.title()} Search",
                "url": f"{GDELT_ENDPOINT}?{urlencode(params)}",
                "language": language,
                "max_articles": max(int(config.get("max_articles", 25)), 1),
                "source_type": "gdelt_doc_api",
            }
        )
    return feeds


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(query), "")
    )


def article_key(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()


def _entry_date(entry: Any) -> datetime | None:
    iso_date = entry.get("publish_date")
    if iso_date:
        try:
            value = datetime.fromisoformat(str(iso_date).replace("Z", "+00:00"))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            pass
    for key in ("published", "updated"):
        raw_date = entry.get(key)
        if raw_date:
            try:
                value = parsedate_to_datetime(raw_date)
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                return value.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pass
    for key in ("published_parsed", "updated_parsed"):
        parts = entry.get(key)
        if parts:
            try:
                return datetime(*parts[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pass
    return None


def _plain_text(value: str) -> str:
    if not value:
        return ""
    decoded = html.unescape(value)
    cleaned = trafilatura.html2txt(decoded) or re.sub(r"<[^>]+>", " ", decoded)
    return re.sub(r"\s+", " ", cleaned).strip()


def _matches_company(text: str, company: dict[str, Any]) -> bool:
    folded = text.casefold()
    candidates = [company["name"], *company.get("aliases", [])]
    for candidate in candidates:
        alias = str(candidate).strip()
        if alias and alias.casefold() in folded:
            return True
    for ticker in company.get("tickers", []):
        ticker = str(ticker).strip()
        if ticker and re.search(rf"(?<![\w]){re.escape(ticker)}(?![\w])", text, re.IGNORECASE):
            return True
    return False


def _matches_topic(text: str, topic: dict[str, Any]) -> bool:
    folded = text.casefold()
    language_terms = topic.get("terms", {})
    for terms in language_terms.values():
        for term in terms:
            candidate = str(term).strip().casefold()
            if not candidate:
                continue
            if candidate.isascii():
                if re.search(rf"(?<![\w]){re.escape(candidate)}(?![\w])", folded):
                    return True
            elif candidate in folded:
                return True
    return False


def _extract_article(client: httpx.Client, url: str) -> str:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError:
        return ""
    return trafilatura.extract(
        response.text,
        url=str(response.url),
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    ) or ""


def _fetch_feed(client: httpx.Client, feed: dict[str, Any]) -> httpx.Response:
    global _last_gdelt_request_at
    is_gdelt = feed.get("source_type") == "gdelt_doc_api"
    for attempt in range(2 if is_gdelt else 1):
        if is_gdelt:
            wait_seconds = 5.5 - (time.monotonic() - _last_gdelt_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        response = client.get(feed["url"])
        if is_gdelt:
            _last_gdelt_request_at = time.monotonic()
            if response.status_code == 429 and attempt == 0:
                print(f"[rate limit] {feed['name']}: waiting 5.5 seconds and retrying")
                time.sleep(5.5)
                continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"Could not fetch feed {feed['name']}")


def _entity_document(company: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": company["id"],
        "name": company["name"],
        "tickers": company.get("tickers", []),
        "exchanges": company.get("exchanges", []),
        "markets": company.get("markets", []),
    }


def collect_feed(
    *,
    feed: dict[str, Any],
    companies: list[dict[str, Any]],
    collection: Any,
    summarizer: Any,
    translator: Any,
    timeout_seconds: float,
    user_agent: str,
    max_article_age_hours: int = 24,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    topics: list[dict[str, Any]] | None = None,
    entries_override: list[dict[str, Any]] | None = None,
) -> tuple[int, int, int]:
    """Collect one configured feed; return (inserted/updated, skipped, failed)."""
    headers = {"User-Agent": user_agent}
    with httpx.Client(
        headers=headers, timeout=timeout_seconds, follow_redirects=True
    ) as client:
        if entries_override is None:
            try:
                response = _fetch_feed(client, feed)
            except httpx.HTTPError as error:
                print(f"[feed error] {feed['name']}: {error}")
                return 0, 0, 1

            parsed = feedparser.parse(response.content)
            if parsed.bozo and not parsed.entries:
                excerpt = response.text[:240].replace("\n", " ")
                print(
                    f"[feed parse error] {feed['name']}: {parsed.bozo_exception}; "
                    f"response begins {excerpt!r}"
                )
                return 0, 0, 1
            entries = parsed.entries
        else:
            entries = entries_override

        updated = skipped = failed = matched_count = 0
        max_articles = max(int(feed.get("max_articles", 5)), 1)
        now = datetime.now(timezone.utc)
        cutoff = window_start or now - timedelta(hours=max_article_age_hours)
        cutoff_end = window_end or now
        dated_entries = sorted(
            entries,
            key=lambda entry: _entry_date(entry) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        for entry in dated_entries:
            language = str(entry.get("language") or feed["language"])
            if language not in {"en", "zh", "zh-Hant", "zh-Hans", "ja"}:
                skipped += 1
                continue
            title = _plain_text(entry.get("title", ""))
            feed_excerpt = _plain_text(
                entry.get("summary", "")
                or entry.get("description", "")
                or ""
            )
            url = entry.get("link", "").strip()
            if not title or not url:
                skipped += 1
                continue

            published_at = _entry_date(entry)
            if published_at is None or published_at < cutoff or published_at >= cutoff_end:
                skipped += 1
                continue

            article_text = _extract_article(client, url)
            match_text = " ".join((title, feed_excerpt, article_text))
            configured_ids = set(feed.get("company_ids", []))
            matched = [
                company
                for company in companies
                if company["id"] in configured_ids or _matches_company(match_text, company)
            ]
            matched_topics = [
                topic["id"]
                for topic in topics or []
                if _matches_topic(match_text, topic)
            ]
            if not matched and not matched_topics:
                skipped += 1
                continue
            matched_count += 1

            canonical_url = canonicalize_url(url)
            key = article_key(canonical_url)
            existing = collection.find_one(
                {"article_key": key},
                {"summary.status": 1, "summary.text_en": 1, "title_en": 1},
            )
            if (
                existing
                and existing.get("summary", {}).get("status") == "complete"
                and existing.get("title_en")
                and existing.get("summary", {}).get("text_en")
            ):
                skipped += 1
                continue

            summary_input = article_text or feed_excerpt or title
            article_domain = urlsplit(canonical_url).hostname
            source_name = feed["name"]
            record = {
                "article_key": key,
                "title": title,
                "title_en": title if language == "en" else None,
                "url": canonical_url,
                "source": {
                    "name": article_domain or source_name,
                    "discovered_by": feed.get("discovered_by", source_name),
                    "feed_url": feed["url"],
                    "type": feed.get("source_type", "rss_or_atom"),
                },
                "published_at": published_at,
                "collected_at": now,
                "language": language,
                "markets": sorted(
                    set(feed.get("markets", []))
                    | {
                        market
                        for company in matched
                        for market in company.get("markets", [])
                    }
                ),
                "entities": [_entity_document(company) for company in matched],
                "topics": matched_topics,
                "summary": {
                    "text": None,
                    "text_en": None,
                    "language": language,
                    "provider": summarizer.provider,
                    "model": summarizer.model_name,
                    "status": "pending",
                    "generated_at": None,
                    "error": None,
                },
                "translation": {
                    "target_language": "en",
                    "provider": translator.provider,
                    "model": translator.model_for_language(language),
                    "status": "pending",
                    "generated_at": None,
                    "error": None,
                },
                "content_storage": "summary_and_metadata_only",
            }

            try:
                summary = summarizer.summarize(summary_input, language)
                record["summary"].update(
                    text=summary,
                    status="complete",
                    generated_at=datetime.now(timezone.utc),
                )
            except Exception as error:
                record["summary"].update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}"[:500],
                )
                collection.update_one(
                    {"article_key": key},
                    {"$set": record, "$setOnInsert": {"first_seen_at": now}},
                    upsert=True,
                )
                print(f"[summary error] {title}: {error}")
                failed += 1
                continue

            if language == "en":
                title_en, summary_en = title, summary
            else:
                try:
                    title_en, summary_en = translator.translate_many(
                        [title, summary], language
                    )
                    record["translation"].update(
                        status="complete",
                        generated_at=datetime.now(timezone.utc),
                    )
                except Exception as error:
                    record["translation"].update(
                        status="failed",
                        error=f"{type(error).__name__}: {error}"[:500],
                    )
                    collection.update_one(
                        {"article_key": key},
                        {"$set": record, "$setOnInsert": {"first_seen_at": now}},
                        upsert=True,
                    )
                    print(f"[translation error] {title}: {error}")
                    failed += 1
                    continue
            record["title_en"] = title_en
            record["summary"]["text_en"] = summary_en
            if language == "en":
                record["translation"].update(
                    status="not_needed",
                    generated_at=datetime.now(timezone.utc),
                )

            collection.update_one(
                {"article_key": key},
                {"$set": record, "$setOnInsert": {"first_seen_at": now}},
                upsert=True,
            )
            updated += 1
            print(f"[saved] {title}")
            if matched_count >= max_articles:
                break
    return updated, skipped, failed
