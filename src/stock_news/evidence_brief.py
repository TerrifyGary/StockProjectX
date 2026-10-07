from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _symbols(article: dict[str, Any]) -> list[str]:
    symbols = {
        str(symbol).strip().upper()
        for entity in (article.get("entities") or [])
        if isinstance(entity, dict)
        for symbol in (entity.get("tickers") or [])
        if str(symbol).strip()
    }
    return sorted(symbols)


def _title_key(article: dict[str, Any]) -> str:
    title = article.get("title_en") or article.get("title") or ""
    return re.sub(r"[^\w]+", " ", str(title).casefold()).strip()


def _priority_components(
    article: dict[str, Any],
    *,
    reference_time: datetime,
    duplicate_count: int,
) -> dict[str, float]:
    symbols = _symbols(article)
    has_watchlist_match = bool(article.get("entities"))
    topics = [topic for topic in (article.get("topics") or []) if topic]
    relevance = 1.0 if has_watchlist_match else 0.65 if topics else 0.35

    source = article.get("source") or {}
    source_name = str(source.get("name") or "").strip() if isinstance(source, dict) else ""
    url = str(article.get("url") or "").strip()
    source_traceability = 1.0 if source_name and url else 0.65 if url else 0.35 if source_name else 0.2

    published_at = _as_utc(article.get("published_at"))
    if published_at:
        age_hours = max(0.0, (reference_time - published_at).total_seconds() / 3600)
        freshness = max(0.5, min(1.0, 1.0 - age_hours / 96.0))
    else:
        freshness = 0.5
    novelty = 1.0 / math.sqrt(max(1, duplicate_count))
    return {
        "relevance": round(relevance, 4),
        "source_traceability": round(source_traceability, 4),
        "freshness": round(freshness, 4),
        "novelty": round(novelty, 4),
    }


def build_evidence_brief(
    articles: list[dict[str, Any]],
    *,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Build a rule-based article-priority brief; it does not predict direction."""
    reference = _as_utc(reference_time) or datetime.now(timezone.utc)
    title_counts = Counter(key for article in articles if (key := _title_key(article)))
    article_scores: list[dict[str, Any]] = []
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_topic_counts: Counter[str] = Counter()

    for article_id, article in enumerate(articles):
        title_key = _title_key(article)
        duplicate_count = title_counts.get(title_key, 1) if title_key else 1
        components = _priority_components(
            article,
            reference_time=reference,
            duplicate_count=duplicate_count,
        )
        priority_score = round(100 * math.prod(components.values()), 1)
        symbols = _symbols(article)
        topics = sorted({str(topic) for topic in (article.get("topics") or []) if topic})
        source = article.get("source") or {}
        source_name = (
            str(source.get("name") or "").strip()
            if isinstance(source, dict)
            else ""
        )
        score = {
            "article_id": article_id,
            "priority_score": priority_score,
            "components": components,
            "symbols": symbols,
            "topics": topics,
            "duplicate_title_count": duplicate_count,
        }
        article_scores.append(score)
        for symbol in symbols:
            by_symbol[symbol].append({**score, "source_name": source_name})
        for topic in topics:
            global_topic_counts[topic] += 1

    ticker_groups = []
    for symbol, items in sorted(
        by_symbol.items(),
        key=lambda pair: sum(row["priority_score"] for row in pair[1]) / len(pair[1]),
        reverse=True,
    ):
        ticker_groups.append(
            {
                "symbol": symbol,
                "story_count": len(items),
                "source_count": len({item["source_name"] for item in items if item["source_name"]}),
                "mean_priority_score": round(
                    sum(item["priority_score"] for item in items) / len(items), 1
                ),
                "article_ids": [item["article_id"] for item in items],
            }
        )

    global_story_count = sum(1 for article in articles if article.get("topics"))
    watchlist_story_count = sum(1 for article in articles if article.get("entities"))
    lines = [
        f"Rule-based evidence priority for {len(articles)} stories; scores rank evidence for review and are not price probabilities or direction calls."
    ]
    if watchlist_story_count:
        lines.append(f"Watchlist-linked stories: {watchlist_story_count}.")
    for group in ticker_groups:
        lines.append(
            f"{group['symbol']}: {group['story_count']} directly matched stories from {group['source_count']} sources; mean priority {group['mean_priority_score']}/100."
        )
    if global_story_count:
        topic_text = ", ".join(
            f"{topic} ({count})" for topic, count in global_topic_counts.most_common(6)
        )
        lines.append(
            f"Global-topic evidence: {global_story_count} stories; topics {topic_text or 'not specified'}. Map to a ticker only when the story supports a clear transmission path."
        )
    if not articles:
        lines.append("No stories are available for scoring.")

    return {
        "method": "rule_based_evidence_priority_v1",
        "score_interpretation": "0-100 ranking aid only; not direction, probability, source credibility, or expected price movement.",
        "story_count": len(articles),
        "watchlist_story_count": watchlist_story_count,
        "global_story_count": global_story_count,
        "ticker_groups": ticker_groups,
        "global_topic_counts": dict(global_topic_counts),
        "article_scores": article_scores,
        "summary": "\n".join(lines),
    }
