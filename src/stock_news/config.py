from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    mongo_uri: str
    mongo_database: str
    model_name: str
    model_device: str
    http_timeout_seconds: float
    user_agent: str
    translation_model_zh: str
    translation_model_ja: str
    impact_model_name: str
    impact_model_device: str
    analysis_model_name: str
    analysis_model_device: str
    window_timezone: str
    window_start_hour: int
    window_hours: int

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            mongo_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            mongo_database=os.getenv("MONGODB_DATABASE", "stock_news"),
            model_name=os.getenv(
                "NEWS_MODEL_NAME", "csebuetnlp/mT5_multilingual_XLSum"
            ),
            model_device=os.getenv("NEWS_MODEL_DEVICE", "auto").lower(),
            http_timeout_seconds=float(os.getenv("NEWS_HTTP_TIMEOUT_SECONDS", "20")),
            user_agent=os.getenv(
                "NEWS_USER_AGENT", "StockNewsEventMonitor/0.1 (personal research)"
            ),
            translation_model_zh=os.getenv(
                "NEWS_TRANSLATION_MODEL_ZH", "Helsinki-NLP/opus-mt-zh-en"
            ),
            translation_model_ja=os.getenv(
                "NEWS_TRANSLATION_MODEL_JA", "Helsinki-NLP/opus-mt-ja-en"
            ),
            impact_model_name=os.getenv("NEWS_IMPACT_MODEL_NAME", "ProsusAI/finbert"),
            impact_model_device=os.getenv("NEWS_IMPACT_MODEL_DEVICE", "auto").lower(),
            analysis_model_name=os.getenv(
                "NEWS_ANALYSIS_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"
            ),
            analysis_model_device=os.getenv("NEWS_ANALYSIS_MODEL_DEVICE", "cpu").lower(),
            window_timezone=os.getenv("NEWS_WINDOW_TIMEZONE", "Asia/Taipei"),
            window_start_hour=int(os.getenv("NEWS_WINDOW_START_HOUR", "8")),
            window_hours=int(os.getenv("NEWS_WINDOW_HOURS", "48")),
        )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return loaded


def load_feeds(path: Path) -> list[dict[str, Any]]:
    feeds = load_yaml(path).get("feeds", [])
    if not isinstance(feeds, list):
        raise ValueError(f"'feeds' must be a list in {path}")
    for index, feed in enumerate(feeds):
        if not isinstance(feed, dict) or not feed.get("name") or not feed.get("url"):
            raise ValueError(f"Feed {index + 1} needs a name and url in {path}")
        language = str(feed.get("language", "en"))
        if language not in {"en", "zh", "zh-Hant", "zh-Hans", "ja"}:
            raise ValueError(
                f"Feed {feed['name']!r} language must be en, zh, zh-Hant, zh-Hans, or ja"
            )
        feed["language"] = language
    return feeds


def load_companies(path: Path) -> list[dict[str, Any]]:
    companies = load_yaml(path).get("companies", [])
    if not isinstance(companies, list):
        raise ValueError(f"'companies' must be a list in {path}")
    for index, company in enumerate(companies):
        if not isinstance(company, dict) or not company.get("id") or not company.get("name"):
            raise ValueError(f"Company {index + 1} needs an id and name in {path}")
        if not isinstance(company.get("aliases", []), list):
            raise ValueError(f"Company {company['id']!r} aliases must be a list")
    return companies


def load_topics(path: Path) -> list[dict[str, Any]]:
    topics = load_yaml(path).get("topics", [])
    if not isinstance(topics, list):
        raise ValueError(f"'topics' must be a list in {path}")
    for index, topic in enumerate(topics):
        if not isinstance(topic, dict) or not topic.get("id") or not isinstance(topic.get("terms"), dict):
            raise ValueError(f"Topic {index + 1} needs an id and language terms in {path}")
        for language, terms in topic["terms"].items():
            if language not in {"en", "zh-Hant", "zh-Hans", "ja"} or not isinstance(terms, list):
                raise ValueError(
                    f"Topic {topic['id']!r} terms must map supported languages to lists in {path}"
                )
    return topics
