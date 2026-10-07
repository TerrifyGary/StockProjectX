from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pymongo import DESCENDING, MongoClient

from stock_news.config import Settings, load_yaml
from stock_news.news_window import latest_completed_window
from stock_news.impact import get_sentiment_analyzer
from stock_news.daily_briefs import validation_summary
from stock_news.evidence_brief import build_evidence_brief


load_dotenv()
settings = Settings.from_environment()
mongo_client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
database = mongo_client[settings.mongo_database]
app = FastAPI(title="Stock News Dashboard API", version="0.1.0")
_quote_lock = threading.Lock()
_quote_cache: dict[str, Any] = {"expires": 0.0, "quotes": []}


def _same_instant(left: Any, right: Any) -> bool:
    """Compare UTC instants across PyMongo's default naive datetime decoding."""
    if not left or not right:
        return False
    if left.tzinfo is None:
        left = left.replace(tzinfo=ZoneInfo("UTC"))
    if right.tzinfo is None:
        right = right.replace(tzinfo=ZoneInfo("UTC"))
    return left.astimezone(ZoneInfo("UTC")) == right.astimezone(ZoneInfo("UTC"))


def _stock_config() -> list[dict[str, Any]]:
    config_path = Path("config/dashboard.yaml")
    if not config_path.exists():
        return []
    return load_yaml(config_path).get("stocks", [])


def _get_quotes() -> list[dict[str, Any]]:
    with _quote_lock:
        if time.monotonic() < _quote_cache["expires"]:
            return _quote_cache["quotes"]

        results: list[dict[str, Any]] = []
        for stock in _stock_config():
            quote = {
                **stock,
                "price": None,
                "change_percent": None,
                "as_of": None,
                "history": [],
                "status": "unavailable",
            }
            try:
                ticker = yf.Ticker(stock["ticker"])
                history = ticker.history(period="1mo", interval="1d", auto_adjust=True)
                closes = history.get("Close")
                if closes is not None:
                    closes = closes.dropna()
                if closes is None or len(closes) == 0:
                    raise ValueError("No recent market prices were returned")

                latest_close = float(closes.iloc[-1])
                prior_close = float(closes.iloc[-2]) if len(closes) > 1 else latest_close
                price = latest_close
                previous_close = prior_close
                try:
                    fast_info = ticker.fast_info
                    price = fast_info.get("last_price") or latest_close
                    previous_close = fast_info.get("previous_close") or prior_close
                except Exception:
                    pass
                quote.update(
                    price=float(price),
                    change_percent=(float(price) - float(previous_close))
                    / float(previous_close)
                    * 100
                    if previous_close
                    else 0.0,
                    as_of=closes.index[-1].isoformat(),
                    history=[
                        {"date": index.strftime("%Y-%m-%d"), "close": round(float(value), 4)}
                        for index, value in closes.tail(20).items()
                    ],
                    status="available",
                )
            except Exception as error:
                quote["error"] = f"{type(error).__name__}: {error}"
            results.append(quote)

        _quote_cache["quotes"] = results
        _quote_cache["expires"] = time.monotonic() + 300
        return results


@app.get("/api/health")
def health() -> dict[str, str]:
    try:
        database.command("ping")
    except Exception as error:
        raise HTTPException(status_code=503, detail="MongoDB unavailable") from error
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    try:
        window_start, window_end = latest_completed_window(
            settings.window_timezone, settings.window_start_hour,
            window_hours=settings.window_hours,
        )
        article_projection = {
            "_id": 0,
            "article_key": 1,
            "title": 1,
            "title_en": 1,
            "url": 1,
            "source": 1,
            "published_at": 1,
            "language": 1,
            "markets": 1,
            "entities": 1,
            "topics": 1,
            "summary": 1,
            "translation.status": 1,
        }
        brief = database.current_brief.find_one({"_id": "current"})
        has_current_brief = bool(
            brief and _same_instant(brief.get("window_end"), window_end)
        )
        if has_current_brief:
            article_keys = brief.get("article_keys", [])
            articles = list(
                database.articles.find(
                    {"article_key": {"$in": article_keys}}, article_projection
                ).sort("published_at", DESCENDING).limit(5)
            )
            news_count = int(brief.get("article_count", len(article_keys)))
            window_news_count = news_count
            if not articles and news_count == 0:
                articles = list(
                    database.articles.find(
                        {"published_at": {"$lt": window_start}}, article_projection
                    )
                    .sort("published_at", DESCENDING)
                    .limit(5)
                )
                news_mode = "archive" if articles else "window"
                news_count = len(articles)
            else:
                news_mode = "window"
        else:
            articles = list(
                database.articles.find(
                    {"published_at": {"$gte": window_start, "$lt": window_end}},
                    article_projection,
                )
                .sort("published_at", DESCENDING)
                .limit(5)
            )
            news_count = database.articles.count_documents(
                {"published_at": {"$gte": window_start, "$lt": window_end}}
            )
            window_news_count = news_count
            news_mode = "window"
        for article in articles:
            article["published_at"] = article["published_at"].isoformat()
            article["display_title"] = article.get("title_en") or article.get("title")
            summary = article.get("summary", {})
            article["display_summary"] = summary.get("text_en") or summary.get("text") or ""

        if has_current_brief:
            saved_analysis = brief.get("impact_analysis", {})
            impact_candidates = {
                key: saved_analysis.get(key, [])
                for key in ("upside", "downside", "neutral", "directions")
            }
            impact_candidates["evidence_brief"] = saved_analysis.get("evidence_brief")
            impact_status = (
                "no_articles"
                if int(brief.get("article_count", 0)) == 0
                else saved_analysis.get("status", "unavailable")
            )
            impact_error = saved_analysis.get("error")
        else:
            try:
                impact_candidates = get_sentiment_analyzer(
                    settings.impact_model_name, settings.impact_model_device
                ).analyze(articles)
                impact_status = "available"
                impact_error = None
            except Exception as error:
                impact_candidates = {"upside": [], "downside": [], "neutral": [], "directions": []}
                impact_status = "unavailable"
                impact_error = f"{type(error).__name__}: {error}"
            impact_candidates["evidence_brief"] = build_evidence_brief(
                articles, reference_time=window_end
            )

        run = database.collection_runs.find_one(
            {"window_start": window_start, "window_end": window_end},
            {"_id": 0}, sort=[("updated_at", DESCENDING)]
        )
        quotes = _get_quotes()
        usable_quotes = [quote for quote in quotes if quote.get("status") == "available"]
        if usable_quotes:
            market_takeaway = "Latest available close: " + "; ".join(
                f"{quote['display_symbol']} {quote['change_percent']:+.2f}%"
                for quote in usable_quotes
            ) + ". Review the linked news and source details before making a decision."
        else:
            market_takeaway = "Market prices are unavailable right now. Review the linked news and source details before making a decision."
        daily_takeaway = (
            brief.get("daily_takeaway")
            if has_current_brief
            else market_takeaway
        )

        return {
            "window": {
                "timezone": settings.window_timezone,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "label": f"{window_start.astimezone(ZoneInfo(settings.window_timezone)).strftime('%Y-%m-%d %H:%M')} – {window_end.astimezone(ZoneInfo(settings.window_timezone)).strftime('%Y-%m-%d %H:%M')} Taiwan time",
            },
            "last_refresh": run,
            "news_mode": news_mode,
            "news_count": news_count,
            "window_news_count": window_news_count,
            "stocks": quotes,
            "news": articles,
            "validation": validation_summary(database),
            "impact_analysis": {
                "status": impact_status,
                "model": (brief or {}).get("impact_analysis", {}).get("model", settings.analysis_model_name),
                "method": (brief or {}).get("impact_analysis", {}).get("method", "local_news_impact_analysis"),
                "error": impact_error,
                **impact_candidates,
                "message": (
                    "No matching company or global-topic stories were collected in this window; no impact cues were generated."
                    if impact_status == "no_articles"
                    else (
                    f"FinBERT could not load or analyze stories ({impact_error})."
                    if impact_status == "unavailable"
                    else (
                        "The local impact model was unavailable, so FinBERT sentiment labels are shown as a fallback."
                        if impact_status == "fallback_sentiment"
                        else "The local Hugging Face model reviewed every collected story in this window and mapped supported cues to watchlist companies. These are hypotheses, not guaranteed price forecasts; verify the source links."
                    )
                    )
                ),
            },
            "daily_takeaway": daily_takeaway,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Dashboard data unavailable: {error}") from error


@app.get("/api/news")
def news(limit: int = 50) -> dict[str, Any]:
    limit = min(max(limit, 1), 100)
    window_start, window_end = latest_completed_window(
        settings.window_timezone, settings.window_start_hour,
        window_hours=settings.window_hours,
    )
    articles = list(
        database.articles.find(
            {"published_at": {"$gte": window_start, "$lt": window_end}},
            {"_id": 0},
        )
        .sort("published_at", DESCENDING)
        .limit(limit)
    )
    for article in articles:
        if article.get("published_at"):
            article["published_at"] = article["published_at"].isoformat()
    return {"window_start": window_start.isoformat(), "window_end": window_end.isoformat(), "items": articles}


@app.get("/api/history")
def history(limit: int = 30) -> dict[str, Any]:
    limit = min(max(limit, 1), 100)
    briefs = list(
        database.brief_history.find(
            {},
            {
                "_id": 0,
                "report_date": 1,
                "window_start": 1,
                "window_end": 1,
                "article_count": 1,
                "daily_takeaway": 1,
                "impact_analysis.upside": 1,
                "impact_analysis.downside": 1,
                "evaluation": 1,
            },
        )
        .sort("report_date", DESCENDING)
        .limit(limit)
    )
    return {"items": briefs, "validation": validation_summary(database)}
