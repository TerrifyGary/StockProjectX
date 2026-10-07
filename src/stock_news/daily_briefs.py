from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any
from zoneinfo import ZoneInfo

from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError

from stock_news.evidence_brief import build_evidence_brief
from stock_news.forecast_tracking import register_pending_batches, evaluate_pending_forecasts, tracking_summary
from stock_news.trading_sessions import session_close, next_session


def fetch_price_snapshots(
    stocks: list[dict[str, Any]], pending_since: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Capture completed, unadjusted bars, including exact dates needed by pending calls."""
    import yfinance as yf

    now = datetime.now(timezone.utc)
    snapshots = []
    for stock in stocks:
        try:
            start = (now - timedelta(days=35)).date().isoformat()
            if (pending_since or {}).get(stock["id"]):
                start = min(start, (pending_since or {})[stock["id"]])
            history = yf.Ticker(stock["ticker"]).history(
                start=start, interval="1d", auto_adjust=False, actions=True
            )
            bars = []
            for timestamp, row in history.iterrows():
                date = timestamp.strftime("%Y-%m-%d")
                close = float(row.get("Close", 0))
                if not close > 0 or not math.isfinite(close) or session_close(stock, date) > now:
                    continue
                bars.append({"date": date, "close": close, "split": float(row.get("Stock Splits", 0) or 0)})
            if not bars:
                continue
            previous_direction = "uncertain"
            if len(bars) >= 2 and not bars[-1]["split"] and next_session(stock, bars[-2]["date"]) == bars[-1]["date"]:
                difference = bars[-1]["close"] - bars[-2]["close"]
                previous_direction = "up" if difference > 0 else "down" if difference < 0 else "uncertain"
            snapshots.append({
                "id": stock["id"], "symbol": stock["display_symbol"],
                "ticker": stock["ticker"], "market": stock["market"],
                **bars[-1], "previous_direction": previous_direction, "history": bars,
                "previous_close": bars[-2]["close"] if len(bars) >= 2 else None,
                "previous_session": bars[-2]["date"] if len(bars) >= 2 else None,
            })
        except Exception as error:
            print(f"[price snapshot error] {stock.get('ticker')}: {error}", flush=True)
    return snapshots


def make_daily_brief(
    *,
    database: Any,
    window_start: datetime,
    window_end: datetime,
    timezone_name: str,
    summarizer: Any,
    companies: list[dict[str, Any]],
    impact_model_name: str,
    impact_model_device: str,
    analysis_model_name: str,
    analysis_model_device: str,
    prices: list[dict[str, Any]],
    saved: int,
    skipped: int,
    failed: int,
) -> dict[str, Any]:

    articles = list(
        database.articles.find(
            {"published_at": {"$gte": window_start, "$lt": window_end}},
            {
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
            },
        )
        .sort("published_at", DESCENDING)
    )
    for article in articles:
        article["display_title"] = article.get("title_en") or article.get("title") or ""
        summary = article.get("summary", {})
        article["display_summary"] = summary.get("text_en") or summary.get("text") or ""
    evidence_brief = build_evidence_brief(articles, reference_time=window_end)

    if articles:
        try:
            from stock_news.local_analysis import get_local_news_impact_model

            impact_analysis = get_local_news_impact_model(
                analysis_model_name, analysis_model_device
            ).analyze(
                articles,
                companies,
                reference_time=window_end,
                prepared_evidence_brief=evidence_brief,
            )
            impact_status = "available"
            impact_error = None
            impact_provider = "huggingface-local"
            impact_model = analysis_model_name
        except Exception as error:
            try:
                from stock_news.impact import get_sentiment_analyzer

                impact_analysis = get_sentiment_analyzer(
                    impact_model_name, impact_model_device
                ).analyze(articles)
                impact_analysis["investor_note"] = "Local impact analysis was unavailable; these fallback cues are sentiment labels, not price forecasts."
                impact_status = "fallback_sentiment"
                impact_error = f"{type(error).__name__}: {error}"
                impact_provider = "huggingface-local-fallback"
                impact_model = impact_model_name
            except Exception as fallback_error:
                impact_analysis = {"upside": [], "downside": [], "neutral": [], "directions": [], "investor_note": "The local news analysis model could not run. Review the saved articles and source links."}
                impact_status = "unavailable"
                impact_error = f"{type(fallback_error).__name__}: {fallback_error}"
                impact_provider = "unavailable"
                impact_model = analysis_model_name

        investor_note = impact_analysis.pop("investor_note", "")
        if not investor_note or impact_status == "fallback_sentiment":
            summaries = [
                " ".join(part for part in (item["display_title"], item["display_summary"]) if part)
                for item in articles
            ]
            summaries = [text for text in summaries if text]
            try:
                chunk_summaries = [
                    summarizer.summarize("\n".join(summaries[start : start + 5]), "en")
                    for start in range(0, len(summaries), 5)
                ]
                investor_note = summarizer.summarize("\n".join(chunk_summaries), "en")
                investor_note = f"News summary across {len(articles)} collected stories: {investor_note} Treat cues as hypotheses and check sources."
            except Exception as summary_error:
                investor_note = f"Collected {len(articles)} stories. Local summary unavailable ({type(summary_error).__name__}); review the linked reporting."
    else:
        impact_analysis = {"upside": [], "downside": [], "neutral": [], "directions": []}
        impact_status = "available"
        impact_error = None
        impact_provider = "huggingface-local"
        impact_model = analysis_model_name
        investor_note = "No matching company or global-topic stories were collected in this window."

    impact_analysis["evidence_brief"] = evidence_brief
    report_date = window_end.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    current = database.current_brief.find_one({"_id": "current"})
    is_new_report_day = bool(current and current.get("report_date") != report_date)
    if current and is_new_report_day:
        archived = dict(current)
        archived.pop("_id", None)
        database.brief_history.replace_one(
            {"report_date": archived["report_date"]}, archived, upsert=True
        )

    # Freeze original inputs once per report date; display/news refreshes remain independent.
    database.forecast_briefs.create_index("report_date", unique=True)
    frozen = database.forecast_briefs.find_one({"report_date": report_date}, {"_id": 0})
    clean_prices = [{k: v for k, v in stock.items() if k != "history"} for stock in prices]
    if frozen is None:
        from stock_news.lstm import predict_next_directions

        candidate = {
            "report_date": report_date, "created_at": datetime.now(timezone.utc),
            "prices": clean_prices, "impact_analysis": {"model": impact_model, **impact_analysis},
            "article_keys": [article["article_key"] for article in articles],
        }
        history = list(database.forecast_briefs.find({}, {"_id": 0}).sort("created_at", 1))
        candidate["lstm"] = predict_next_directions(history, candidate)
        # Issuance follows inference; a late model cannot claim a session that already closed.
        candidate["created_at"] = datetime.now(timezone.utc)
        try:
            database.forecast_briefs.update_one(
                {"report_date": report_date}, {"$setOnInsert": candidate}, upsert=True
            )
        except DuplicateKeyError:
            pass  # Another collector won the unique report-date insertion.
        frozen = database.forecast_briefs.find_one({"report_date": report_date}, {"_id": 0})
    register_pending_batches(database)
    registered = database.forecast_briefs.find_one({"report_date": report_date}, {"_id": 0})
    registration = registered.get("registration", {})
    evaluate_pending_forecasts(database, prices)
    lstm_result = frozen["lstm"]
    brief = {
        "_id": "current",
        "report_date": report_date,
        "created_at": datetime.now(window_end.tzinfo),
        "window_start": window_start,
        "window_end": window_end,
        "article_keys": [article["article_key"] for article in articles],
        "article_count": len(articles),
        "impact_analysis": {
            "status": impact_status,
            "provider": impact_provider,
            "model": impact_model,
            "method": "local_news_impact_analysis_with_evidence_priority_v1_and_next_session_validation",
            "error": impact_error,
            **impact_analysis,
        },
        "daily_takeaway": investor_note,
        "prices": clean_prices,
        "lstm": lstm_result,
        "forecast_issued_at": frozen["created_at"],
        "forecast_inputs_frozen": True,
        "forecast_registration": registration,
        "collection": {"saved": saved, "skipped": skipped, "failed": failed},
    }
    database.current_brief.replace_one({"_id": "current"}, brief, upsert=True)
    database.brief_history.create_index("report_date", unique=True)
    return brief


def validation_summary(database: Any) -> dict[str, Any]:
    current = database.current_brief.find_one({"_id": "current"}, {"_id": 0, "report_date": 1, "lstm": 1, "forecast_registration": 1}) or {}
    summary = tracking_summary(database)
    dates = set(database.brief_history.distinct("report_date"))
    if current.get("report_date"):
        dates.add(current["report_date"])
    news = summary["models"]["news_only"]
    return {
        **summary, "days_running": len(dates),
        "status": "available" if summary["evaluated_calls"] else "collecting_history",
        "cue_accuracy_percent": news["accuracy_percent"],
        "cue_evaluated_calls": news["evaluated_calls"], "cue_correct_calls": news["correct_calls"],
        "lstm_status": current.get("lstm", {}).get("status", "warming_up"),
        "lstm_training_days": current.get("lstm", {}).get("training_days", 0),
        "lstm_required_days": 30, "lstm_required_samples": 40,
        "training_coverage": current.get("lstm", {}).get("per_security", []),
        "legacy_evaluations_included": False,
        "reference_price_rejections": current.get("forecast_registration", {}).get("rejected", []),
    }
