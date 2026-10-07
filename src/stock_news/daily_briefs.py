from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pymongo import DESCENDING

from stock_news.evidence_brief import build_evidence_brief


def fetch_price_snapshots(stocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Capture the latest completed daily close for each configured security."""
    import yfinance as yf

    snapshots = []
    for stock in stocks:
        try:
            history = yf.Ticker(stock["ticker"]).history(
                period="5d", interval="1d", auto_adjust=True
            )
            closes = history.get("Close")
            if closes is not None:
                closes = closes.dropna()
            if closes is None or closes.empty:
                continue
            snapshots.append(
                {
                    "id": stock["id"],
                    "symbol": stock["display_symbol"],
                    "ticker": stock["ticker"],
                    "market": stock["market"],
                    "close": float(closes.iloc[-1]),
                    "date": closes.index[-1].strftime("%Y-%m-%d"),
                }
            )
        except Exception as error:
            print(f"[price snapshot error] {stock.get('ticker')}: {error}", flush=True)
    return snapshots


def _snapshot_by_symbol(snapshots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_symbol: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        for symbol in (snapshot.get("symbol"), snapshot.get("ticker")):
            if symbol:
                by_symbol[str(symbol).upper()] = snapshot
        ticker_root = str(snapshot.get("ticker", "")).split(".", 1)[0]
        if ticker_root:
            by_symbol[ticker_root.upper()] = snapshot
    return by_symbol


def evaluate_previous_brief(
    previous: dict[str, Any] | None,
    current_prices: list[dict[str, Any]],
) -> dict[str, Any]:
    if not previous:
        empty = {"status": "waiting_for_next_session", "evaluated": 0, "correct": 0, "items": []}
        return {"cues": empty, "lstm": empty}

    old_prices = _snapshot_by_symbol(previous.get("prices", []))
    new_prices = _snapshot_by_symbol(current_prices)

    def score_predictions(predictions: list[dict[str, Any]], use_ticker: bool) -> dict[str, Any]:
        items = []
        for prediction in predictions:
            symbol = str(
                prediction.get("symbol") if use_ticker else prediction.get("symbol")
            ).upper()
            if not use_ticker:
                old = next(
                    (item for item in previous.get("prices", []) if item.get("id") == prediction.get("security_id")),
                    None,
                )
                new = next(
                    (item for item in current_prices if item.get("id") == prediction.get("security_id")),
                    None,
                )
            else:
                old = old_prices.get(symbol)
                new = new_prices.get(symbol)
            if not old or not new or new["date"] <= old["date"]:
                continue
            actual = "up" if new["close"] > old["close"] else "down" if new["close"] < old["close"] else "flat"
            expected = prediction.get("direction")
            items.append(
                {
                    "symbol": symbol,
                    "expected": expected,
                    "actual": actual,
                    "from_date": old["date"],
                    "to_date": new["date"],
                    "correct": actual == expected,
                }
            )
        correct = sum(1 for item in items if item["correct"])
        return {
            "status": "evaluated" if items else "waiting_for_next_session",
            "evaluated": len(items),
            "correct": correct,
            "items": items,
        }

    cue_scores = score_predictions(
        previous.get("impact_analysis", {}).get("directions", []), True
    )
    lstm_scores = score_predictions(previous.get("lstm", {}).get("predictions", []), False)
    return {"cues": cue_scores, "lstm": lstm_scores}


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
    previous_evaluation = evaluate_previous_brief(current, prices) if is_new_report_day else None
    current_evaluation = (
        (current or {}).get("evaluation")
        if not is_new_report_day
        else {"status": "waiting_for_next_session", "evaluated": 0, "correct": 0, "items": []}
    ) or {"status": "waiting_for_next_session", "evaluated": 0, "correct": 0, "items": []}
    if current and is_new_report_day:
        archived = dict(current)
        archived.pop("_id", None)
        archived["evaluation"] = previous_evaluation
        database.brief_history.replace_one(
            {"report_date": archived["report_date"]}, archived, upsert=True
        )

    report_date_briefs = list(database.brief_history.find({}, {"_id": 0}).sort("report_date", 1))
    from stock_news.lstm import predict_next_directions

    lstm_report = {
        "report_date": report_date,
        "prices": prices,
        "impact_analysis": impact_analysis,
    }
    lstm_result = predict_next_directions(report_date_briefs, lstm_report)
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
        "prices": prices,
        "lstm": lstm_result,
        "evaluation": current_evaluation,
        "collection": {"saved": saved, "skipped": skipped, "failed": failed},
    }
    database.current_brief.replace_one({"_id": "current"}, brief, upsert=True)
    database.brief_history.create_index("report_date", unique=True)
    return brief


def validation_summary(database: Any) -> dict[str, Any]:
    history = list(database.brief_history.find({}, {"_id": 0, "evaluation": 1}))
    current = database.current_brief.find_one(
        {"_id": "current"},
        {"_id": 0, "report_date": 1, "lstm": 1, "evaluation": 1},
    )
    briefs = history + ([current] if current else [])
    evaluations = [item.get("evaluation") or {} for item in briefs]
    total = sum((item.get("lstm") or {}).get("evaluated", 0) for item in evaluations)
    correct = sum((item.get("lstm") or {}).get("correct", 0) for item in evaluations)
    cue_total = sum((item.get("cues") or {}).get("evaluated", 0) for item in evaluations)
    cue_correct = sum((item.get("cues") or {}).get("correct", 0) for item in evaluations)
    dates = database.brief_history.distinct("report_date")
    if current and current.get("report_date"):
        dates.append(current["report_date"])
    return {
        "days_running": len(set(dates)),
        "evaluated_calls": total,
        "correct_calls": correct,
        "accuracy_percent": round(correct * 100 / total, 1) if total else None,
        "cue_accuracy_percent": round(cue_correct * 100 / cue_total, 1) if cue_total else None,
        "cue_evaluated_calls": cue_total,
        "cue_correct_calls": cue_correct,
        "status": "available" if total else "collecting_history",
        "metric": "next_trading_day_close_direction",
        "lstm_status": (
            (current or {}).get("lstm", {}).get("status", "warming_up")
            if current
            else "warming_up"
        ),
        "lstm_training_days": (
            (current or {}).get("lstm", {}).get("training_days", 0)
            if current
            else 0
        ),
        "lstm_required_days": 30,
    }
