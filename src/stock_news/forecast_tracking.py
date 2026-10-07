"""Immutable issued calls and idempotent evaluation of exact target-session closes."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Any

from stock_news.trading_sessions import session_pair, session_close, utc

TRACKING_VERSION = "session_forecast_v1"


def pending_price_start_dates(database: Any) -> dict[str, str]:
    dates: dict[str, str] = {}
    for record in database.prediction_records.find({"status": "pending"}):
        key = record["security_id"]
        date = record["reference_session"]
        dates[key] = min(date, dates.get(key, date))
    # Recover a process interrupted between freezing a report and registering its calls.
    for brief in database.forecast_briefs.find({"ledger_registered_at": {"$exists": False}}):
        for stock in brief.get("prices", []):
            if stock.get("id") and stock.get("date"):
                key, date = stock["id"], stock["date"]
                dates[key] = min(date, dates.get(key, date))
    return dates


def _news_direction(brief: dict[str, Any], stock: dict[str, Any]) -> str:
    aliases = {str(stock.get(key, "")).upper() for key in ("symbol", "ticker")}
    aliases.add(str(stock.get("ticker", "")).split(".")[0].upper())
    for item in brief.get("impact_analysis", {}).get("directions", []):
        if str(item.get("symbol", "")).upper() in aliases and item.get("direction") in {"up", "down"}:
            return item["direction"]
    return "uncertain"


def register_forecasts(database: Any, frozen: dict[str, Any]) -> dict[str, Any]:
    """First call for a security/model/target wins, including across weekend reruns."""
    database.prediction_records.create_index([("status", 1), ("target_close_at", 1)])
    registration: dict[str, Any] = {"inserted": 0, "existing": 0, "rejected": []}
    lstm_by_id = {p["security_id"]: p for p in frozen.get("lstm", {}).get("predictions", [])}
    for stock in frozen.get("prices", []):
        try:
            if not math.isfinite(float(stock["close"])) or float(stock["close"]) <= 0:
                raise ValueError("Invalid reference close")
            sessions = session_pair(stock, utc(frozen["created_at"]))
            if stock.get("date") != sessions["reference_session"]:
                # Never silently forecast against an old or an unfinished reference bar.
                registration["rejected"].append({"security_id": stock["id"], "reason": "stale_or_unfinished_reference", "required_session": sessions["reference_session"]})
                continue
        except (ValueError, KeyError) as error:
            print(f"[forecast session unavailable] {stock.get('ticker')}: {error}", flush=True)
            registration["rejected"].append({"security_id": stock["id"], "reason": str(error)})
            continue
        candidates = {
            "always_up": {"direction": "up", "model_version": "always_up_v1"},
            "previous_direction": {
                "direction": stock.get("previous_direction", "uncertain"),
                "model_version": "previous_session_direction_v1",
                "baseline_inputs": {"previous_close": stock.get("previous_close"), "previous_session": stock.get("previous_session"), "reference_close": stock["close"]},
            },
            "news_only": {"direction": _news_direction(frozen, stock), "model_version": frozen.get("impact_analysis", {}).get("model", "unknown")},
        }
        if stock["id"] in lstm_by_id:
            candidates["lstm"] = lstm_by_id[stock["id"]]
        for model, prediction in candidates.items():
            identifier = f"{stock['id']}|{sessions['target_session']}|{model}"
            record = {
                "_id": identifier,
                "tracking_version": TRACKING_VERSION,
                "origin_report_date": frozen["report_date"],
                "issued_at": frozen["created_at"],
                "security_id": stock["id"],
                "symbol": stock.get("symbol"),
                "ticker": stock.get("ticker"),
                "market": stock.get("market"),
                **sessions,
                "reference_close": float(stock["close"]),
                "reference_split": float(stock.get("split", 0) or 0),
                "model": model,
                "model_version": prediction.get("model_version", "unknown"),
                "direction": prediction.get("direction", "uncertain"),
                "probability_up": prediction.get("probability_up"),
                "confidence": prediction.get("score"),
                "confidence_calibrated": False,
                "input_features": prediction.get("input_features", []),
                "baseline_inputs": prediction.get("baseline_inputs"),
                "training_cutoff": prediction.get("training_cutoff"),
                "training_samples": prediction.get("training_samples"),
                "training_parameters": frozen.get("lstm", {}).get("training_parameters") if model == "lstm" else None,
                "input_article_keys": frozen.get("article_keys", []),
                "input_model": frozen.get("impact_analysis", {}).get("model"),
                "status": "pending",
            }
            result = database.prediction_records.update_one({"_id": identifier}, {"$setOnInsert": record}, upsert=True)
            registration["inserted" if result.upserted_id is not None else "existing"] += 1
    return registration


def register_pending_batches(database: Any) -> None:
    for frozen in database.forecast_briefs.find({"ledger_registered_at": {"$exists": False}}).sort("created_at", 1):
        registration = register_forecasts(database, frozen)
        # Completion metadata may change; original forecast inputs and outputs may not.
        database.forecast_briefs.update_one(
            {"report_date": frozen["report_date"]},
            {"$set": {"ledger_registered_at": datetime.now(timezone.utc), "registration": registration}},
        )


def evaluate_pending_forecasts(
    database: Any, prices: list[dict[str, Any]], *, now: datetime | None = None
) -> int:
    now = utc(now or datetime.now(timezone.utc))
    by_id = {stock["id"]: stock for stock in prices}
    updated = 0
    for record in database.prediction_records.find({"status": "pending", "target_close_at": {"$lte": now}}):
        stock = by_id.get(record["security_id"])
        if not stock:
            continue
        bars = stock.get("history", []) + [stock]
        target = next((bar for bar in bars if bar.get("date") == record["target_session"]), None)
        if not target:
            continue  # A later available close is never a substitute for the target.
        try:
            if session_close(record, record["target_session"]) > now:
                continue
        except ValueError:
            continue
        # A split between reference and target makes a raw-close comparison invalid.
        corporate_action = any(
            float(bar.get("split", 0) or 0) != 0
            for bar in bars
            if record["reference_session"] < bar.get("date", "") <= record["target_session"]
        )
        close = float(target["close"])
        reference = float(record["reference_close"])
        if not math.isfinite(close) or close <= 0:
            continue
        actual = "up" if close > reference else "down" if close < reference else "flat"
        called = record["direction"] in {"up", "down"}
        outcome = {
            "status": "excluded" if corporate_action else "evaluated",
            "exclusion_reason": "stock_split" if corporate_action else None,
            "evaluated_at": now,
            "actual": actual,
            "target_close": close,
            "return_percent": round((close / reference - 1) * 100, 6),
            "made_call": called,
            "correct": actual == record["direction"] if called and not corporate_action else None,
        }
        result = database.prediction_records.update_one(
            {"_id": record["_id"], "status": "pending"}, {"$set": outcome}
        )
        updated += result.modified_count
    return updated


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    mature = [r for r in records if r.get("status") == "evaluated"]
    calls = [r for r in mature if r.get("made_call")]
    correct = sum(bool(r.get("correct")) for r in calls)
    binary = [r for r in mature if r.get("actual") in {"up", "down"} and r.get("probability_up") is not None]
    brier = sum((float(r["probability_up"]) - (r["actual"] == "up")) ** 2 for r in binary)
    bins = []
    for low, high in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        selected = [r for r in binary if low <= max(float(r["probability_up"]), 1 - float(r["probability_up"])) < high]
        bins.append({
            "range": [low, min(high, 1.0)], "count": len(selected),
            "mean_confidence": sum(max(float(r["probability_up"]), 1 - float(r["probability_up"])) for r in selected) / len(selected) if selected else None,
            "observed_accuracy": sum((float(r["probability_up"]) >= 0.5) == (r["actual"] == "up") for r in selected) / len(selected) if selected else None,
        })
    return {
        "issued": len(records), "matured": len(mature), "evaluated_calls": len(calls),
        "correct_calls": correct, "accuracy_percent": round(100 * correct / len(calls), 1) if calls else None,
        "coverage_percent": round(100 * len(calls) / len(mature), 1) if mature else None,
        "uncertain_calls": len(mature) - len(calls),
        "pending_calls": sum(r.get("status") == "pending" for r in records),
        "excluded_calls": sum(r.get("status") == "excluded" for r in records),
        "brier_score": round(brier / len(binary), 4) if binary else None,
        "probability_samples": len(binary), "confidence_calibrated": False,
        "calibration_bins": bins,
        "expected_calibration_error": round(sum(abs(b["mean_confidence"] - b["observed_accuracy"]) * b["count"] for b in bins if b["count"]) / len(binary), 4) if binary else None,
    }


def tracking_summary(database: Any) -> dict[str, Any]:
    records = list(database.prediction_records.find({}, {"_id": 0}))
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_security: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[record["model"]].append(record)
        if record["model"] == "lstm":
            by_security[record["security_id"]].append(record)
    metrics = {model: _metrics(by_model[model]) for model in ("lstm", "always_up", "previous_direction", "news_only")}
    # Baselines use the same security, target session, and issuance batch as made LSTM calls.
    paired_keys = {
        (r["security_id"], r["target_session"], r["origin_report_date"])
        for r in by_model["lstm"] if r.get("status") == "evaluated" and r.get("made_call")
    }
    for model, rows in by_model.items():
        paired = [r for r in rows if (r["security_id"], r["target_session"], r["origin_report_date"]) in paired_keys]
        metrics[model]["paired_with_lstm"] = _metrics(paired)
    return {
        **metrics["lstm"], "models": metrics,
        "per_security": [{"security_id": key, **_metrics(rows)} for key, rows in sorted(by_security.items())],
        "metric": "exact_target_session_close_direction",
        "tracking_version": TRACKING_VERSION,
    }
