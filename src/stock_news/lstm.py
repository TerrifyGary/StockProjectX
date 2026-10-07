from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from stock_news.trading_sessions import next_session, session_close, session_pair, utc

SEQUENCE_LENGTH = 5
MIN_TRAINING_DAYS = 30
MIN_TRAINING_SAMPLES = 40
ABSTENTION_THRESHOLD = 0.60
MODEL_VERSION = "per_security_lstm_v2"


def _symbol_sentiment(brief: dict[str, Any], snapshot: dict[str, Any]) -> float:
    aliases = {str(snapshot.get(key, "")).upper() for key in ("symbol", "ticker")}
    aliases.add(str(snapshot.get("ticker", "")).split(".")[0].upper())
    for item in brief.get("impact_analysis", {}).get("directions", []):
        if str(item.get("symbol", "")).upper() in aliases:
            direction = item.get("direction")
            if direction in {"up", "down"}:
                score = float(item.get("score", 0))
                return max(0.0, min(1.0, score)) * (1 if direction == "up" else -1) if math.isfinite(score) else 0.0
    return 0.0


def _report_features(briefs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """First issued, completed session per stock; calendar/report duplicates add no data."""
    per_security: dict[str, list[dict[str, Any]]] = {}
    for brief in sorted(briefs, key=lambda b: utc(b["created_at"])):
        issued_at = utc(brief["created_at"])
        for snapshot in brief.get("prices", []):
            if not snapshot.get("id") or not snapshot.get("date"):
                continue
            try:
                expected = session_pair(snapshot, issued_at)["reference_session"]
            except ValueError:
                continue
            if snapshot["date"] != expected or float(snapshot.get("split", 0) or 0):
                continue
            records = per_security.setdefault(snapshot["id"], [])
            if records and snapshot["date"] <= records[-1]["price_date"]:
                continue
            close = float(snapshot["close"])
            if close <= 0 or not math.isfinite(close):
                continue
            previous = records[-1]["close"] if records else close
            return_gap = bool(records and next_session(snapshot, records[-1]["price_date"]) != snapshot["date"])
            change = 0.0 if return_gap else close / previous - 1
            records.append({
                "feature": [max(-0.15, min(0.15, change)), _symbol_sentiment(brief, snapshot)],
                "close": close, "price_date": snapshot["date"], "issued_at": issued_at,
                "snapshot": snapshot,
                "return_gap": return_gap,
            })
    return per_security


def _training_samples(records: list[dict[str, Any]], cutoff: datetime) -> tuple[list, list, datetime | None]:
    train_x, train_y = [], []
    last_label_close = None
    for end in range(SEQUENCE_LENGTH - 1, len(records) - 1):
        sequence = records[end - SEQUENCE_LENGTH + 1:end + 2]
        stock = records[end]["snapshot"]
        if any(row.get("return_gap") for row in sequence[:-1]):
            continue
        if any(next_session(stock, left["price_date"]) != right["price_date"] for left, right in zip(sequence, sequence[1:])):
            continue  # Missing sessions do not become next-session labels or five-session sequences.
        target_close = session_close(stock, records[end + 1]["price_date"])
        if records[end]["issued_at"] >= target_close or target_close > cutoff:
            continue
        movement = records[end + 1]["close"] - records[end]["close"]
        if movement == 0:
            continue
        train_x.append([row["feature"] for row in sequence[:-1]])
        train_y.append(int(movement > 0))
        last_label_close = target_close
    return train_x, train_y, last_label_close


def _fit_probability(train_x: list, train_y: list, recent: list) -> float:
    import torch
    from torch import nn

    class DirectionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=2, hidden_size=12, batch_first=True)
            self.output = nn.Linear(12, 2)

        def forward(self, values: Any) -> Any:
            sequence, _ = self.lstm(values)
            return self.output(sequence[:, -1, :])

    torch.manual_seed(7)
    model = DirectionModel()
    model.train()
    x_tensor = torch.tensor(train_x, dtype=torch.float32)
    y_tensor = torch.tensor(train_y, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(35):
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        probabilities = model(torch.tensor([recent], dtype=torch.float32)).softmax(dim=-1)[0]
    return float(probabilities[1].item())


def predict_next_directions(history_briefs: list[dict[str, Any]], current_brief: dict[str, Any]) -> dict[str, Any]:
    """Train separately per stock with only outcomes available before actual issuance."""
    cutoff = utc(current_brief.get("created_at") or datetime.now(timezone.utc))
    briefs = [b for b in history_briefs if b.get("created_at") and utc(b["created_at"]) < cutoff]
    briefs.append({**current_brief, "created_at": cutoff})
    per_security = _report_features(briefs)
    current = {p["id"]: p for p in current_brief.get("prices", [])}
    predictions, coverage = [], []
    for security_id in sorted(set(per_security) | set(current)):
        records = per_security.get(security_id, [])
        x, y, last_label_close = _training_samples(records, cutoff)
        row = {
            "security_id": security_id, "symbol": current.get(security_id, {}).get("symbol"),
            "observations": len(records), "training_samples": len(y),
            "required_observations": MIN_TRAINING_DAYS, "required_samples": MIN_TRAINING_SAMPLES,
            "status": "warming_up",
        }
        coverage.append(row)
        if len(records) < MIN_TRAINING_DAYS or len(y) < MIN_TRAINING_SAMPLES or len(set(y)) < 2:
            continue
        if security_id in current:
            try:
                reference_session = session_pair(current[security_id], cutoff)["reference_session"]
            except ValueError:
                reference_session = None
            if current[security_id].get("date") != reference_session:
                row["status"] = "stale_reference_price"
                continue
        if security_id not in current or records[-1]["price_date"] != current[security_id].get("date"):
            row["status"] = "missing_current_price"
            continue
        recent_records = records[-SEQUENCE_LENGTH:]
        if any(row.get("return_gap") for row in recent_records) or any(next_session(records[-1]["snapshot"], a["price_date"]) != b["price_date"] for a, b in zip(recent_records, recent_records[1:])):
            row["status"] = "missing_sessions"
            continue
        recent = [record["feature"] for record in recent_records]
        try:
            probability_up = _fit_probability(x, y, recent)
            if not math.isfinite(probability_up) or not 0 <= probability_up <= 1:
                raise ValueError("Invalid LSTM probability")
        except Exception as error:
            row["status"] = "training_failed"
            row["error"] = f"{type(error).__name__}: {error}"
            continue
        confidence = max(probability_up, 1 - probability_up)
        direction = ("up" if probability_up >= 0.5 else "down") if confidence >= ABSTENTION_THRESHOLD else "uncertain"
        row["status"] = "available"
        predictions.append({
            "security_id": security_id, "symbol": current[security_id].get("symbol"),
            "direction": direction, "score": round(confidence, 4),
            "probability_up": round(probability_up, 6), "confidence_calibrated": False,
            "model_version": MODEL_VERSION, "input_features": recent,
            "training_cutoff": last_label_close, "training_samples": len(y),
        })
    return {
        "status": "available" if predictions else "unavailable" if any(row["status"] == "training_failed" for row in coverage) else "warming_up",
        "training_days": min((row["observations"] for row in coverage), default=0),
        "required_days": MIN_TRAINING_DAYS, "required_samples": MIN_TRAINING_SAMPLES,
        "sequence_length": SEQUENCE_LENGTH, "model_version": MODEL_VERSION,
        "abstention_threshold": ABSTENTION_THRESHOLD, "confidence_calibrated": False,
        "per_security": coverage, "predictions": predictions,
        "training_parameters": {"seed": 7, "epochs": 35, "hidden_size": 12, "learning_rate": 0.01, "sequence_length": SEQUENCE_LENGTH},
    }
