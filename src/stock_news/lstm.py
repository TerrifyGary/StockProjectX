from __future__ import annotations

from typing import Any

SEQUENCE_LENGTH = 5
MIN_TRAINING_DAYS = 30


def _symbol_sentiment(brief: dict[str, Any], snapshot: dict[str, Any]) -> float:
    aliases = {
        str(value).upper()
        for value in (
            snapshot.get("symbol"),
            snapshot.get("ticker"),
            str(snapshot.get("ticker", "")).split(".", 1)[0],
        )
        if value
    }
    for item in brief.get("impact_analysis", {}).get("directions", []):
        if str(item.get("symbol", "")).upper() in aliases:
            return float(item.get("score", 0)) * (1 if item.get("direction") == "up" else -1)
    return 0.0


def _report_features(briefs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    per_security: dict[str, list[dict[str, Any]]] = {}
    previous_closes: dict[str, float] = {}
    for brief in briefs:
        current = {item["id"]: item for item in brief.get("prices", []) if item.get("id")}
        for security_id, snapshot in current.items():
            close = float(snapshot["close"])
            previous = previous_closes.get(security_id)
            change = (close - previous) / previous if previous else 0.0
            previous_closes[security_id] = close
            per_security.setdefault(security_id, []).append(
                {
                    "feature": [max(-0.15, min(0.15, change)), _symbol_sentiment(brief, snapshot)],
                    "close": close,
                    "price_date": snapshot.get("date", ""),
                    "snapshot": snapshot,
                }
            )
    return per_security


def predict_next_directions(
    history_briefs: list[dict[str, Any]], current_brief: dict[str, Any]
) -> dict[str, Any]:
    """Train a small walk-forward LSTM once 30 prior daily reports exist."""
    import torch
    from torch import nn

    briefs = sorted(
        history_briefs + [current_brief], key=lambda item: item.get("report_date", "")
    )
    unique_days = {item.get("report_date") for item in briefs if item.get("report_date")}
    if len(unique_days) < MIN_TRAINING_DAYS:
        return {
            "status": "warming_up",
            "training_days": len(unique_days),
            "required_days": MIN_TRAINING_DAYS,
            "predictions": [],
        }

    per_security = _report_features(briefs)
    train_x: list[list[list[float]]] = []
    train_y: list[int] = []
    for records in per_security.values():
        for end in range(SEQUENCE_LENGTH - 1, len(records) - 1):
            start = end - SEQUENCE_LENGTH + 1
            if records[end + 1]["price_date"] <= records[end]["price_date"]:
                continue
            movement = records[end + 1]["close"] - records[end]["close"]
            if movement == 0:
                continue
            train_x.append([record["feature"] for record in records[start : end + 1]])
            train_y.append(1 if movement > 0 else 0)

    if len(train_y) < 40:
        return {
            "status": "warming_up",
            "training_days": len(unique_days),
            "required_days": MIN_TRAINING_DAYS,
            "training_samples": len(train_y),
            "predictions": [],
        }

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
    predictions = []
    with torch.inference_mode():
        for security_id, records in per_security.items():
            if len(records) < SEQUENCE_LENGTH:
                continue
            stock = records[-1]["snapshot"]
            recent = [record["feature"] for record in records[-SEQUENCE_LENGTH:]]
            probabilities = model(torch.tensor([recent], dtype=torch.float32)).softmax(dim=-1)[0]
            predicted_class = int(torch.argmax(probabilities).item())
            predictions.append(
                {
                    "security_id": security_id,
                    "symbol": stock.get("symbol"),
                    "direction": "up" if predicted_class == 1 else "down",
                    "score": round(float(probabilities[predicted_class].item()), 4),
                }
            )
    return {
        "status": "available" if predictions else "warming_up",
        "training_days": len(unique_days),
        "required_days": MIN_TRAINING_DAYS,
        "training_samples": len(train_y),
        "sequence_length": SEQUENCE_LENGTH,
        "predictions": predictions,
    }
