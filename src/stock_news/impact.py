from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass
class FinancialSentimentAnalyzer:
    """Run local FinBERT sentiment classification on English story text."""

    model_name: str = "ProsusAI/finbert"
    device_choice: str = "cpu"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = self._choose_device(torch, self.device_choice)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, low_cpu_mem_usage=False
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _choose_device(torch: Any, choice: str) -> str:
        if choice != "auto":
            if choice == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("NEWS_IMPACT_MODEL_DEVICE=cuda, but CUDA is unavailable")
            if choice == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("NEWS_IMPACT_MODEL_DEVICE=mps, but Apple Metal is unavailable")
            if choice not in {"cpu", "cuda", "mps"}:
                raise ValueError("NEWS_IMPACT_MODEL_DEVICE must be auto, cpu, cuda, or mps")
            return choice
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def analyze(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            " ".join(
                part.strip()
                for part in (
                    article.get("display_title") or article.get("title") or "",
                    article.get("display_summary") or "",
                )
                if part and part.strip()
            )
            for article in articles
        ]
        nonempty = [(index, text) for index, text in enumerate(texts) if text]
        if not nonempty:
            return {"upside": [], "downside": [], "neutral": []}

        inputs = self.tokenizer(
            [text for _, text in nonempty],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1).cpu().tolist()

        result: dict[str, list[dict[str, Any]]] = {
            "upside": [],
            "downside": [],
            "neutral": [],
        }
        direction_votes: dict[str, dict[str, float]] = {}
        for (article_index, _), scores in zip(nonempty, probabilities, strict=True):
            article = articles[article_index]
            labels = self.model.config.id2label
            label = str(labels[max(range(len(scores)), key=scores.__getitem__)]).lower()
            if label.startswith("label_"):
                label = {0: "positive", 1: "negative", 2: "neutral"}.get(
                    max(range(len(scores)), key=scores.__getitem__), "neutral"
                )
            if label not in {"positive", "negative", "neutral"}:
                label = "neutral"
            score = max(scores)
            group = {"positive": "upside", "negative": "downside"}.get(label, "neutral")

            entities = article.get("entities", [])
            symbols = sorted(
                {
                    ticker
                    for entity in entities
                    for ticker in entity.get("tickers", [])
                }
            )
            if label in {"positive", "negative"}:
                for symbol in symbols:
                    tally = direction_votes.setdefault(
                        symbol,
                        {"positive_score": 0.0, "negative_score": 0.0, "signal_count": 0},
                    )
                    tally[f"{group}_score"] += float(score)
                    tally["signal_count"] += 1
            result[group].append(
                {
                    "title": article.get("display_title") or article.get("title"),
                    "summary": article.get("display_summary") or "",
                    "url": article.get("url"),
                    "source": article.get("source", {}).get("name"),
                    "entities": [entity.get("name") for entity in entities],
                    "symbols": symbols,
                    "sentiment": label,
                    "score": round(float(score), 4),
                }
            )

        result["upside"].sort(key=lambda item: item["score"], reverse=True)
        result["downside"].sort(key=lambda item: item["score"], reverse=True)
        result["neutral"].sort(key=lambda item: item["score"], reverse=True)
        directions = []
        for symbol, tally in sorted(direction_votes.items()):
            difference = tally["positive_score"] - tally["negative_score"]
            if difference:
                directions.append({
                    "symbol": symbol,
                    "direction": "up" if difference > 0 else "down",
                    "score": round(
                        abs(difference)
                        / (tally["positive_score"] + tally["negative_score"]),
                        4,
                    ),
                    "signal_count": tally["signal_count"],
                })
        return {
            key: values if key == "neutral" else values[:3]
            for key, values in result.items()
        } | {"directions": directions}


@lru_cache(maxsize=4)
def get_sentiment_analyzer(
    model_name: str, device_choice: str
) -> FinancialSentimentAnalyzer:
    """Load one shared model instance per model/device configuration."""
    return FinancialSentimentAnalyzer(model_name, device_choice)
