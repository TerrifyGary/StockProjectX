from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

from stock_news.evidence_brief import build_evidence_brief


@dataclass
class LocalNewsImpactModel:
    """Local instruction model that turns saved news into cited impact cues."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device_choice: str = "cpu"
    max_input_tokens: int = 8192
    max_new_tokens: int = 900

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = self._choose_device(torch, self.device_choice)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _choose_device(torch: Any, choice: str) -> str:
        if choice == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if choice == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NEWS_ANALYSIS_MODEL_DEVICE=cuda, but CUDA is unavailable")
        if choice == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("NEWS_ANALYSIS_MODEL_DEVICE=mps, but Apple Metal is unavailable")
        if choice not in {"cpu", "cuda", "mps"}:
            raise ValueError("NEWS_ANALYSIS_MODEL_DEVICE must be auto, cpu, cuda, or mps")
        return choice

    def _generate_json(self, instruction: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cautious financial news analyst. Use only the supplied news. "
                    "Do not invent facts, sources, or certainty. Return one valid JSON object only."
                ),
            },
            {"role": "user", "content": instruction},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The local impact model did not return a JSON object")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("The local impact model response must be a JSON object")
        return parsed

    def analyze(
        self,
        articles: list[dict[str, Any]],
        companies: list[dict[str, Any]],
        reference_time: datetime | None = None,
        prepared_evidence_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        company_rows = []
        allowed_symbols: dict[str, str] = {}
        for company in companies:
            tickers = [str(value) for value in company.get("tickers", [])]
            allowed_symbols.update({ticker.casefold(): ticker for ticker in tickers})
            company_rows.append(
                {"name": company["name"], "aliases": company.get("aliases", []), "symbols": tickers}
            )

        evidence_brief = prepared_evidence_brief or build_evidence_brief(
            articles, reference_time=reference_time
        )
        score_by_article = {
            item["article_id"]: item for item in evidence_brief["article_scores"]
        }
        indexed_articles = []
        for index, article in enumerate(articles):
            evidence_score = score_by_article[index]
            indexed_articles.append(
                {
                    "id": index,
                    "title": article.get("display_title") or article.get("title") or "",
                    "summary": article.get("display_summary") or "",
                    "topics": article.get("topics", []),
                    "published_at": (
                        article["published_at"].isoformat()
                        if hasattr(article.get("published_at"), "isoformat")
                        else article.get("published_at")
                    ),
                    "source": article.get("source", {}).get("name"),
                    "companies_already_matched": [
                        entity.get("name") for entity in article.get("entities", [])
                    ],
                    "evidence_priority_score": evidence_score["priority_score"],
                    "evidence_score_components": evidence_score["components"],
                    "url": article.get("url"),
                }
            )

        cues: list[dict[str, Any]] = []
        for offset in range(0, len(indexed_articles), 8):
            batch = indexed_articles[offset : offset + 8]
            instruction = (
                "Assess each listed story for plausible, evidence-based short-term effects on the "
                "provided watchlist. Global macro or geopolitical stories may affect a company, "
                "but connect them only when the article supports a clear transmission channel. "
                "Do not turn generic positive/negative wording into a stock call. Omit unrelated "
                "stories. The evidence priority score is a rule-based ranking aid using relevance, "
                "source traceability, freshness, and duplicate-title count. It is not a probability, "
                "direction, or source credibility rating; do not infer upside/downside from the score. "
                "Return {\"cues\":[{\"article_id\":integer,\"direction\":\"upside\"|\"downside\","
                "\"symbols\":[string],\"reason\":string,\"confidence\":number}]} where confidence "
                "is your separate 0..1 confidence in the evidence-based cue, article_id must be from "
                "the supplied stories, and symbols must come from the watchlist. Use concise reasons "
                "grounded in the article. Preserve mixed evidence instead of forcing a net call.\n"
                "PRE-ANALYSIS EVIDENCE BRIEF:\n"
                + evidence_brief["summary"]
                + "\nWATCHLIST:\n"
                + json.dumps(company_rows, ensure_ascii=False)
                + "\nSTORIES:\n"
                + json.dumps(batch, ensure_ascii=False)
            )
            parsed = self._generate_json(instruction)
            batch_ids = {item["id"] for item in batch}
            for cue in parsed.get("cues", []):
                if not isinstance(cue, dict):
                    continue
                try:
                    article_index = int(cue.get("article_id"))
                except (TypeError, ValueError):
                    continue
                if article_index not in batch_ids:
                    continue
                direction = cue.get("direction")
                if direction not in {"upside", "downside"}:
                    continue
                symbols = sorted(
                    {
                        allowed_symbols[str(symbol).casefold()]
                        for symbol in cue.get("symbols", [])
                        if str(symbol).casefold() in allowed_symbols
                    }
                )
                reason = str(cue.get("reason", "")).strip()
                if not symbols or not reason:
                    continue
                try:
                    confidence = max(0.0, min(1.0, float(cue.get("confidence", 0.0))))
                except (TypeError, ValueError):
                    confidence = 0.0
                article = articles[article_index]
                cues.append(
                    {
                        "title": article.get("display_title") or article.get("title"),
                        "summary": article.get("display_summary", ""),
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("name"),
                        "entities": [item.get("name") for item in article.get("entities", [])],
                        "topics": article.get("topics", []),
                        "article_id": article_index,
                        "symbols": symbols,
                        "direction": direction,
                        "reason": reason[:500],
                        "score": round(confidence, 4),
                        "evidence_priority_score": score_by_article[article_index]["priority_score"],
                        "rank_score": round(
                            confidence * score_by_article[article_index]["priority_score"] / 100,
                            4,
                        ),
                        "sentiment": direction,
                    }
                )

        upside = sorted(
            [cue for cue in cues if cue["direction"] == "upside"],
            key=lambda cue: cue["rank_score"],
            reverse=True,
        )
        downside = sorted(
            [cue for cue in cues if cue["direction"] == "downside"],
            key=lambda cue: cue["rank_score"],
            reverse=True,
        )
        votes: dict[str, dict[str, float]] = {}
        for cue in cues:
            for symbol in cue["symbols"]:
                ballot = votes.setdefault(symbol, {"upside": 0.0, "downside": 0.0})
                ballot[cue["direction"]] += cue["rank_score"]
        directions = []
        for symbol, ballot in sorted(votes.items()):
            margin = ballot["upside"] - ballot["downside"]
            if margin:
                directions.append(
                    {
                        "symbol": symbol,
                        "direction": "up" if margin > 0 else "down",
                        "score": round(abs(margin) / sum(ballot.values()), 4),
                        "signal_count": sum(1 for cue in cues if symbol in cue["symbols"]),
                    }
                )

        cue_summary = [
            {
                "direction": cue["direction"],
                "symbols": cue["symbols"],
                "reason": cue["reason"],
                "evidence_priority_score": cue["evidence_priority_score"],
            }
            for cue in cues
        ]
        note_prompt = (
            "Write one concise daily investor note from these model-reviewed news cues. Mention "
            "the main risks and possible supports, avoid unsupported price targets, and state "
            "when evidence is mixed. Return JSON {\"investor_note\":string}. This is research "
            "context, not personalized financial advice. The evidence priority score only ranks "
            "traceable, relevant, fresh, and less-duplicated stories; it is not a forecast.\n"
            "PRE-ANALYSIS EVIDENCE BRIEF:\n"
            + evidence_brief["summary"]
            + "\nCUES:\n"
            + json.dumps(cue_summary, ensure_ascii=False)
        )
        note_data = self._generate_json(note_prompt, max_new_tokens=220)
        investor_note = str(note_data.get("investor_note", "")).strip()
        if not investor_note:
            raise ValueError("The local impact model returned an empty investor note")
        return {
            "upside": upside[:5],
            "downside": downside[:5],
            "neutral": [],
            "directions": directions,
            "investor_note": investor_note,
            "analyzed_articles": len(articles),
            "evidence_brief": evidence_brief,
        }


@lru_cache(maxsize=2)
def get_local_news_impact_model(model_name: str, device_choice: str) -> LocalNewsImpactModel:
    return LocalNewsImpactModel(model_name=model_name, device_choice=device_choice)
