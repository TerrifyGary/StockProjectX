from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LocalMarianTranslator:
    """Translate supported Chinese and Japanese text into English locally."""

    device_choice: str = "auto"
    model_by_language: dict[str, str] = field(default_factory=dict)
    max_input_tokens: int = 512
    max_output_tokens: int = 192

    provider: str = "huggingface-local"

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self.device = self._choose_device(torch, self.device_choice)
        self._loaded_models: dict[str, tuple[Any, Any]] = {}

    @staticmethod
    def _choose_device(torch: object, choice: str) -> str:
        if choice != "auto":
            if choice == "cuda" and not torch.cuda.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("NEWS_MODEL_DEVICE=cuda, but CUDA is unavailable")
            if choice == "mps" and not torch.backends.mps.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("NEWS_MODEL_DEVICE=mps, but Apple Metal is unavailable")
            return choice
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"

    def model_for_language(self, language: str) -> str | None:
        if language == "en":
            return None
        try:
            return self.model_by_language[language]
        except KeyError as error:
            raise ValueError(f"No English translation model configured for {language!r}") from error

    def _load(self, language: str) -> tuple[Any, Any]:
        model_name = self.model_for_language(language)
        if model_name is None:
            raise ValueError("English text does not require a translation model")
        if model_name not in self._loaded_models:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            self._loaded_models[model_name] = (tokenizer, model)
        return self._loaded_models[model_name]

    def translate_many(self, texts: list[str], source_language: str) -> list[str]:
        if source_language == "en":
            return texts
        cleaned = [" ".join(text.split()) for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Cannot translate empty text")

        tokenizer, model = self._load(source_language)
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                num_beams=4,
            )
        outputs = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        translated = [text.strip() for text in outputs]
        if len(translated) != len(texts) or any(not text for text in translated):
            raise RuntimeError("The local translation model returned an empty translation")
        return translated


def backfill_english_translations(
    collection: Any, translator: LocalMarianTranslator
) -> tuple[int, int, int]:
    """Translate missing title/summary fields on existing MongoDB records."""
    missing = {"$in": [None, ""]}
    query = {
        "$or": [
            {"title_en": missing},
            {"summary.text_en": missing},
        ]
    }
    updated = skipped = failed = 0
    for article in collection.find(query):
        language = article.get("language", "en")
        summary = article.get("summary", {})
        fields: list[tuple[str, str]] = []
        title = article.get("title", "")
        summary_text = summary.get("text", "")
        if not article.get("title_en") and title:
            fields.append(("title_en", title))
        if not summary.get("text_en") and summary_text:
            fields.append(("summary.text_en", summary_text))
        if not fields:
            skipped += 1
            continue

        try:
            translated = (
                [text for _, text in fields]
                if language == "en"
                else translator.translate_many([text for _, text in fields], language)
            )
            update_fields = {
                field_name: value
                for (field_name, _), value in zip(fields, translated)
            }
            model_name = translator.model_by_language.get(language)
            update_fields.update(
                {
                    "translation.target_language": "en",
                    "translation.provider": translator.provider,
                    "translation.model": model_name,
                    "translation.status": "not_needed" if language == "en" else "complete",
                    "translation.generated_at": datetime.now(timezone.utc),
                    "translation.error": None,
                }
            )
            collection.update_one(
                {"_id": article["_id"]},
                {"$set": update_fields},
            )
            updated += 1
            print(f"[translated] {title or article.get('url', article['_id'])}")
        except Exception as error:
            model_name = translator.model_by_language.get(language)
            collection.update_one(
                {"_id": article["_id"]},
                {
                    "$set": {
                        "translation.target_language": "en",
                        "translation.provider": translator.provider,
                        "translation.model": model_name,
                        "translation.status": "failed",
                        "translation.error": f"{type(error).__name__}: {error}"[:500],
                        "translation.generated_at": datetime.now(timezone.utc),
                    }
                },
            )
            failed += 1
            print(f"[translation error] {title or article.get('url', article['_id'])}: {error}")
    return updated, skipped, failed
