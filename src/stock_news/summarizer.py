from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Summarizer(Protocol):
    provider: str
    model_name: str

    def summarize(self, text: str, language: str) -> str: ...


@dataclass
class LocalMTS5Summarizer:
    """Local multilingual abstractive summaries using the mT5 XLSum checkpoint."""

    model_name: str
    device_choice: str = "auto"
    max_input_tokens: int = 512
    max_summary_tokens: int = 110

    provider: str = "huggingface-local"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.device = self._choose_device(torch, self.device_choice)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

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

    def summarize(self, text: str, language: str) -> str:
        del language  # mT5 XLSum generates in the source language without a language prefix.
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("Cannot summarize empty article text")

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_summary_tokens,
                num_beams=4,
                no_repeat_ngram_size=2,
            )
        summary = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
        if not summary:
            raise RuntimeError("The local model returned an empty summary")
        return summary
