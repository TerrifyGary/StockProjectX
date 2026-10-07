FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY requirements.txt pyproject.toml ./

RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.10.0+cpu" \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir "transformers>=4.45,<5" "sentencepiece>=0.2,<1" "protobuf>=4.25,<7"

COPY src ./src

RUN python -m pip install --no-cache-dir --no-deps .

COPY config ./config

CMD ["stock-news"]
