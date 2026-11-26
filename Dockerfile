FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    zsh \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY broutegpt.py .
COPY broutegpt_agent/ ./broutegpt_agent/

RUN mkdir -p /workspace /app/chatbot_outputs

ENTRYPOINT ["python3", "/app/broutegpt.py"]
CMD ["--help"]

