FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps + pip install + cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
 && pip install --upgrade pip pip-tools \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.in .
RUN pip-compile requirements.in && pip install -r requirements.txt \
 && apt-get purge -y build-essential git curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* ~/.cache/pip

COPY . .

CMD ["python", "main.py"]
