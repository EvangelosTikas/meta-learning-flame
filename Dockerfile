# 1. Use a lightweight Python image
FROM python:3.10-slim

# 2. Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Working directory
WORKDIR /app

# 4. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Upgrade pip and install pip-tools
RUN pip install --upgrade pip \
    && pip install pip-tools

# 6. Copy project files (includes pyproject.toml, requirements.in, source code)

COPY . /app

# 7. Compile and install dependencies
RUN pip-compile ./requirements.in --output-file=requirements.txt
RUN pip install -r ./requirements.txt
