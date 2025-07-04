# 1. Use lightweight image
FROM python:3.10-slim

# 2. Env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Working dir
WORKDIR /app

# 4. Install only needed system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Upgrade pip & install pip-tools
RUN pip install --upgrade pip \
    && pip install pip-tools

# 6. Copy project
COPY requirements.in .

# 7. Compile and install deps
RUN pip-compile requirements.in \
    pip install --no-deps -r requirements.txt

# 8. Clean up (only after all installs)
RUN apt-get purge -y build-essential git curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* ~/.cache/pip

# 9. Copy source code
COPY . .
