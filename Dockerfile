# 1. Choose a lightweight official Python image
FROM python:3.10-slim

# 2. Set environment variables
# No .pyc files
ENV PYTHONDONTWRITEBYTECODE=1  
# Instant log output
ENV PYTHONUNBUFFERED=1         

# 3. Set working directory inside the container
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pip-tools (for dependency resolution)
RUN pip install --upgrade pip \
    && pip install pip-tools

# Copy the requirements file into the container
COPY requirements.in .
RUN pip-compile requirements.in --output-file=requirements.txt \
    && pip install -r requirements.txt


# 7. Compile and install dependencies safely
# This ensures conflicts are caught if packages are incompatible
RUN pip-compile requirements.txt --output-file=requirements-locked.txt \
    && pip install -r requirements-locked.txt
