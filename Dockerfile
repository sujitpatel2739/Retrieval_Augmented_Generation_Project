# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for building some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ src/
# Optional: Render usually sets env vars via dashboard, not .env
COPY .env .

# Expose FastAPI port
EXPOSE 8000

# Run using Uvicorn (faster startup, cleaner logging)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
