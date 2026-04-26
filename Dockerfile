FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY backend/ ./backend/
COPY data/ ./data/

# Generate demo data on build
WORKDIR /app/backend
RUN python generate_data.py

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]
