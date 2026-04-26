#!/usr/bin/env bash
set -e

echo ""
echo "  ChurnGuard AI — Local Dev Startup"
echo "  ─────────────────────────────────"
echo ""

# 1. Check Python
python3 --version || { echo "Python 3.10+ required"; exit 1; }

# 2. Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

# 3. Generate demo data
echo "Generating demo data..."
cd backend
python3 generate_data.py
cd ..

# 4. Start API
echo ""
echo "Starting API on http://localhost:8002"
echo "Frontend: open frontend.html in your browser"
echo "Swagger:  http://localhost:8002/docs"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd backend
python3 -m uvicorn api:app --host 0.0.0.0 --port 8002 --reload
