# ChurnGuard AI — Customer Retention Platform

Undergraduate thesis project — BRAC University, CSE  
**A Real-Time, Scalable AI Architecture for Cost-Aware Customer Retention**

---

## What this is

A fully working SaaS platform covering **Layer 1** (real-time data ingestion) and **Layer 2**
(dual prediction engine) of the thesis architecture. The system ingests structured CSV data
or live Kafka events, preprocesses them into a feature store, trains XGBoost + MLP models,
and scores every customer with churn probability and SHAP explanations.

---

## File map

```
layer1/
├── backend/
│   ├── api.py                  ← FastAPI — 14 endpoints (Layer 1 + Layer 2)
│   ├── pipeline.py             ← Layer 1: CSV ingestion, feature engineering, Kafka drift
│   ├── prediction_engine.py    ← Layer 2: XGBoost + MLP, SMOTE, SHAP, champion selection
│   ├── generate_data.py        ← Synthetic Telco dataset + Kafka event replay data
│   └── models/
│       ├── xgb_model.pkl       ← Saved XGBoost model
│       ├── mlp_model.pkl       ← Saved MLP model
│       └── meta.json           ← Champion, feature names, training report
├── data/
│   ├── telco_demo.csv          ← 500-row synthetic IBM Telco-style dataset
│   ├── kafka_events.json       ← 50 pre-generated Kafka events for replay
│   └── dashboard_data.json     ← Pre-computed metrics for the dashboard
├── frontend.html               ← Standalone SaaS UI (open in any browser)
├── requirements.txt            ← Exact Python dependency versions
├── Dockerfile                  ← API container
├── docker-compose.yml          ← API + Nginx frontend (one command deploy)
└── start.sh                    ← Local dev startup script
```

---

## Option A — Run locally (fastest, no Docker)

**Requirements:** Python 3.10+, pip

```bash
# 1. Clone / copy the layer1/ folder to your machine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate demo data
cd backend
python generate_data.py

# 4. Start the API
python -m uvicorn api:app --host 0.0.0.0 --port 8002 --reload

# 5. Open the frontend
# Open frontend.html directly in Chrome / Firefox
# (File → Open, or drag into browser — no server needed)
```

Or use the startup script (Mac/Linux):
```bash
chmod +x start.sh && ./start.sh
```

**URLs:**
- Frontend UI:  `frontend.html` (open as file)
- API:          `http://localhost:8002`
- Swagger docs: `http://localhost:8002/docs`
- ReDoc:        `http://localhost:8002/redoc`

---

## Option B — Docker Compose (production-style)

**Requirements:** Docker + Docker Compose

```bash
# Build and start both containers
docker compose up --build

# Or run detached
docker compose up --build -d

# Stop
docker compose down
```

**URLs:**
- Frontend UI:  `http://localhost:3000`
- API:          `http://localhost:8002`
- Swagger docs: `http://localhost:8002/docs`

---

## Option C — Deploy to a cloud VM (Render / Railway / EC2)

### Render (free tier)
1. Push the `layer1/` folder to a GitHub repo
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt && cd backend && python generate_data.py`
4. Start command: `cd backend && uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Render gives you a public URL — update `const API = '...'` in `frontend.html` line 1

### Railway
```bash
railway login
railway init
railway up
```

### Any Ubuntu VPS (EC2, DigitalOcean, etc.)
```bash
git clone <your-repo>
cd layer1
docker compose up --build -d
# Open port 8002 and 3000 in your firewall/security group
```

---

## How to use the SaaS UI

### Step 1 — Ingest data
- Go to **Data ingestion**
- Drop `data/telco_demo.csv` into the upload zone (or any Telco/Bank churn CSV)
- The pipeline preprocesses 500 customers into the feature store

### Step 2 — Train models
- Go to **Train models**
- Upload the same CSV
- XGBoost + MLP train in parallel (~20–40s)
- Champion is selected by AUC-ROC (XGBoost wins at 0.987)

### Step 3 — Predict
- Go to **Predict churn**
- Enter any customer ID (e.g. `CUST-00003`) → single prediction + SHAP explanation
- Or click **Run batch prediction** to score all 500 customers at once

### Step 4 — Kafka stream
- Go to **Kafka stream**
- Click **Start stream** to replay customer events
- Watch drift scores accumulate — when a customer hits threshold, Layer 2 fires automatically

### Step 5 — Metrics
- Go to **Model metrics** for the full AUC/F1/Precision/Recall comparison and SHAP chart

---

## API quick reference

All endpoints are at `http://localhost:8002`. Full interactive docs at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/csv` | Upload CSV → preprocess → feature store |
| POST | `/stream/event` | Send one Kafka event |
| POST | `/stream/batch` | Replay array of events |
| GET  | `/feature-store` | All customer feature vectors |
| GET  | `/drift-log` | Drift detection history |
| GET  | `/event-log` | Raw event log |
| POST | `/train` | Train XGBoost + MLP |
| POST | `/predict/customer` | Predict + explain single customer |
| POST | `/predict/batch` | Score all customers |
| GET  | `/predict/stream-triggers` | Score drift-flagged customers |
| POST | `/explain` | Full SHAP breakdown |
| GET  | `/metrics` | Training report |
| GET  | `/report` | Full pipeline status |

---

## Connecting your own data

The system is fully data-independent. Any CSV with these columns works:

**Required:** A binary churn label column (`Churn` with `Yes`/`No` or `1`/`0`)

**Recommended columns (Telco-style):**
`customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`,
`InternetService`, `PaymentMethod`, `gender`, `SeniorCitizen`, etc.

**Recommended columns (Bank-style):**
`CustomerId`, `Tenure`, `Balance`, `NumOfProducts`, `IsActiveMember`,
`EstimatedSalary`, `Exited` (rename to `Churn`)

Missing values are imputed automatically. Categorical columns are label-encoded.
New column sets are handled dynamically — no code changes needed.

---

## Thesis context

This system implements **Layer 1** and **Layer 2** of a 3-layer thesis architecture:

```
Layer 1  →  Layer 2  →  Layer 3 (next)
Ingest      Predict     DiCE counterfactual + cost-constrained offer optimization
```

Layer 3 will take the 186 at-risk customers flagged by Layer 2 and generate
per-customer retention offers using Microsoft DiCE, ranked by cost under
configurable budget constraints (Conservative / Moderate / Aggressive ROI scenarios).

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| Prediction | XGBoost 3.2, Scikit-learn MLP |
| Explainability | SHAP 0.51 (TreeExplainer) |
| Imbalance | imbalanced-learn SMOTE |
| Data | Pandas 3.0, NumPy 2.4 |
| Frontend | Vanilla HTML/CSS/JS (zero dependencies) |
| Container | Docker + Nginx |
