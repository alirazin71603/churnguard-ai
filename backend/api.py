"""
Full API — Layer 1 + Layer 2
Endpoints: /ingest/csv, /stream/event, /stream/batch, /report, /feature-store
           /train, /predict/customer, /predict/batch, /predict/stream-triggers
           /explain, /metrics
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import json
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import Layer1Pipeline
from prediction_engine import DualPredictionEngine

app = FastAPI(
    title="Churn Retention AI — Layer 1 + Layer 2",
    description="Real-Time Ingestion + Dual Prediction Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Layer1Pipeline()
engine = DualPredictionEngine()
engine.load_models()


class KafkaEvent(BaseModel):
    event_id: str
    customer_id: str
    event_type: str
    timestamp: str
    metadata: dict = {}

class KafkaBatchRequest(BaseModel):
    events: List[KafkaEvent]

class PredictRequest(BaseModel):
    customer_id: str
    model: Optional[str] = 'champion'

class PredictBatchRequest(BaseModel):
    customer_ids: Optional[List[str]] = None
    model: Optional[str] = 'champion'


@app.get("/")
def root():
    return {
        "system": "Churn Retention AI",
        "pipeline_fitted": pipeline.is_fitted,
        "engine_trained": engine.is_trained,
        "champion_model": engine.champion if engine.is_trained else None,
        "layer1_endpoints": ["/ingest/csv","/stream/event","/stream/batch","/report","/feature-store","/drift-log","/event-log"],
        "layer2_endpoints": ["/train","/predict/customer","/predict/batch","/predict/stream-triggers","/explain","/metrics"]
    }


# ── LAYER 1 ──────────────────────────────────────────

@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only .csv files accepted")
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {str(e)}")
    try:
        feature_matrix, labels, customer_ids, summary = pipeline.ingest_csv(df)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {str(e)}")
    preview = []
    for i in range(min(5, len(customer_ids))):
        preview.append({
            'customerID': customer_ids[i],
            'churn_label': int(labels[i]) if labels is not None else None,
            'feature_vector_preview': [round(float(x), 4) for x in feature_matrix[i][:6]]
        })
    return {"status":"success","mode":"csv_batch","summary":summary,"preview":preview}


@app.post("/stream/event")
async def stream_event(event: KafkaEvent):
    result = pipeline.process_kafka_event(event.dict())
    response = {
        "status": "processed", "mode": "kafka_stream",
        "result": result,
        "message": (
            f"Drift detected for {event.customer_id} — triggering Layer 2"
            if result['trigger_prediction']
            else f"Event logged for {event.customer_id}"
        )
    }
    if result['trigger_prediction'] and engine.is_trained:
        stored = pipeline.feature_store.get(event.customer_id, {})
        if 'features' in stored:
            pred = engine.predict_single(stored['features'], customer_id=event.customer_id)
            response['auto_prediction'] = pred
    return response


@app.post("/stream/batch")
async def stream_batch(request: KafkaBatchRequest):
    events = [e.dict() for e in request.events]
    results = pipeline.process_kafka_batch(events)
    triggered = [r for r in results if r['trigger_prediction']]
    return {
        "status":"processed","mode":"kafka_batch_replay",
        "total_events":len(results),"drift_triggered":len(triggered),
        "triggered_customers":list(set(r['customer_id'] for r in triggered)),
        "event_results":results
    }


@app.get("/report")
def get_report():
    report = pipeline.get_pipeline_report()
    if engine.is_trained:
        report['layer2'] = {
            'champion': engine.champion,
            'auc': engine.training_report.get(engine.champion, {}).get('auc_roc'),
            'f1': engine.training_report.get(engine.champion, {}).get('f1_score')
        }
    return report


@app.get("/feature-store")
def get_feature_store(limit: int = 20):
    items = list(pipeline.feature_store.items())[:limit]
    return {"total_customers":len(pipeline.feature_store),"feature_store":{k:v for k,v in items}}


@app.get("/drift-log")
def get_drift_log():
    return {"total":len(pipeline.drift_log),"drift_log":pipeline.drift_log}


@app.get("/event-log")
def get_event_log(limit: int = 50):
    return {"total_events":len(pipeline.event_log),"events":pipeline.event_log[-limit:]}


# ── LAYER 2 ──────────────────────────────────────────

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only .csv files accepted")
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        X, y, ids, summary = pipeline.ingest_csv(df)
        report = engine.train(X, y, feature_names=summary['feature_names'])
    except Exception as e:
        raise HTTPException(500, f"Training error: {str(e)}")
    return {
        "status":"trained",
        "champion":report['champion'],
        "xgboost":{k:v for k,v in report['xgboost'].items() if k!='classification_report'},
        "mlp":{k:v for k,v in report['mlp'].items() if k!='classification_report'},
        "cv_auc":f"{report['champion_cv_auc_mean']} ± {report['champion_cv_auc_std']}",
        "smote_applied":report['smote_applied'],
        "shap_top_features":report['shap_top_features'],
        "feature_names":report['feature_names']
    }


@app.post("/predict/customer")
def predict_customer(req: PredictRequest):
    if not engine.is_trained:
        raise HTTPException(400, "Engine not trained. POST /train first.")
    stored = pipeline.feature_store.get(req.customer_id)
    if not stored:
        raise HTTPException(404, f"{req.customer_id} not in feature store.")
    if 'features' not in stored:
        raise HTTPException(400, f"{req.customer_id} has no feature vector.")
    pred = engine.predict_single(stored['features'], customer_id=req.customer_id, model=req.model)
    exp = engine.explain_prediction(stored['features'], customer_id=req.customer_id)
    pred['explanation'] = exp['top_churn_drivers']
    return pred


@app.post("/predict/batch")
def predict_batch(req: PredictBatchRequest):
    if not engine.is_trained:
        raise HTTPException(400, "Engine not trained.")
    subset = (
        {cid: pipeline.feature_store[cid] for cid in req.customer_ids if cid in pipeline.feature_store}
        if req.customer_ids else pipeline.feature_store
    )
    return engine.predict_from_feature_store(subset, model=req.model)


@app.get("/predict/stream-triggers")
def predict_stream_triggers():
    if not engine.is_trained:
        raise HTTPException(400, "Engine not trained.")
    triggered = {
        cid: data for cid, data in pipeline.feature_store.items()
        if isinstance(data, dict) and data.get('feature_drift_score', 0) >= 0.5 and 'features' in data
    }
    if not triggered:
        return {"message":"No drift-triggered customers with feature vectors.","results":[]}
    return engine.predict_from_feature_store(triggered)


@app.post("/explain")
def explain_customer(req: PredictRequest):
    if not engine.is_trained:
        raise HTTPException(400, "Engine not trained.")
    stored = pipeline.feature_store.get(req.customer_id)
    if not stored or 'features' not in stored:
        raise HTTPException(404, f"{req.customer_id} not found.")
    return engine.explain_prediction(stored['features'], customer_id=req.customer_id)


@app.get("/metrics")
def get_metrics():
    if not engine.is_trained:
        raise HTTPException(400, "Engine not trained.")
    return {
        "champion": engine.champion,
        "xgboost": engine.training_report.get('xgboost', {}),
        "mlp": engine.training_report.get('mlp', {}),
        "cv_auc_mean": engine.training_report.get('champion_cv_auc_mean'),
        "cv_auc_std": engine.training_report.get('champion_cv_auc_std'),
        "shap_top_features": engine.training_report.get('shap_top_features', []),
        "smote_applied": engine.training_report.get('smote_applied'),
        "feature_names": engine.training_report.get('feature_names', [])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
