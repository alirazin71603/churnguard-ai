"""
Layer 1 — Preprocessing Pipeline
Handles feature engineering, encoding, normalization for both CSV and Kafka stream modes.
Outputs a clean feature vector ready for Layer 2 (prediction engine).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json
import warnings
warnings.filterwarnings('ignore')


NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
TARGET_COL = 'Churn'
ID_COL = 'customerID'


class Layer1Pipeline:
    """
    Full Layer 1 Data Ingestion & Preprocessing Pipeline.
    
    Modes:
        - csv_batch:   Accepts a CSV file path or DataFrame
        - kafka_event: Processes a single JSON event from Kafka consumer
        - kafka_batch: Processes a list of events (for simulation/replay)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_store = {}         # customerID → latest feature vector
        self.event_log = []             # audit trail
        self.drift_log = []             # feature drift detections
        self.is_fitted = False
        self.feature_names = []

    # ─────────────────────────────────────────────
    # CSV BATCH MODE
    # ─────────────────────────────────────────────

    def ingest_csv(self, filepath_or_df):
        """
        Ingest and preprocess a full CSV dataset.
        Returns: (feature_matrix, labels, customer_ids, summary_dict)
        """
        if isinstance(filepath_or_df, str):
            df = pd.read_csv(filepath_or_df)
        else:
            df = filepath_or_df.copy()

        summary = {
            'mode': 'csv_batch',
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_before': df.isnull().sum().to_dict(),
            'churn_distribution': {}
        }

        # Store IDs and labels
        customer_ids = df[ID_COL].tolist() if ID_COL in df.columns else [f'CUST-{i}' for i in range(len(df))]
        labels = None
        if TARGET_COL in df.columns:
            raw_labels = df[TARGET_COL].astype(str).str.strip()
            labels = raw_labels.map({'Yes': 1, 'No': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}).fillna(0).astype(int).values
            churn_counts = df[TARGET_COL].value_counts().to_dict()
            summary['churn_distribution'] = churn_counts
            summary['churn_rate'] = round(
                (churn_counts.get('Yes', churn_counts.get(1, 0)) / len(df)) * 100, 2
            )

        # Drop non-feature columns
        drop_cols = [c for c in [ID_COL, TARGET_COL] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # Fix TotalCharges (sometimes stored as string with spaces)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Impute missing numerics
        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if num_cols:
            df[num_cols] = self.imputer.fit_transform(df[num_cols])

        # Encode categoricals
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Scale numerics
        if num_cols:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])

        self.feature_names = list(df.columns)
        self.is_fitted = True

        # Update feature store
        feature_matrix = df.values
        for i, cid in enumerate(customer_ids):
            self.feature_store[cid] = {
                'features': feature_matrix[i].tolist(),
                'label': int(labels[i]) if labels is not None else None
            }

        summary['missing_after'] = df.isnull().sum().to_dict()
        summary['feature_names'] = self.feature_names
        summary['feature_count'] = len(self.feature_names)

        return feature_matrix, labels, customer_ids, summary

    # ─────────────────────────────────────────────
    # KAFKA EVENT MODE
    # ─────────────────────────────────────────────

    def process_kafka_event(self, event: dict):
        """
        Process a single Kafka event.
        Updates feature store and checks for drift.
        Returns: {customer_id, updated_features, drift_detected, trigger_prediction}
        """
        cid = event.get('customer_id', 'UNKNOWN')
        event_type = event.get('event_type', 'unknown')
        metadata = event.get('metadata', {})

        self.event_log.append({
            'customer_id': cid,
            'event_type': event_type,
            'timestamp': event.get('timestamp')
        })

        # Get or initialize customer feature state
        existing = self.feature_store.get(cid, {
            'tenure': metadata.get('tenure', 0),
            'monthly_charges': metadata.get('monthly_charges', 50.0),
            'event_count': 0,
            'churn_signal_count': 0,
            'last_event': None,
            'feature_drift_score': 0.0
        })

        # Update event counters
        existing['event_count'] = existing.get('event_count', 0) + 1
        existing['last_event'] = event_type

        churn_signal_events = {
            'call_drop', 'payment_failure', 'service_complaint',
            'plan_downgrade', 'data_usage_drop', 'support_ticket'
        }
        if event_type in churn_signal_events:
            existing['churn_signal_count'] = existing.get('churn_signal_count', 0) + 1

        # Compute drift score
        signal_count = existing.get('churn_signal_count', 0)
        drift_score = min(signal_count / 3.0, 1.0)  # saturates at 3 churn events
        existing['feature_drift_score'] = round(drift_score, 3)

        drift_detected = drift_score >= 0.5
        trigger_prediction = drift_detected

        if drift_detected:
            self.drift_log.append({
                'customer_id': cid,
                'drift_score': drift_score,
                'event_type': event_type,
                'churn_signal_count': signal_count
            })

        self.feature_store[cid] = existing

        return {
            'customer_id': cid,
            'event_type': event_type,
            'updated_features': existing,
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'trigger_prediction': trigger_prediction,
            'churn_signal_count': signal_count
        }

    def process_kafka_batch(self, events: list):
        """Process a list of Kafka events (replay mode)."""
        results = []
        for event in events:
            result = self.process_kafka_event(event)
            results.append(result)
        return results

    # ─────────────────────────────────────────────
    # REPORTS
    # ─────────────────────────────────────────────

    def get_pipeline_report(self):
        """Returns a summary report of the current pipeline state."""
        triggered = [cid for cid, data in self.feature_store.items()
                     if isinstance(data, dict) and data.get('feature_drift_score', 0) >= 0.5]
        return {
            'feature_store_size': len(self.feature_store),
            'total_events_processed': len(self.event_log),
            'drift_detections': len(self.drift_log),
            'customers_triggered_for_prediction': len(triggered),
            'triggered_customer_ids': triggered[:10],
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names
        }
