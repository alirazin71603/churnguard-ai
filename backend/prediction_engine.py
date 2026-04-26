"""
Layer 2 — Dual Prediction Engine
XGBoost (gradient-boosted ensemble) + Sklearn MLP (deep learning proxy)
Both trained on Layer 1 output. Winner feeds into Layer 3 (DiCE).
Includes SHAP-based feature importance and class imbalance handling via SMOTE.
"""
import numpy as np
import pandas as pd
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import shap


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class DualPredictionEngine:
    """
    Layer 2 — Dual Prediction Engine.

    Components:
        - XGBoost: Gradient-boosted ensemble (baseline, interpretable)
        - MLP:     Multi-layer perceptron [256→128→64] with ReLU + dropout proxy
        - SHAP:    Feature importance for XGBoost
        - SMOTE:   Handles class imbalance (applied to training set only)
        - Champion selection: Best AUC model feeds Layer 3
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names or []
        self.xgb_model = None
        self.mlp_model = None
        self.champion = None          # 'xgboost' or 'mlp'
        self.champion_model = None
        self.shap_explainer = None
        self.shap_values = None
        self.training_report = {}
        self.is_trained = False

    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────

    def train(self, X, y, feature_names=None):
        """
        Full training pipeline:
        1. Train/test split (stratified 80/20)
        2. SMOTE on training set only
        3. Train XGBoost + MLP
        4. Evaluate both on test set
        5. Select champion by AUC-ROC
        6. Compute SHAP values for XGBoost
        """
        if feature_names:
            self.feature_names = feature_names

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # ── 1. Train/test split ──
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── 2. SMOTE (training only) ──
        class_counts = np.bincount(y_train)
        imbalance_ratio = class_counts[0] / max(class_counts[1], 1)
        smote_applied = False

        if imbalance_ratio > 1.5 and class_counts[1] >= 6:
            smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts[1]-1))
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            smote_applied = True
        else:
            X_train_res, y_train_res = X_train, y_train

        # ── 3a. XGBoost ──
        scale_pos = class_counts[0] / max(class_counts[1], 1)
        self.xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
        self.xgb_model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # ── 3b. MLP ──
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,           # L2 regularization (proxy for dropout)
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.mlp_model.fit(X_train_res, y_train_res)

        # ── 4. Evaluate ──
        xgb_metrics = self._evaluate(self.xgb_model, X_test, y_test, 'XGBoost')
        mlp_metrics = self._evaluate(self.mlp_model, X_test, y_test, 'MLP')

        # ── 5. Champion selection ──
        if xgb_metrics['auc_roc'] >= mlp_metrics['auc_roc']:
            self.champion = 'xgboost'
            self.champion_model = self.xgb_model
        else:
            self.champion = 'mlp'
            self.champion_model = self.mlp_model

        # ── 6. SHAP for XGBoost ──
        shap_summary = self._compute_shap(X_test)

        # ── Cross-validation (champion) ──
        cv_scores = cross_val_score(
            self.champion_model, X, y, cv=5, scoring='roc_auc', n_jobs=-1
        )

        self.training_report = {
            'smote_applied': smote_applied,
            'train_size': len(X_train_res),
            'test_size': len(X_test),
            'class_distribution_original': {
                'no_churn': int(class_counts[0]),
                'churn': int(class_counts[1])
            },
            'xgboost': xgb_metrics,
            'mlp': mlp_metrics,
            'champion': self.champion,
            'champion_cv_auc_mean': round(float(cv_scores.mean()), 4),
            'champion_cv_auc_std': round(float(cv_scores.std()), 4),
            'shap_top_features': shap_summary,
            'feature_names': self.feature_names
        }

        self.is_trained = True
        self._save_models()
        return self.training_report

    def _evaluate(self, model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred).tolist()
        return {
            'model': name,
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'auc_roc': round(roc_auc_score(y_test, y_prob), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True
            )
        }

    def _compute_shap(self, X_test):
        """Compute SHAP values for XGBoost and return top feature importances."""
        try:
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_vals = explainer.shap_values(X_test)
            self.shap_explainer = explainer
            self.shap_values = shap_vals

            if self.feature_names:
                mean_shap = np.abs(shap_vals).mean(axis=0)
                top_idx = np.argsort(mean_shap)[::-1][:8]
                return [
                    {
                        'feature': self.feature_names[i] if i < len(self.feature_names) else f'f{i}',
                        'mean_abs_shap': round(float(mean_shap[i]), 5)
                    }
                    for i in top_idx
                ]
        except Exception as e:
            return [{'error': str(e)}]
        return []

    # ─────────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────────

    def predict(self, X, model='champion'):
        """
        Predict churn probability for one or more customers.
        Returns list of {churn_probability, churn_predicted, model_used}
        """
        if not self.is_trained:
            raise RuntimeError("Engine not trained. Call train() first.")

        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        m = self._resolve_model(model)
        probs = m.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results = []
        for prob, pred in zip(probs, preds):
            results.append({
                'churn_probability': round(float(prob), 4),
                'churn_predicted': int(pred),
                'churn_label': 'CHURN' if pred == 1 else 'RETAIN',
                'model_used': model if model != 'champion' else self.champion,
                'risk_tier': self._risk_tier(float(prob))
            })
        return results

    def predict_single(self, feature_vector, customer_id=None, model='champion'):
        """Convenience wrapper for a single customer."""
        result = self.predict([feature_vector], model=model)[0]
        if customer_id:
            result['customer_id'] = customer_id
        return result

    def explain_prediction(self, feature_vector, customer_id=None):
        """Returns SHAP explanation for a single customer (XGBoost only)."""
        if self.shap_explainer is None:
            return {'error': 'SHAP explainer not available'}

        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        shap_vals = self.shap_explainer.shap_values(X)[0]

        contributions = []
        for i, (name, val) in enumerate(zip(
            self.feature_names or [f'f{i}' for i in range(len(shap_vals))],
            shap_vals
        )):
            contributions.append({
                'feature': name,
                'shap_value': round(float(val), 5),
                'direction': 'increases_churn' if val > 0 else 'decreases_churn'
            })

        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        return {
            'customer_id': customer_id,
            'top_churn_drivers': contributions[:5],
            'all_contributions': contributions
        }

    # ─────────────────────────────────────────────
    # BATCH PREDICTION (for Layer 1 feature store)
    # ─────────────────────────────────────────────

    def predict_from_feature_store(self, feature_store: dict, model='champion'):
        """
        Run predictions on all customers in the Layer 1 feature store.
        Only processes customers that have a 'features' key (CSV mode entries).
        """
        results = {}
        triggered_only = []

        for cid, data in feature_store.items():
            if isinstance(data, dict) and 'features' in data:
                fv = data['features']
                pred = self.predict_single(fv, customer_id=cid, model=model)
                results[cid] = pred
                if pred['churn_predicted'] == 1:
                    triggered_only.append(pred)

        return {
            'total_scored': len(results),
            'churn_predicted_count': len(triggered_only),
            'churn_rate': round(len(triggered_only) / max(len(results), 1) * 100, 2),
            'at_risk_customers': triggered_only[:20],
            'all_predictions': results
        }

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _resolve_model(self, model_name):
        if model_name == 'champion':
            return self.champion_model
        elif model_name == 'xgboost':
            return self.xgb_model
        elif model_name == 'mlp':
            return self.mlp_model
        raise ValueError(f"Unknown model: {model_name}")

    def _risk_tier(self, prob):
        if prob >= 0.75:
            return 'HIGH'
        elif prob >= 0.5:
            return 'MEDIUM'
        elif prob >= 0.25:
            return 'LOW'
        return 'MINIMAL'

    def _save_models(self):
        with open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(self.xgb_model, f)
        with open(os.path.join(MODEL_DIR, 'mlp_model.pkl'), 'wb') as f:
            pickle.dump(self.mlp_model, f)
        meta = {
            'champion': self.champion,
            'feature_names': self.feature_names,
            'training_report': self.training_report
        }
        with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    def load_models(self):
        xgb_path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
        mlp_path = os.path.join(MODEL_DIR, 'mlp_model.pkl')
        meta_path = os.path.join(MODEL_DIR, 'meta.json')

        if not all(os.path.exists(p) for p in [xgb_path, mlp_path, meta_path]):
            return False

        with open(xgb_path, 'rb') as f:
            self.xgb_model = pickle.load(f)
        with open(mlp_path, 'rb') as f:
            self.mlp_model = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)

        self.champion = meta['champion']
        self.feature_names = meta.get('feature_names', [])
        self.training_report = meta.get('training_report', {})
        self.champion_model = self.xgb_model if self.champion == 'xgboost' else self.mlp_model

        # Rebuild SHAP explainer
        try:
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        except Exception:
            pass

        self.is_trained = True
        return True
