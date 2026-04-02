"""
Unsupervised Fraud Detection Module
- Isolation Forest for anomaly detection
- Autoencoder reconstruction error for suspicious transaction detection
- Combined fraud risk scoring
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from config import FRAUD_CONFIG, MODEL_DIR


class FraudDetector:
    def __init__(self):
        self.isolation_forest = None
        self.reconstruction_threshold = None
        self.risk_scaler = MinMaxScaler()
        self.is_fitted = False

    def fit_isolation_forest(self, X):
        """Fit Isolation Forest model"""
        self.isolation_forest = IsolationForest(
            n_estimators=FRAUD_CONFIG["n_estimators"],
            contamination=FRAUD_CONFIG["isolation_forest_contamination"],
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X)
        print(f"[FraudDetector] Isolation Forest fitted on {len(X)} samples")

    def set_reconstruction_threshold(self, reconstruction_errors):
        """Set threshold based on reconstruction error distribution"""
        self.reconstruction_threshold = np.percentile(
            reconstruction_errors,
            FRAUD_CONFIG["reconstruction_threshold_percentile"]
        )
        print(f"[FraudDetector] Reconstruction threshold: {self.reconstruction_threshold:.6f} "
              f"(P{FRAUD_CONFIG['reconstruction_threshold_percentile']})")

    def compute_fraud_scores(self, X, reconstruction_errors):
        """
        Compute combined fraud risk scores.
        Returns scores between 0 (normal) and 1 (highly suspicious)
        """
        # Isolation Forest scores (-1 to 1, lower = more anomalous)
        if_scores = self.isolation_forest.decision_function(X)
        # Convert: lower decision function -> higher fraud risk
        if_risk = 1 - MinMaxScaler().fit_transform(if_scores.reshape(-1, 1)).flatten()

        # Reconstruction error scores
        if self.reconstruction_threshold and self.reconstruction_threshold > 0:
            re_risk = reconstruction_errors / (self.reconstruction_threshold * 2)
            re_risk = np.clip(re_risk, 0, 1)
        else:
            re_risk = MinMaxScaler().fit_transform(reconstruction_errors.reshape(-1, 1)).flatten()

        # Combined score (weighted average)
        combined_risk = 0.5 * if_risk + 0.5 * re_risk

        # Normalize to 0-1
        combined_risk = np.clip(combined_risk, 0, 1)

        self.is_fitted = True
        return {
            "combined_risk": combined_risk.tolist(),
            "isolation_forest_risk": if_risk.tolist(),
            "reconstruction_error_risk": re_risk.tolist(),
            "reconstruction_errors": reconstruction_errors.tolist(),
        }

    def classify_risk(self, fraud_scores):
        """Classify transactions into risk categories"""
        combined = np.array(fraud_scores["combined_risk"])
        categories = []
        for score in combined:
            if score >= 0.8:
                categories.append("Critical")
            elif score >= 0.6:
                categories.append("High")
            elif score >= 0.4:
                categories.append("Medium")
            elif score >= 0.2:
                categories.append("Low")
            else:
                categories.append("Normal")
        return categories

    def get_anomaly_labels(self, X):
        """Get Isolation Forest anomaly labels (-1 = anomaly, 1 = normal)"""
        return self.isolation_forest.predict(X).tolist()

    def get_statistics(self, fraud_scores):
        """Return fraud detection statistics"""
        combined = np.array(fraud_scores["combined_risk"])
        categories = self.classify_risk(fraud_scores)

        stats = {
            "total_transactions": len(combined),
            "mean_risk_score": float(np.mean(combined)),
            "median_risk_score": float(np.median(combined)),
            "max_risk_score": float(np.max(combined)),
            "min_risk_score": float(np.min(combined)),
            "std_risk_score": float(np.std(combined)),
            "category_distribution": {
                "Normal": categories.count("Normal"),
                "Low": categories.count("Low"),
                "Medium": categories.count("Medium"),
                "High": categories.count("High"),
                "Critical": categories.count("Critical"),
            },
            "flagged_count": sum(1 for c in categories if c in ("High", "Critical")),
            "flagged_percentage": float(sum(1 for c in categories if c in ("High", "Critical")) / len(categories) * 100)
        }
        return stats

    def save_model(self):
        """Save fraud detection models"""
        joblib.dump(self.isolation_forest, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
        joblib.dump(self.reconstruction_threshold, os.path.join(MODEL_DIR, "recon_threshold.pkl"))
        print("[FraudDetector] Models saved")

    def load_model(self):
        """Load fraud detection models"""
        self.isolation_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
        self.reconstruction_threshold = joblib.load(os.path.join(MODEL_DIR, "recon_threshold.pkl"))
        self.is_fitted = True
        print("[FraudDetector] Models loaded")
