"""
Training Pipeline - Orchestrates the full ML pipeline:
1. Data preprocessing
2. Autoencoder training + feature extraction
3. SNN training for price prediction
4. Fraud detection fitting
5. User segmentation clustering
"""
import numpy as np
import sys
import os
import io

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import DataPreprocessor
from autoencoder_model import DeepAutoencoder
from snn_model import SNNTrainer
from fraud_detector import FraudDetector
from user_segmentation import UserSegmentation
from config import CSV_PATH


def run_training_pipeline(csv_path=None):
    """Run the complete training pipeline"""
    csv_path = csv_path or CSV_PATH

    print("=" * 60)
    print("  AI Real Estate Analytics - Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocess data
    print("\n[Step 1/5] Data Preprocessing...")
    preprocessor = DataPreprocessor()
    X, y, processed_df = preprocessor.preprocess_pipeline(csv_path)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 2: Train autoencoder
    print("\n[Step 2/5] Training Deep Autoencoder...")
    autoencoder = DeepAutoencoder(input_dim=X.shape[1])
    autoencoder.train(X_train, X_val=X_test)
    autoencoder.save_model()

    # Extract encoded features
    X_encoded_train = autoencoder.encode(X_train)
    X_encoded_test = autoencoder.encode(X_test)
    X_encoded_all = autoencoder.encode(X)

    print(f"  Encoded features shape: {X_encoded_train.shape}")

    ae_history = autoencoder.get_training_history()

    # Step 3: Train SNN
    print("\n[Step 3/5] Training Spiking Neural Network...")
    snn_trainer = SNNTrainer(input_size=X_encoded_train.shape[1])
    snn_history = snn_trainer.train(X_encoded_train, y_train, X_encoded_test, y_test)
    snn_metrics = snn_trainer.evaluate(X_encoded_test, y_test)

    # Predictions
    predictions_normalized = snn_trainer.predict(X_encoded_test)
    predictions_actual = preprocessor.inverse_transform_price(predictions_normalized)
    actual_prices = preprocessor.inverse_transform_price(y_test)

    print(f"  Sample predictions vs actual:")
    for i in range(min(5, len(predictions_actual))):
        print(f"    Predicted: INR {predictions_actual[i]:,.0f} | Actual: INR {actual_prices[i]:,.0f}")

    # Step 4: Fraud Detection
    print("\n[Step 4/5] Fitting Fraud Detection...")
    fraud_detector = FraudDetector()
    fraud_detector.fit_isolation_forest(X)

    reconstruction_errors = autoencoder.reconstruction_error(X)
    fraud_detector.set_reconstruction_threshold(reconstruction_errors)

    fraud_scores = fraud_detector.compute_fraud_scores(X, reconstruction_errors)
    fraud_stats = fraud_detector.get_statistics(fraud_scores)
    fraud_categories = fraud_detector.classify_risk(fraud_scores)

    fraud_detector.save_model()

    print(f"  Flagged transactions: {fraud_stats['flagged_count']}/{fraud_stats['total_transactions']} "
          f"({fraud_stats['flagged_percentage']:.1f}%)")

    # Step 5: User Segmentation
    print("\n[Step 5/5] Running User Segmentation...")
    segmentation = UserSegmentation()
    elbow_data = segmentation.find_optimal_clusters(X)
    labels, X_pca = segmentation.fit(X)
    cluster_profiles = segmentation.profile_clusters(
        processed_df, labels, preprocessor.feature_columns
    )

    segmentation.save_model()

    print(f"  Clusters found: {segmentation.n_clusters}")
    for cid, profile in cluster_profiles.items():
        print(f"    Cluster {cid} ({profile['segment_label']}): "
              f"{profile['size']} properties ({profile['percentage']:.1f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Price Prediction R²: {snn_metrics['r2_score']:.4f}")
    print(f"  Fraud Detection: {fraud_stats['flagged_count']} suspicious transactions")
    print(f"  User Segments: {segmentation.n_clusters} clusters identified")
    print("=" * 60)

    return {
        "preprocessor": preprocessor,
        "autoencoder": autoencoder,
        "snn_trainer": snn_trainer,
        "fraud_detector": fraud_detector,
        "segmentation": segmentation,
        "metrics": {
            "snn_metrics": snn_metrics,
            "ae_history": ae_history,
            "snn_history": snn_history,
            "fraud_stats": fraud_stats,
            "elbow_data": elbow_data,
            "cluster_profiles": {str(k): v for k, v in cluster_profiles.items()},
        },
        "data": {
            "X": X,
            "y": y,
            "X_encoded": X_encoded_all,
            "predictions_actual": predictions_actual.tolist(),
            "actual_prices": actual_prices.tolist(),
            "fraud_scores": fraud_scores,
            "fraud_categories": fraud_categories,
            "labels": labels.tolist(),
            "pca_data": segmentation.get_pca_data(X, labels),
        }
    }


if __name__ == "__main__":
    results = run_training_pipeline()
