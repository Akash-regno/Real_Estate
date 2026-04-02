"""
FastAPI Backend - AI Real Estate Analytics System
Serves the React frontend with REST API endpoints for:
- Price prediction
- Fraud detection
- User segmentation
- Dataset exploration
- Model training & metrics
"""
import sys
import os
import json
import traceback
import io

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'buffer') and sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd

from config import CSV_PATH, AMENITY_COLUMNS, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN
from data_preprocessor import DataPreprocessor
from autoencoder_model import DeepAutoencoder
from snn_model import SNNTrainer
from fraud_detector import FraudDetector
from user_segmentation import UserSegmentation

app = FastAPI(
    title="AI Real Estate Analytics API",
    description="Intelligent real estate analytics with SNN price prediction, fraud detection, and user segmentation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline_results = None
is_trained = False


# ===================== Pydantic Models =====================
class PredictionRequest(BaseModel):
    area: float
    bedrooms: int
    location: str
    new_resale: str  # "New" or "Resale"
    gymnasium: int = 0
    swimming_pool: int = 0
    landscaped_gardens: int = 0
    jogging_track: int = 0
    rainwater_harvesting: int = 0
    indoor_games: int = 0
    shopping_mall: int = 0
    intercom: int = 0
    sports_facility: int = 0
    atm: int = 0
    club_house: int = 0
    school: int = 0
    hospital_township: int = 0
    security_24x7: int = 1
    power_backup: int = 1
    car_parking: int = 1
    staff_quarter: int = 0
    cafeteria: int = 0
    multipurpose_room: int = 0
    hospital_2km: int = 0
    locality_2km: int = 1
    property_tax: float = 50000
    super_built_up_area: float = 0
    carpet_area: float = 0
    maintenance: float = 3000


class TrainRequest(BaseModel):
    csv_path: Optional[str] = None


# ===================== Startup =====================
@app.on_event("startup")
async def startup_event():
    """Auto-train on startup if not already trained"""
    global pipeline_results, is_trained
    try:
        from train_pipeline import run_training_pipeline
        pipeline_results = run_training_pipeline()
        is_trained = True
        print("[API] Pipeline trained successfully on startup")
    except Exception as e:
        print(f"[API] Startup training error: {e}")
        traceback.print_exc()
        is_trained = False


# ===================== API Endpoints =====================

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "trained": is_trained}


@app.get("/api/dataset/info")
async def dataset_info():
    """Get dataset overview"""
    try:
        df = pd.read_csv(CSV_PATH)
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "price_stats": {
                "mean": float(df[TARGET_COLUMN].mean()),
                "median": float(df[TARGET_COLUMN].median()),
                "min": float(df[TARGET_COLUMN].min()),
                "max": float(df[TARGET_COLUMN].max()),
                "std": float(df[TARGET_COLUMN].std()),
            },
            "locations": df["Location"].unique().tolist() if "Location" in df.columns else [],
            "bedroom_counts": sorted(df["No. of Bedrooms"].unique().tolist()) if "No. of Bedrooms" in df.columns else [],
            "sample_data": df.head(10).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/statistics")
async def dataset_statistics():
    """Get detailed dataset statistics for dashboard"""
    try:
        df = pd.read_csv(CSV_PATH)

        # Price distribution by location
        price_by_location = df.groupby("Location")[TARGET_COLUMN].agg(['mean', 'count']).reset_index()
        price_by_location.columns = ['location', 'avg_price', 'count']
        price_by_location = price_by_location.sort_values('avg_price', ascending=False)

        # Price distribution by bedrooms
        price_by_bedrooms = df.groupby("No. of Bedrooms")[TARGET_COLUMN].agg(['mean', 'count']).reset_index()
        price_by_bedrooms.columns = ['bedrooms', 'avg_price', 'count']

        # Area vs Price
        area_price = df[['Area', TARGET_COLUMN]].dropna().values.tolist()

        # Amenity popularity
        amenity_counts = {}
        for col in AMENITY_COLUMNS:
            if col in df.columns:
                amenity_counts[col] = int(df[col].sum())

        # New vs Resale distribution
        condition_dist = df["New/Resale"].value_counts().to_dict() if "New/Resale" in df.columns else {}

        return {
            "price_by_location": price_by_location.to_dict(orient="records"),
            "price_by_bedrooms": price_by_bedrooms.to_dict(orient="records"),
            "area_price_scatter": area_price[:200],
            "amenity_popularity": amenity_counts,
            "condition_distribution": {str(k): int(v) for k, v in condition_dist.items()},
            "total_properties": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(request: TrainRequest = TrainRequest()):
    """Train the full pipeline"""
    global pipeline_results, is_trained
    try:
        from train_pipeline import run_training_pipeline
        pipeline_results = run_training_pipeline(request.csv_path)
        is_trained = True
        return {
            "status": "success",
            "metrics": pipeline_results["metrics"]["snn_metrics"],
            "fraud_stats": pipeline_results["metrics"]["fraud_stats"],
            "n_clusters": pipeline_results["segmentation"].n_clusters,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict_price(request: PredictionRequest):
    """Predict property price"""
    if not is_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Call /api/train first.")

    try:
        preprocessor = pipeline_results["preprocessor"]
        autoencoder = pipeline_results["autoencoder"]
        snn_trainer = pipeline_results["snn_trainer"]

        # Build feature vector in the same order as training
        feature_dict = {
            "Area": request.area,
            "No. of Bedrooms": request.bedrooms,
            "Property Tax": request.property_tax,
            "Super built-up Area": request.super_built_up_area or request.area * 1.15,
            "Carpet Area": request.carpet_area or request.area * 0.8,
            "Maintenance (INR/Month)": request.maintenance,
            "Gymnasium": request.gymnasium,
            "Swimming Pool": request.swimming_pool,
            "Landscaped Gardens": request.landscaped_gardens,
            "Jogging Track": request.jogging_track,
            "RainWater Harvesting": request.rainwater_harvesting,
            "Indoor Games": request.indoor_games,
            "Shopping Mall": request.shopping_mall,
            "Intercom": request.intercom,
            "Sports Facility": request.sports_facility,
            "ATM": request.atm,
            "Club House": request.club_house,
            "School / University in Township": request.school,
            "Hospital / Clinic in Township": request.hospital_township,
            "24X7Security": request.security_24x7,
            "Power Back up": request.power_backup,
            "Car Parking": request.car_parking,
            "Staff Quarter": request.staff_quarter,
            "Cafeteria": request.cafeteria,
            "Multipurpose Room": request.multipurpose_room,
            "Hospital within 2 KM": request.hospital_2km,
            "Locality within 2 KM": request.locality_2km,
        }

        # Encode categorical
        for col in CATEGORICAL_COLUMNS:
            if col == "Location":
                le = preprocessor.label_encoders.get(col)
                if le and request.location in le.classes_:
                    feature_dict["Location_encoded"] = int(le.transform([request.location])[0])
                else:
                    feature_dict["Location_encoded"] = 0
            elif col == "New/Resale":
                le = preprocessor.label_encoders.get(col)
                if le and request.new_resale in le.classes_:
                    feature_dict["New/Resale_encoded"] = int(le.transform([request.new_resale])[0])
                else:
                    feature_dict["New/Resale_encoded"] = 0

        # Build feature vector in correct order
        feature_vector = []
        for col in preprocessor.feature_columns:
            feature_vector.append(feature_dict.get(col, 0))

        feature_array = np.array([feature_vector])
        feature_scaled = preprocessor.scaler.transform(feature_array)

        # Encode with autoencoder
        encoded = autoencoder.encode(feature_scaled)

        # Predict with SNN
        prediction_normalized = snn_trainer.predict(encoded)
        predicted_price = preprocessor.inverse_transform_price(prediction_normalized)[0]

        # Fraud check
        fraud_detector = pipeline_results["fraud_detector"]
        recon_error = autoencoder.reconstruction_error(feature_scaled)
        fraud_scores = fraud_detector.compute_fraud_scores(feature_scaled, recon_error)
        fraud_category = fraud_detector.classify_risk(fraud_scores)[0]

        # Segment assignment
        segmentation = pipeline_results["segmentation"]
        cluster = segmentation.predict(feature_scaled)[0]
        cluster_profile = pipeline_results["metrics"]["cluster_profiles"].get(str(cluster), {})

        return {
            "predicted_price": float(predicted_price),
            "predicted_price_formatted": f"₹{predicted_price:,.0f}",
            "fraud_risk_score": float(fraud_scores["combined_risk"][0]),
            "fraud_category": fraud_category,
            "segment": int(cluster),
            "segment_label": cluster_profile.get("segment_label", f"Segment {cluster}"),
            "input_features": feature_dict,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/metrics")
async def get_model_metrics():
    """Get all model metrics and training history"""
    if not is_trained:
        raise HTTPException(status_code=400, detail="Model not trained.")

    try:
        return {
            "snn_metrics": pipeline_results["metrics"]["snn_metrics"],
            "ae_history": pipeline_results["metrics"]["ae_history"],
            "snn_history": pipeline_results["metrics"]["snn_history"],
            "snn_info": pipeline_results["snn_trainer"].get_model_info(),
            "ae_info": pipeline_results["autoencoder"].summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fraud/results")
async def get_fraud_results():
    """Get fraud detection results"""
    if not is_trained:
        raise HTTPException(status_code=400, detail="Model not trained.")

    try:
        df = pd.read_csv(CSV_PATH)
        fraud_scores = pipeline_results["data"]["fraud_scores"]
        fraud_categories = pipeline_results["data"]["fraud_categories"]
        fraud_stats = pipeline_results["metrics"]["fraud_stats"]

        # Build result table
        results = []
        for i in range(len(df)):
            results.append({
                "index": i,
                "location": df.iloc[i].get("Location", "N/A"),
                "price": float(df.iloc[i][TARGET_COLUMN]),
                "area": float(df.iloc[i].get("Area", 0)),
                "bedrooms": int(df.iloc[i].get("No. of Bedrooms", 0)),
                "combined_risk": float(fraud_scores["combined_risk"][i]),
                "if_risk": float(fraud_scores["isolation_forest_risk"][i]),
                "recon_risk": float(fraud_scores["reconstruction_error_risk"][i]),
                "category": fraud_categories[i],
            })

        return {
            "results": results,
            "statistics": fraud_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/segmentation/results")
async def get_segmentation_results():
    """Get user segmentation results"""
    if not is_trained:
        raise HTTPException(status_code=400, detail="Model not trained.")

    try:
        df = pd.read_csv(CSV_PATH)
        labels = pipeline_results["data"]["labels"]
        pca_data = pipeline_results["data"]["pca_data"]
        cluster_profiles = pipeline_results["metrics"]["cluster_profiles"]
        elbow_data = pipeline_results["metrics"]["elbow_data"]

        # Build result with property info
        properties = []
        for i in range(len(df)):
            properties.append({
                "index": i,
                "location": df.iloc[i].get("Location", "N/A"),
                "price": float(df.iloc[i][TARGET_COLUMN]),
                "area": float(df.iloc[i].get("Area", 0)),
                "bedrooms": int(df.iloc[i].get("No. of Bedrooms", 0)),
                "cluster": int(labels[i]),
            })

        return {
            "properties": properties,
            "pca_data": pca_data,
            "cluster_profiles": cluster_profiles,
            "elbow_data": elbow_data,
            "n_clusters": pipeline_results["segmentation"].n_clusters,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/comparison")
async def get_prediction_comparison():
    """Get actual vs predicted price comparison for test set"""
    if not is_trained:
        raise HTTPException(status_code=400, detail="Model not trained.")

    try:
        return {
            "predicted_prices": pipeline_results["data"]["predictions_actual"],
            "actual_prices": pipeline_results["data"]["actual_prices"],
            "metrics": pipeline_results["metrics"]["snn_metrics"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
