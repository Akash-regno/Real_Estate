"""
Configuration settings for the AI Real Estate Analytics System
"""
import os

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(DATA_DIR, "sample_mumbai_housing.csv")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Autoencoder config
AUTOENCODER_CONFIG = {
    "encoding_dim": 8,
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "validation_split": 0.2,
}

# SNN config
SNN_CONFIG = {
    "hidden_size": 64,
    "num_steps": 25,
    "epochs": 150,
    "batch_size": 16,
    "learning_rate": 0.005,
    "beta": 0.95,
}

# Fraud detection config
FRAUD_CONFIG = {
    "isolation_forest_contamination": 0.1,
    "reconstruction_threshold_percentile": 90,
    "n_estimators": 100,
}

# Clustering config
CLUSTERING_CONFIG = {
    "max_clusters": 8,
    "random_state": 42,
}

# Feature columns
AMENITY_COLUMNS = [
    "Gymnasium", "Swimming Pool", "Landscaped Gardens", "Jogging Track",
    "RainWater Harvesting", "Indoor Games", "Shopping Mall", "Intercom",
    "Sports Facility", "ATM", "Club House", "School / University in Township",
    "Hospital / Clinic in Township", "24X7Security", "Power Back up",
    "Car Parking", "Staff Quarter", "Cafeteria", "Multipurpose Room",
    "Hospital within 2 KM", "Locality within 2 KM"
]

NUMERICAL_COLUMNS = [
    "Area", "No. of Bedrooms", "Property Tax",
    "Super built-up Area", "Carpet Area", "Maintenance (INR/Month)"
]

CATEGORICAL_COLUMNS = ["Location", "New/Resale"]
TARGET_COLUMN = "Price"
