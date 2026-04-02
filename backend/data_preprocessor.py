"""
Data Preprocessing Module
- Data cleaning, handling missing values, removing duplicates
- Encoding categorical variables
- Normalizing numerical features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from config import (
    CSV_PATH, MODEL_DIR, AMENITY_COLUMNS,
    NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN
)


class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = []
        self.raw_df = None
        self.processed_df = None

    def load_data(self, filepath=None):
        """Load CSV data"""
        path = filepath or CSV_PATH
        self.raw_df = pd.read_csv(path)
        print(f"[DataPreprocessor] Loaded {len(self.raw_df)} records with {len(self.raw_df.columns)} columns")
        return self.raw_df

    def clean_data(self, df=None):
        """Clean data: remove duplicates, handle missing values"""
        if df is None:
            df = self.raw_df.copy()

        initial_len = len(df)

        # Remove duplicates
        df = df.drop_duplicates()
        print(f"[DataPreprocessor] Removed {initial_len - len(df)} duplicates")

        # Handle missing values
        for col in NUMERICAL_COLUMNS + [TARGET_COLUMN]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                missing = df[col].isna().sum()
                if missing > 0:
                    df[col].fillna(median_val, inplace=True)
                    print(f"[DataPreprocessor] Filled {missing} missing values in '{col}' with median ({median_val})")

        for col in AMENITY_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                missing = df[col].isna().sum()
                if missing > 0:
                    df[col].fillna(mode_val, inplace=True)
                    print(f"[DataPreprocessor] Filled {missing} missing values in '{col}' with mode ({mode_val})")

        # Remove rows with zero or negative prices
        df = df[df[TARGET_COLUMN] > 0]

        print(f"[DataPreprocessor] Clean data: {len(df)} records remaining")
        self.processed_df = df
        return df

    def encode_categorical(self, df=None):
        """Encode categorical variables using LabelEncoder"""
        if df is None:
            df = self.processed_df.copy()

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"[DataPreprocessor] Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

        self.processed_df = df
        return df

    def normalize_features(self, df=None):
        """Normalize numerical features using MinMaxScaler"""
        if df is None:
            df = self.processed_df.copy()

        # Determine feature columns
        self.feature_columns = []
        for col in NUMERICAL_COLUMNS:
            if col in df.columns:
                self.feature_columns.append(col)
        for col in AMENITY_COLUMNS:
            if col in df.columns:
                self.feature_columns.append(col)
        for col in CATEGORICAL_COLUMNS:
            enc_col = f"{col}_encoded"
            if enc_col in df.columns:
                self.feature_columns.append(enc_col)

        # Normalize features
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])

        # Normalize target separately
        df[[f"{TARGET_COLUMN}_normalized"]] = self.target_scaler.fit_transform(df[[TARGET_COLUMN]])

        self.processed_df = df
        print(f"[DataPreprocessor] Normalized {len(self.feature_columns)} features")
        return df

    def get_feature_matrix(self):
        """Return feature matrix X and target y"""
        X = self.processed_df[self.feature_columns].values
        y = self.processed_df[f"{TARGET_COLUMN}_normalized"].values
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split into train/test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"[DataPreprocessor] Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def inverse_transform_price(self, normalized_price):
        """Convert normalized price back to actual price"""
        return self.target_scaler.inverse_transform(
            np.array(normalized_price).reshape(-1, 1)
        ).flatten()

    def save_preprocessor(self):
        """Save fitted scalers and encoders"""
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "feature_scaler.pkl"))
        joblib.dump(self.target_scaler, os.path.join(MODEL_DIR, "target_scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
        joblib.dump(self.feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))
        print("[DataPreprocessor] Saved preprocessor artifacts")

    def load_preprocessor(self):
        """Load fitted scalers and encoders"""
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
        self.target_scaler = joblib.load(os.path.join(MODEL_DIR, "target_scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
        self.feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
        print("[DataPreprocessor] Loaded preprocessor artifacts")

    def preprocess_pipeline(self, filepath=None):
        """Run full preprocessing pipeline"""
        self.load_data(filepath)
        self.clean_data()
        self.encode_categorical()
        self.normalize_features()
        self.save_preprocessor()
        X, y = self.get_feature_matrix()
        return X, y, self.processed_df
