"""
User Segmentation Module using Clustering
- K-Means clustering for buyer segmentation
- Elbow method for optimal cluster selection
- Segment profiling and characterization
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import os
from config import CLUSTERING_CONFIG, MODEL_DIR


class UserSegmentation:
    def __init__(self):
        self.kmeans = None
        self.n_clusters = None
        self.pca = None
        self.cluster_profiles = {}
        self.is_fitted = False

    def find_optimal_clusters(self, X, max_clusters=None):
        """Find optimal number of clusters using elbow method and silhouette score"""
        max_clusters = max_clusters or CLUSTERING_CONFIG["max_clusters"]
        max_clusters = min(max_clusters, len(X) - 1)

        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=CLUSTERING_CONFIG["random_state"],
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(X)
            inertias.append(float(kmeans.inertia_))
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(float(sil_score))

        # Find optimal k based on silhouette score
        optimal_idx = np.argmax(silhouette_scores)
        self.n_clusters = list(k_range)[optimal_idx]

        print(f"[UserSegmentation] Optimal clusters: {self.n_clusters} "
              f"(silhouette: {silhouette_scores[optimal_idx]:.4f})")

        return {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "optimal_k": self.n_clusters,
            "optimal_silhouette": silhouette_scores[optimal_idx]
        }

    def fit(self, X, n_clusters=None):
        """Fit K-Means clustering"""
        if n_clusters:
            self.n_clusters = n_clusters

        if self.n_clusters is None:
            self.find_optimal_clusters(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=CLUSTERING_CONFIG["random_state"],
            n_init=10,
            max_iter=300
        )
        labels = self.kmeans.fit_predict(X)

        # Fit PCA for 2D visualization
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X)

        self.is_fitted = True
        print(f"[UserSegmentation] Fitted {self.n_clusters} clusters on {len(X)} samples")

        return labels, X_pca

    def predict(self, X):
        """Predict cluster assignments"""
        return self.kmeans.predict(X)

    def get_cluster_centers(self):
        """Return cluster centers"""
        if self.kmeans is None:
            return []
        return self.kmeans.cluster_centers_.tolist()

    def profile_clusters(self, df, labels, feature_columns):
        """Generate cluster profiles with statistics"""
        df = df.copy()
        df['Cluster'] = labels

        profiles = {}
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id]
            profile = {
                "cluster_id": cluster_id,
                "size": int(len(cluster_data)),
                "percentage": float(len(cluster_data) / len(df) * 100),
                "features": {}
            }

            for col in feature_columns:
                if col in cluster_data.columns:
                    profile["features"][col] = {
                        "mean": float(cluster_data[col].mean()),
                        "median": float(cluster_data[col].median()),
                        "std": float(cluster_data[col].std()) if len(cluster_data) > 1 else 0.0,
                        "min": float(cluster_data[col].min()),
                        "max": float(cluster_data[col].max()),
                    }

            # Price statistics (use original price)
            if 'Price' in cluster_data.columns:
                profile["price_stats"] = {
                    "mean": float(cluster_data['Price'].mean()),
                    "median": float(cluster_data['Price'].median()),
                    "min": float(cluster_data['Price'].min()),
                    "max": float(cluster_data['Price'].max()),
                }

            # Location distribution
            if 'Location' in cluster_data.columns:
                loc_dist = cluster_data['Location'].value_counts().head(5).to_dict()
                profile["top_locations"] = {str(k): int(v) for k, v in loc_dist.items()}

            # Bedroom distribution
            if 'No. of Bedrooms' in cluster_data.columns:
                bed_dist = cluster_data['No. of Bedrooms'].value_counts().to_dict()
                profile["bedroom_distribution"] = {str(k): int(v) for k, v in bed_dist.items()}

            # Segment label
            if 'Price' in cluster_data.columns:
                avg_price = cluster_data['Price'].mean()
                avg_area = cluster_data['Area'].mean() if 'Area' in cluster_data.columns else 0
                if avg_price > 20000000:
                    profile["segment_label"] = "Premium Buyers"
                elif avg_price > 10000000:
                    profile["segment_label"] = "Mid-Range Buyers"
                elif avg_price > 5000000:
                    profile["segment_label"] = "Budget Buyers"
                else:
                    profile["segment_label"] = "Economy Buyers"
            else:
                profile["segment_label"] = f"Segment {cluster_id + 1}"

            profiles[cluster_id] = profile

        self.cluster_profiles = profiles
        return profiles

    def get_pca_data(self, X, labels):
        """Get PCA-transformed data for visualization"""
        if self.pca is None:
            self.pca = PCA(n_components=2)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)

        return {
            "x": X_pca[:, 0].tolist(),
            "y": X_pca[:, 1].tolist(),
            "labels": labels.tolist() if hasattr(labels, 'tolist') else list(labels),
            "explained_variance": self.pca.explained_variance_ratio_.tolist()
        }

    def save_model(self):
        """Save clustering models"""
        joblib.dump(self.kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))
        joblib.dump(self.pca, os.path.join(MODEL_DIR, "pca.pkl"))
        joblib.dump(self.cluster_profiles, os.path.join(MODEL_DIR, "cluster_profiles.pkl"))
        print("[UserSegmentation] Models saved")

    def load_model(self):
        """Load clustering models"""
        self.kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
        self.pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
        self.cluster_profiles = joblib.load(os.path.join(MODEL_DIR, "cluster_profiles.pkl"))
        self.n_clusters = self.kmeans.n_clusters
        self.is_fitted = True
        print("[UserSegmentation] Models loaded")
