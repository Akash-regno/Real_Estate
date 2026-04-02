# AI Real Estate Analytics Platform

An intelligent, full-stack real estate analytics solution that provides accurate property price predictions, automated fraud detection, and user/property segmentation. 

Built with a **FastAPI / PyTorch (SNN)** backend and a **React** frontend.

## 🌟 Key Features

1. **Price Prediction Pipeline (SNN)**:
   - Uses a Deep Autoencoder to extract non-linear patterns and reduce feature dimensionality.
   - Leverages a Spiking Neural Network (SNN) built in PyTorch (`snntorch`) to accurately predict housing prices based on comprehensive property features (e.g., location, area, amenities).

2. **Fraud Detection System**:
   - Uses an **Isolation Forest** paired with **Reconstruction Error** evaluation from the Autoencoder.
   - Detects highly anomalous transaction behavior, labeling properties with varying risk scores ("Normal" through "Critical").

3. **Property & User Segmentation**:
   - Employs **K-Means clustering** and **PCA** to classify properties into distinct market segments, aiding targeted recommendations and market evaluations.

4. **Modern React Frontend**:
   - Fully responsive dashboard for visualization of data metrics, insights, and interactive price prediction evaluation.

---

## 🏗️ Architecture

```
Project Root
│
├── backend/                  # FastAPI & PyTorch ML Backend
│   ├── main.py               # REST API Endpoints
│   ├── snn_model.py          # Spiking Neural Network Implementation
│   ├── autoencoder_model.py  # Dimensionality Reduction
│   ├── fraud_detector.py     # Isolation Forest Anomaly Detection 
│   ├── user_segmentation.py  # User Clustering Scripts
│   ├── train_pipeline.py     # End-to-End Orchestrator for ML Training
│   └── data/                 # Sample Data & Generated Assets
│
└── frontend/                 # React UI
    ├── src/                  # React Components & Services
    └── public/               # Static UI Assets
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v16+)
- Python (3.9+)

### 1. Setting up the Backend
```bash
cd backend

# Install the required Python dependencies
pip install -r requirements.txt
# (Make sure tensorflow, torch, and snntorch are installed)

# Run the FastAPI server
python main.py
```
*Note: The backend runs on `http://localhost:8000` and will perform an initial ML pipeline training step upon startup. Do not interrupt this initial configuration.*

### 2. Setting up the Frontend
```bash
cd frontend

# Install the necessary Node/React modules
npm install

# Start the React development frontend
npm start
```
*The frontend will run on `http://localhost:3000` and automatically proxy/connect to your backend endpoints.*

---

## 📖 API Documentation
Once the backend is running, you can access the automatic **Swagger UI** documentation at `http://localhost:8000/docs` or the **ReDoc** UI at `http://localhost:8000/redoc`.

---

## 🛠️ Built With

* **Frontend**: React, React Router, Recharts, Radix UI, Axios, Tailwind CSS
* **Backend**: FastAPI, Uvicorn, Pydantic
* **Machine Learning**: PyTorch, snntorch, TensorFlow (Autoencoder), Scikit-Learn, Pandas, NumPy
