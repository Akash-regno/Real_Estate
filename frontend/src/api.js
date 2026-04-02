import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 min timeout for training
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const checkHealth = () => api.get('/health');

// Dataset
export const getDatasetInfo = () => api.get('/dataset/info');
export const getDatasetStatistics = () => api.get('/dataset/statistics');

// Training
export const trainModel = (csvPath = null) => 
  api.post('/train', { csv_path: csvPath });

// Prediction
export const predictPrice = (data) => api.post('/predict', data);

// Model metrics
export const getModelMetrics = () => api.get('/model/metrics');

// Fraud detection
export const getFraudResults = () => api.get('/fraud/results');

// Segmentation
export const getSegmentationResults = () => api.get('/segmentation/results');

// Prediction comparison
export const getPredictionComparison = () => api.get('/predictions/comparison');

export default api;
