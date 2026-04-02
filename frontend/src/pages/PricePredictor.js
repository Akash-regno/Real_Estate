import React, { useState, useEffect } from 'react';
import { Target, MapPin, Home, DollarSign, Shield, Users, Sparkles } from 'lucide-react';
import { predictPrice, getDatasetInfo } from '../api';

const AMENITY_LIST = [
  { key: 'gymnasium', label: 'Gymnasium' },
  { key: 'swimming_pool', label: 'Swimming Pool' },
  { key: 'landscaped_gardens', label: 'Gardens' },
  { key: 'jogging_track', label: 'Jogging Track' },
  { key: 'rainwater_harvesting', label: 'Rainwater Harvesting' },
  { key: 'indoor_games', label: 'Indoor Games' },
  { key: 'shopping_mall', label: 'Shopping Mall' },
  { key: 'intercom', label: 'Intercom' },
  { key: 'sports_facility', label: 'Sports Facility' },
  { key: 'atm', label: 'ATM' },
  { key: 'club_house', label: 'Club House' },
  { key: 'school', label: 'School Nearby' },
  { key: 'hospital_township', label: 'Hospital in Township' },
  { key: 'security_24x7', label: '24x7 Security' },
  { key: 'power_backup', label: 'Power Backup' },
  { key: 'car_parking', label: 'Car Parking' },
  { key: 'staff_quarter', label: 'Staff Quarter' },
  { key: 'cafeteria', label: 'Cafeteria' },
  { key: 'multipurpose_room', label: 'Multipurpose Room' },
  { key: 'hospital_2km', label: 'Hospital within 2 KM' },
  { key: 'locality_2km', label: 'Locality within 2 KM' },
];

export default function PricePredictor() {
  const [locations, setLocations] = useState([]);
  const [formData, setFormData] = useState({
    area: 1200,
    bedrooms: 2,
    location: 'Andheri West',
    new_resale: 'New',
    property_tax: 50000,
    super_built_up_area: 0,
    carpet_area: 0,
    maintenance: 3000,
  });
  const [amenities, setAmenities] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        const res = await getDatasetInfo();
        setLocations(res.data.locations || []);
        if (res.data.locations?.length) {
          setFormData(prev => ({ ...prev, location: res.data.locations[0] }));
        }
      } catch (err) {
        console.error('Failed to fetch locations:', err);
      }
    };
    fetchLocations();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['area', 'bedrooms', 'property_tax', 'super_built_up_area', 'carpet_area', 'maintenance'].includes(name)
        ? parseFloat(value) || 0
        : value
    }));
  };

  const toggleAmenity = (key) => {
    setAmenities(prev => ({
      ...prev,
      [key]: prev[key] ? 0 : 1
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const payload = { ...formData, ...amenities };
      const res = await predictPrice(payload);
      setPrediction(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Ensure the model is trained.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (category) => {
    const colors = {
      'Normal': 'var(--accent-400)',
      'Low': 'var(--primary-400)',
      'Medium': 'var(--warning-400)',
      'High': 'var(--danger-400)',
      'Critical': 'var(--danger-500)',
    };
    return colors[category] || 'var(--text-secondary)';
  };

  return (
    <div className="page-container">
      <div className="page-header animate-fade-in-up">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="stat-icon blue">
            <Target size={22} />
          </div>
          <div>
            <h1 className="page-title">Price Prediction</h1>
            <p className="page-subtitle">SNN-powered property price estimation with fraud risk assessment</p>
          </div>
        </div>
      </div>

      <div className="card animate-fade-in-up" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <h3 className="card-title">Property Details</h3>
          <span className="card-badge info">
            <Sparkles size={12} style={{ marginRight: '4px' }} />
            AI Prediction
          </span>
        </div>

        <div className="form-grid" style={{ marginBottom: '20px' }}>
          <div className="form-group">
            <label className="form-label">Location</label>
            <select
              name="location"
              value={formData.location}
              onChange={handleInputChange}
              className="form-select"
            >
              {locations.map(loc => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Area (sq ft)</label>
            <input
              type="number"
              name="area"
              value={formData.area}
              onChange={handleInputChange}
              className="form-input"
              placeholder="e.g. 1200"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Bedrooms</label>
            <select
              name="bedrooms"
              value={formData.bedrooms}
              onChange={handleInputChange}
              className="form-select"
            >
              {[1, 2, 3, 4, 5, 6].map(n => (
                <option key={n} value={n}>{n} BHK</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Condition</label>
            <select
              name="new_resale"
              value={formData.new_resale}
              onChange={handleInputChange}
              className="form-select"
            >
              <option value="New">New</option>
              <option value="Resale">Resale</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Property Tax (₹/year)</label>
            <input
              type="number"
              name="property_tax"
              value={formData.property_tax}
              onChange={handleInputChange}
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Maintenance (₹/month)</label>
            <input
              type="number"
              name="maintenance"
              value={formData.maintenance}
              onChange={handleInputChange}
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Super Built-up Area</label>
            <input
              type="number"
              name="super_built_up_area"
              value={formData.super_built_up_area}
              onChange={handleInputChange}
              className="form-input"
              placeholder="Auto-calculated if 0"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Carpet Area</label>
            <input
              type="number"
              name="carpet_area"
              value={formData.carpet_area}
              onChange={handleInputChange}
              className="form-input"
              placeholder="Auto-calculated if 0"
            />
          </div>
        </div>

        {/* Amenities Toggle */}
        <div style={{ marginBottom: '24px' }}>
          <label className="form-label" style={{ marginBottom: '10px', display: 'block' }}>
            Select Amenities
          </label>
          <div className="toggle-group">
            {AMENITY_LIST.map(({ key, label }) => (
              <button
                key={key}
                className={`toggle-chip ${amenities[key] ? 'active' : ''}`}
                onClick={() => toggleAmenity(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <button
          className="btn btn-primary btn-lg"
          onClick={handlePredict}
          disabled={loading}
          style={{ width: '100%' }}
        >
          {loading ? (
            <>
              <div className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }}></div>
              Processing with SNN...
            </>
          ) : (
            <>
              <Target size={18} />
              Predict Property Price
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="card animate-fade-in-up" style={{ 
          borderColor: 'rgba(244, 63, 94, 0.3)', 
          background: 'rgba(244, 63, 94, 0.05)',
          marginBottom: '24px' 
        }}>
          <p style={{ color: 'var(--danger-400)', fontSize: '14px' }}>⚠ {error}</p>
        </div>
      )}

      {/* Results */}
      {prediction && (
        <div className="result-card animate-fade-in-up">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <div className="stat-icon green">
              <DollarSign size={22} />
            </div>
            <div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Predicted Price
              </div>
              <div className="result-price">{prediction.predicted_price_formatted}</div>
            </div>
          </div>

          <div className="result-details">
            <div className="result-detail-item">
              <div className="result-detail-label">
                <MapPin size={12} style={{ display: 'inline', marginRight: '4px' }} />
                Location
              </div>
              <div className="result-detail-value">{formData.location}</div>
            </div>

            <div className="result-detail-item">
              <div className="result-detail-label">
                <Home size={12} style={{ display: 'inline', marginRight: '4px' }} />
                Property
              </div>
              <div className="result-detail-value">{formData.bedrooms} BHK, {formData.area} sq ft</div>
            </div>

            <div className="result-detail-item">
              <div className="result-detail-label">
                <Shield size={12} style={{ display: 'inline', marginRight: '4px' }} />
                Fraud Risk
              </div>
              <div className="result-detail-value" style={{ color: getRiskColor(prediction.fraud_category) }}>
                {prediction.fraud_category} ({(prediction.fraud_risk_score * 100).toFixed(1)}%)
              </div>
            </div>

            <div className="result-detail-item">
              <div className="result-detail-label">
                <Users size={12} style={{ display: 'inline', marginRight: '4px' }} />
                Buyer Segment
              </div>
              <div className="result-detail-value">{prediction.segment_label}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
