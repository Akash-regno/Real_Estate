import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, ScatterChart, Scatter, Legend
} from 'recharts';
import { ShieldAlert, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { getFraudResults } from '../api';

const RISK_COLORS = {
  Normal: '#10b981',
  Low: '#3b82f6',
  Medium: '#f59e0b',
  High: '#f43f5e',
  Critical: '#dc2626',
};

const formatPrice = (value) => {
  if (value >= 10000000) return `₹${(value / 10000000).toFixed(1)}Cr`;
  if (value >= 100000) return `₹${(value / 100000).toFixed(1)}L`;
  return `₹${value.toLocaleString()}`;
};

export default function FraudDetection() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await getFraudResults();
        setData(res.data);
      } catch (err) {
        console.error('Failed to fetch fraud results:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-container">
          <div className="spinner"></div>
          <span className="loading-text">Analyzing transactions for anomalies...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="page-container">
        <div className="empty-state">
          <ShieldAlert size={64} className="empty-state-icon" />
          <h3 className="empty-state-title">No Fraud Data Available</h3>
          <p className="empty-state-text">Train the model first to see fraud detection results.</p>
        </div>
      </div>
    );
  }

  const stats = data.statistics;
  const categoryDist = stats.category_distribution;
  const pieData = Object.entries(categoryDist).map(([name, value]) => ({ name, value }));
  
  const filteredResults = filter === 'all' 
    ? data.results 
    : data.results.filter(r => r.category.toLowerCase() === filter);

  const riskDistribution = data.results.map(r => ({
    price: r.price,
    risk: parseFloat((r.combined_risk * 100).toFixed(1)),
    category: r.category,
    location: r.location,
  }));

  return (
    <div className="page-container">
      <div className="page-header animate-fade-in-up">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="stat-icon rose">
            <ShieldAlert size={22} />
          </div>
          <div>
            <h1 className="page-title">Fraud Detection</h1>
            <p className="page-subtitle">Unsupervised anomaly detection using Isolation Forest & Autoencoder reconstruction error</p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card animate-fade-in-up delay-1">
          <div className="stat-icon green">
            <CheckCircle size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Normal Transactions</div>
            <div className="stat-value">{(categoryDist.Normal || 0) + (categoryDist.Low || 0)}</div>
            <div className="stat-change positive">Low risk</div>
          </div>
        </div>

        <div className="stat-card animate-fade-in-up delay-2">
          <div className="stat-icon amber">
            <AlertTriangle size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Medium Risk</div>
            <div className="stat-value">{categoryDist.Medium || 0}</div>
            <div className="stat-change" style={{ color: 'var(--warning-400)' }}>Needs review</div>
          </div>
        </div>

        <div className="stat-card animate-fade-in-up delay-3">
          <div className="stat-icon rose">
            <XCircle size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Flagged</div>
            <div className="stat-value" style={{ color: 'var(--danger-400)' }}>{stats.flagged_count}</div>
            <div className="stat-change negative">{stats.flagged_percentage.toFixed(1)}% suspicious</div>
          </div>
        </div>

        <div className="stat-card animate-fade-in-up delay-4">
          <div className="stat-icon purple">
            <ShieldAlert size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Avg Risk Score</div>
            <div className="stat-value">{(stats.mean_risk_score * 100).toFixed(1)}%</div>
            <div className="stat-change">σ = {(stats.std_risk_score * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="charts-grid">
        <div className="chart-card animate-fade-in-up">
          <div className="card-header">
            <h3 className="card-title">Risk Category Distribution</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={4}
                dataKey="value"
                animationDuration={800}
              >
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={RISK_COLORS[entry.name]} stroke="none" />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-default)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card animate-fade-in-up">
          <div className="card-header">
            <h3 className="card-title">Price vs Risk Score</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis dataKey="price" name="Price" tickFormatter={formatPrice} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis dataKey="risk" name="Risk %" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border-default)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
                formatter={(value, name) => {
                  if (name === 'Price') return formatPrice(value);
                  return `${value}%`;
                }}
              />
              <Scatter data={riskDistribution}>
                {riskDistribution.map((entry, idx) => (
                  <Cell key={idx} fill={RISK_COLORS[entry.category]} opacity={0.8} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Table */}
      <div className="card animate-fade-in-up">
        <div className="card-header">
          <h3 className="card-title">Transaction Analysis</h3>
          <div style={{ display: 'flex', gap: '6px' }}>
            {['all', 'critical', 'high', 'medium', 'low', 'normal'].map(f => (
              <button
                key={f}
                className={`toggle-chip ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Location</th>
                <th>Price</th>
                <th>Area</th>
                <th>BHK</th>
                <th>IF Risk</th>
                <th>AE Risk</th>
                <th>Combined</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredResults.map((row, idx) => (
                <tr key={row.index}>
                  <td>{row.index + 1}</td>
                  <td style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{row.location}</td>
                  <td>{formatPrice(row.price)}</td>
                  <td>{row.area} sqft</td>
                  <td>{row.bedrooms}</td>
                  <td>{(row.if_risk * 100).toFixed(1)}%</td>
                  <td>{(row.recon_risk * 100).toFixed(1)}%</td>
                  <td style={{ fontWeight: 600 }}>{(row.combined_risk * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`risk-badge ${row.category.toLowerCase()}`}>
                      {row.category}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
