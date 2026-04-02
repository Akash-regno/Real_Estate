import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, ScatterChart, Scatter, Legend
} from 'recharts';
import {
  TrendingUp, Home, ShieldAlert, Users, Brain, 
  MapPin, DollarSign, Activity, Layers
} from 'lucide-react';
import { getDatasetStatistics, checkHealth } from '../api';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6', '#06b6d4', '#ec4899', '#14b8a6'];

const formatPrice = (value) => {
  if (value >= 10000000) return `₹${(value / 10000000).toFixed(1)}Cr`;
  if (value >= 100000) return `₹${(value / 100000).toFixed(1)}L`;
  return `₹${value.toLocaleString()}`;
};

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-default)',
        borderRadius: '8px',
        padding: '12px 16px',
        backdropFilter: 'blur(10px)',
      }}>
        <p style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>{label}</p>
        {payload.map((entry, idx) => (
          <p key={idx} style={{ color: entry.color, fontSize: '13px' }}>
            {entry.name}: {typeof entry.value === 'number' ? formatPrice(entry.value) : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, healthRes] = await Promise.all([
          getDatasetStatistics(),
          checkHealth()
        ]);
        setStats(statsRes.data);
        setHealth(healthRes.data);
      } catch (err) {
        console.error('Failed to fetch dashboard data:', err);
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
          <span className="loading-text">Loading analytics dashboard...</span>
        </div>
      </div>
    );
  }

  const priceByLocation = stats?.price_by_location?.slice(0, 10).map(item => ({
    ...item,
    avg_price: Math.round(item.avg_price),
    name: item.location,
  })) || [];

  const priceByBedrooms = stats?.price_by_bedrooms?.map(item => ({
    ...item,
    name: `${item.bedrooms} BHK`,
    avg_price: Math.round(item.avg_price),
  })) || [];

  const conditionData = stats?.condition_distribution
    ? Object.entries(stats.condition_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const amenityData = stats?.amenity_popularity
    ? Object.entries(stats.amenity_popularity)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([name, value]) => ({ 
          name: name.length > 15 ? name.substring(0, 15) + '...' : name, 
          count: value 
        }))
    : [];

  const scatterData = stats?.area_price_scatter?.map(([area, price]) => ({
    area, price
  })) || [];

  return (
    <div className="page-container">
      {/* Hero */}
      <div className="dashboard-hero animate-fade-in-up">
        <div className="hero-content">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <Brain size={32} color="#3b82f6" />
            <span className="card-badge info">AI-Powered</span>
          </div>
          <h1 className="hero-title">Real Estate Analytics</h1>
          <p className="hero-subtitle">
            Intelligent property analysis powered by Deep Autoencoders, Spiking Neural Networks, 
            Isolation Forest fraud detection, and K-Means user segmentation.
          </p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        <div className="stat-card animate-fade-in-up delay-1">
          <div className="stat-icon blue">
            <Home size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Total Properties</div>
            <div className="stat-value">{stats?.total_properties || 0}</div>
            <div className="stat-change positive">Mumbai Dataset</div>
          </div>
        </div>

        <div className="stat-card animate-fade-in-up delay-2">
          <div className="stat-icon green">
            <TrendingUp size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Avg Price</div>
            <div className="stat-value">
              {stats?.price_by_location ? formatPrice(
                stats.price_by_location.reduce((a, b) => a + b.avg_price, 0) / stats.price_by_location.length
              ) : '—'}
            </div>
            <div className="stat-change positive">Across all locations</div>
          </div>
        </div>

        <div className="stat-card animate-fade-in-up delay-3">
          <div className="stat-icon amber">
            <MapPin size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Locations</div>
            <div className="stat-value">{stats?.price_by_location?.length || 0}</div>
            <div className="stat-change positive">Mumbai areas</div>
          </div>
        </div>

        <div className="stat-card animate-fade-in-up delay-4">
          <div className="stat-icon purple">
            <Activity size={22} />
          </div>
          <div className="stat-info">
            <div className="stat-label">Model Status</div>
            <div className="stat-value" style={{ fontSize: '20px' }}>
              {health?.trained ? 'Trained' : 'Pending'}
            </div>
            <div className={`stat-change ${health?.trained ? 'positive' : 'negative'}`}>
              {health?.trained ? '✓ Ready for prediction' : '○ Needs training'}
            </div>
          </div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="charts-grid">
        <div className="chart-card animate-fade-in-up">
          <div className="card-header">
            <h3 className="card-title">Avg Price by Location</h3>
            <span className="card-badge info">Top 10</span>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={priceByLocation} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis type="number" tickFormatter={formatPrice} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={110} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="avg_price" name="Avg Price" fill="url(#blueGradient)" radius={[0, 4, 4, 0]} />
              <defs>
                <linearGradient id="blueGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#3b82f6" />
                  <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card animate-fade-in-up">
          <div className="card-header">
            <h3 className="card-title">Area vs Price</h3>
            <span className="card-badge success">Scatter</span>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis dataKey="area" name="Area (sq ft)" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} label={{ value: 'Area (sq ft)', position: 'bottom', fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis dataKey="price" name="Price" tickFormatter={formatPrice} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
              <Scatter data={scatterData} fill="#3b82f6">
                {scatterData.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx % COLORS.length]} opacity={0.7} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="charts-grid three-col">
        <div className="chart-card animate-fade-in-up">
          <div className="card-header">
            <h3 className="card-title">Price by Bedrooms</h3>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={priceByBedrooms}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <YAxis tickFormatter={formatPrice} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="avg_price" name="Avg Price" fill="url(#greenGradient)" radius={[6, 6, 0, 0]} />
              <defs>
                <linearGradient id="greenGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="100%" stopColor="#059669" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card animate-fade-in-up">
          <div className="card-header">
            <h3 className="card-title">Property Condition</h3>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={conditionData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={95}
                paddingAngle={5}
                dataKey="value"
                animationBegin={200}
                animationDuration={800}
              >
                {conditionData.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx]} stroke="none" />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '12px', color: 'var(--text-secondary)' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card animate-fade-in-up">
          <div className="card-header">
            <h3 className="card-title">Top Amenities</h3>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={amenityData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={100} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" name="Properties" fill="url(#amberGradient)" radius={[0, 4, 4, 0]} />
              <defs>
                <linearGradient id="amberGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#f59e0b" />
                  <stop offset="100%" stopColor="#f43f5e" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
