import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, Target, ShieldAlert, Users, Database, 
  Activity, Brain, TrendingUp 
} from 'lucide-react';
import Dashboard from './pages/Dashboard';
import PricePredictor from './pages/PricePredictor';
import FraudDetection from './pages/FraudDetection';
import UserSegments from './pages/UserSegments';
import DataExplorer from './pages/DataExplorer';
import ModelMetrics from './pages/ModelMetrics';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-container">
        {/* Sidebar Navigation */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-logo">
              <div className="sidebar-logo-icon">
                <Brain size={22} color="white" />
              </div>
              <div>
                <div className="sidebar-logo-text">RealEstateAI</div>
                <div className="sidebar-logo-subtitle">Analytics Platform</div>
              </div>
            </div>
          </div>

          <nav className="sidebar-nav">
            <span className="nav-section-label">Overview</span>
            <NavLink to="/" end className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <LayoutDashboard className="nav-link-icon" />
              Dashboard
            </NavLink>

            <span className="nav-section-label">AI Modules</span>
            <NavLink to="/predict" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Target className="nav-link-icon" />
              Price Prediction
            </NavLink>
            <NavLink to="/fraud" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <ShieldAlert className="nav-link-icon" />
              Fraud Detection
            </NavLink>
            <NavLink to="/segments" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Users className="nav-link-icon" />
              User Segments
            </NavLink>

            <span className="nav-section-label">Data & Models</span>
            <NavLink to="/data" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Database className="nav-link-icon" />
              Data Explorer
            </NavLink>
            <NavLink to="/metrics" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
              <Activity className="nav-link-icon" />
              Model Metrics
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            <div className="training-status">
              <span className="status-dot online"></span>
              <div>
                <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>System Online</div>
                <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>SNN + Autoencoder</div>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<PricePredictor />} />
            <Route path="/fraud" element={<FraudDetection />} />
            <Route path="/segments" element={<UserSegments />} />
            <Route path="/data" element={<DataExplorer />} />
            <Route path="/metrics" element={<ModelMetrics />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
