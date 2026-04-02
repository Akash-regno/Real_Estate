import React, { useState, useEffect } from 'react';
import { Database, Search, Filter } from 'lucide-react';
import { getDatasetInfo } from '../api';

const fmt = v => typeof v==='number'?(v>=1e7?`₹${(v/1e7).toFixed(1)}Cr`:v>=1e5?`₹${(v/1e5).toFixed(1)}L`:`₹${v.toLocaleString()}`):v;

export default function DataExplorer() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    getDatasetInfo().then(r => setInfo(r.data)).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-container"><div className="loading-container"><div className="spinner"></div><span className="loading-text">Loading dataset...</span></div></div>;
  if (!info) return <div className="page-container"><div className="empty-state"><Database size={64} className="empty-state-icon"/><h3 className="empty-state-title">No Dataset</h3></div></div>;

  const filteredSamples = info.sample_data?.filter(row =>
    !search || Object.values(row).some(v => String(v).toLowerCase().includes(search.toLowerCase()))
  ) || [];

  return (
    <div className="page-container">
      <div className="page-header animate-fade-in-up">
        <div style={{display:'flex',alignItems:'center',gap:12}}>
          <div className="stat-icon blue"><Database size={22}/></div>
          <div><h1 className="page-title">Data Explorer</h1><p className="page-subtitle">Mumbai Housing Dataset — {info.rows} records, {info.columns} features</p></div>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card animate-fade-in-up delay-1">
          <div className="stat-icon blue"><Database size={22}/></div>
          <div className="stat-info"><div className="stat-label">Records</div><div className="stat-value">{info.rows}</div></div>
        </div>
        <div className="stat-card animate-fade-in-up delay-2">
          <div className="stat-icon green"><Filter size={22}/></div>
          <div className="stat-info"><div className="stat-label">Features</div><div className="stat-value">{info.columns}</div></div>
        </div>
        <div className="stat-card animate-fade-in-up delay-3">
          <div className="stat-icon amber" style={{fontSize:18,fontWeight:700}}>📍</div>
          <div className="stat-info"><div className="stat-label">Locations</div><div className="stat-value">{info.locations?.length || 0}</div></div>
        </div>
        <div className="stat-card animate-fade-in-up delay-4">
          <div className="stat-icon purple" style={{fontSize:18,fontWeight:700}}>🛏</div>
          <div className="stat-info"><div className="stat-label">BHK Range</div><div className="stat-value">{info.bedroom_counts?.join(', ')}</div></div>
        </div>
      </div>

      {/* Price Stats */}
      <div className="card animate-fade-in-up" style={{marginBottom:24}}>
        <div className="card-header"><h3 className="card-title">Price Statistics</h3><span className="card-badge info">INR</span></div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit, minmax(180px, 1fr))',gap:12}}>
          {info.price_stats && Object.entries(info.price_stats).map(([k,v]) => (
            <div key={k} className="result-detail-item">
              <div className="result-detail-label">{k.replace('_',' ')}</div>
              <div className="result-detail-value">{fmt(v)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Column Info */}
      <div className="card animate-fade-in-up" style={{marginBottom:24}}>
        <div className="card-header"><h3 className="card-title">Column Information</h3></div>
        <div className="table-wrapper" style={{maxHeight:300}}>
          <table className="data-table">
            <thead><tr><th>Column</th><th>Data Type</th><th>Missing Values</th></tr></thead>
            <tbody>
              {info.column_names?.map(col => (
                <tr key={col}>
                  <td style={{color:'var(--text-primary)',fontWeight:500}}>{col}</td>
                  <td>{info.dtypes?.[col] || '—'}</td>
                  <td>{info.missing_values?.[col] || 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Sample Data */}
      <div className="card animate-fade-in-up">
        <div className="card-header">
          <h3 className="card-title">Sample Data</h3>
          <div style={{position:'relative'}}>
            <Search size={14} style={{position:'absolute',left:10,top:'50%',transform:'translateY(-50%)',color:'var(--text-muted)'}}/>
            <input type="text" value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search..." className="form-input" style={{paddingLeft:32,width:200}}/>
          </div>
        </div>
        <div className="table-wrapper">
          <table className="data-table">
            <thead><tr>{info.column_names?.slice(0,8).map(c => <th key={c}>{c.length>12?c.substring(0,12)+'…':c}</th>)}</tr></thead>
            <tbody>
              {filteredSamples.map((row, i) => (
                <tr key={i}>{info.column_names?.slice(0,8).map(c => (
                  <td key={c}>{c==='Price'?fmt(row[c]):String(row[c]||'—').substring(0,20)}</td>
                ))}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
