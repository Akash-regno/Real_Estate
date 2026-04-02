import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, Cell, Legend, BarChart, Bar
} from 'recharts';
import { Activity, Brain, Cpu, Zap } from 'lucide-react';
import { getModelMetrics, getPredictionComparison } from '../api';

const fmt = v => v>=1e7?`₹${(v/1e7).toFixed(1)}Cr`:v>=1e5?`₹${(v/1e5).toFixed(1)}L`:`₹${v.toLocaleString()}`;
const ttStyle = {background:'var(--bg-card)',border:'1px solid var(--border-default)',borderRadius:'8px',color:'var(--text-primary)'};

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    Promise.all([getModelMetrics(), getPredictionComparison()])
      .then(([m,c]) => { setMetrics(m.data); setComparison(c.data); })
      .catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-container"><div className="loading-container"><div className="spinner"></div><span className="loading-text">Loading model metrics...</span></div></div>;
  if (!metrics) return <div className="page-container"><div className="empty-state"><Activity size={64} className="empty-state-icon"/><h3 className="empty-state-title">No Metrics</h3><p className="empty-state-text">Train the model first.</p></div></div>;

  const snn = metrics.snn_metrics;
  const aeH = metrics.ae_history;
  const snnH = metrics.snn_history;

  // Training loss data
  const aeLoss = (aeH.loss||[]).map((v,i) => ({ epoch:i+1, loss:+v.toFixed(6), val_loss:aeH.val_loss?.[i]?+aeH.val_loss[i].toFixed(6):null }));
  const snnLoss = (snnH.train_loss||[]).map((v,i) => ({ epoch:i+1, loss:+v.toFixed(6), val_loss:snnH.val_loss?.[i]?+snnH.val_loss[i].toFixed(6):null }));

  // Prediction comparison
  const compData = comparison ? comparison.actual_prices.map((a,i) => ({
    actual:Math.round(a), predicted:Math.round(comparison.predicted_prices[i]),
    error:Math.abs(a-comparison.predicted_prices[i]),
    idx:i+1
  })) : [];

  return (
    <div className="page-container">
      <div className="page-header animate-fade-in-up">
        <div style={{display:'flex',alignItems:'center',gap:12}}>
          <div className="stat-icon purple"><Activity size={22}/></div>
          <div><h1 className="page-title">Model Performance</h1><p className="page-subtitle">Training metrics, loss curves, and prediction analysis</p></div>
        </div>
      </div>

      {/* Metric Stats */}
      <div className="stats-grid">
        <div className="stat-card animate-fade-in-up delay-1">
          <div className="stat-icon blue"><Zap size={22}/></div>
          <div className="stat-info"><div className="stat-label">R² Score</div><div className="stat-value">{(snn.r2_score*100).toFixed(1)}%</div><div className="stat-change positive">Coefficient of determination</div></div>
        </div>
        <div className="stat-card animate-fade-in-up delay-2">
          <div className="stat-icon green"><Activity size={22}/></div>
          <div className="stat-info"><div className="stat-label">RMSE</div><div className="stat-value">{snn.rmse.toFixed(4)}</div><div className="stat-change">Normalized scale</div></div>
        </div>
        <div className="stat-card animate-fade-in-up delay-3">
          <div className="stat-icon amber"><Brain size={22}/></div>
          <div className="stat-info"><div className="stat-label">MAE</div><div className="stat-value">{snn.mae.toFixed(4)}</div><div className="stat-change">Mean absolute error</div></div>
        </div>
        <div className="stat-card animate-fade-in-up delay-4">
          <div className="stat-icon purple"><Cpu size={22}/></div>
          <div className="stat-info"><div className="stat-label">SNN Params</div><div className="stat-value">{metrics.snn_info?.total_params?.toLocaleString()}</div><div className="stat-change">{metrics.snn_info?.uses_snntorch?'snntorch LIF':'MLP fallback'}</div></div>
        </div>
      </div>

      {/* Tabs */}
      <div className="tab-bar">
        {['overview','autoencoder','snn','comparison'].map(t => (
          <button key={t} className={`tab-button ${tab===t?'active':''}`} onClick={()=>setTab(t)}>
            {t==='overview'?'Overview':t==='autoencoder'?'Autoencoder':t==='snn'?'SNN Training':'Predictions'}
          </button>
        ))}
      </div>

      {tab==='overview' && (
        <div className="charts-grid animate-fade-in">
          <div className="chart-card">
            <div className="card-header"><h3 className="card-title">Autoencoder Loss</h3><span className="card-badge info">{aeLoss.length} epochs</span></div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={aeLoss}><CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)"/>
                <XAxis dataKey="epoch" tick={{fill:'var(--text-muted)',fontSize:11}}/><YAxis tick={{fill:'var(--text-muted)',fontSize:11}}/>
                <Tooltip contentStyle={ttStyle}/><Legend wrapperStyle={{fontSize:12}}/>
                <Line type="monotone" dataKey="loss" name="Train" stroke="#3b82f6" strokeWidth={2} dot={false}/>
                <Line type="monotone" dataKey="val_loss" name="Validation" stroke="#f59e0b" strokeWidth={2} dot={false}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-card">
            <div className="card-header"><h3 className="card-title">SNN Loss</h3><span className="card-badge success">{snnLoss.length} epochs</span></div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={snnLoss}><CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)"/>
                <XAxis dataKey="epoch" tick={{fill:'var(--text-muted)',fontSize:11}}/><YAxis tick={{fill:'var(--text-muted)',fontSize:11}}/>
                <Tooltip contentStyle={ttStyle}/><Legend wrapperStyle={{fontSize:12}}/>
                <Line type="monotone" dataKey="loss" name="Train" stroke="#10b981" strokeWidth={2} dot={false}/>
                <Line type="monotone" dataKey="val_loss" name="Validation" stroke="#f43f5e" strokeWidth={2} dot={false}/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {tab==='autoencoder' && (
        <div className="card animate-fade-in">
          <div className="card-header"><h3 className="card-title">Autoencoder Architecture</h3></div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))',gap:12,marginBottom:20}}>
            <div className="result-detail-item"><div className="result-detail-label">Input Dim</div><div className="result-detail-value">{metrics.ae_info?.input_dim}</div></div>
            <div className="result-detail-item"><div className="result-detail-label">Encoding Dim</div><div className="result-detail-value">{metrics.ae_info?.encoding_dim}</div></div>
            <div className="result-detail-item"><div className="result-detail-label">Total Params</div><div className="result-detail-value">{metrics.ae_info?.total_params?.toLocaleString()}</div></div>
          </div>
          {metrics.ae_info?.architecture && (
            <div className="table-wrapper"><table className="data-table"><thead><tr><th>Layer</th><th>Output Shape</th><th>Parameters</th></tr></thead><tbody>
              {metrics.ae_info.architecture.map((l,i) => <tr key={i}><td style={{color:'var(--text-primary)',fontWeight:500}}>{l.layer}</td><td>{l.output_shape}</td><td>{l.params.toLocaleString()}</td></tr>)}
            </tbody></table></div>
          )}
        </div>
      )}

      {tab==='snn' && (
        <div className="card animate-fade-in">
          <div className="card-header"><h3 className="card-title">SNN Architecture</h3></div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))',gap:12}}>
            <div className="result-detail-item"><div className="result-detail-label">Hidden Size</div><div className="result-detail-value">{metrics.snn_info?.hidden_size}</div></div>
            <div className="result-detail-item"><div className="result-detail-label">Time Steps</div><div className="result-detail-value">{metrics.snn_info?.num_steps}</div></div>
            <div className="result-detail-item"><div className="result-detail-label">Device</div><div className="result-detail-value">{metrics.snn_info?.device}</div></div>
            <div className="result-detail-item"><div className="result-detail-label">Framework</div><div className="result-detail-value">{metrics.snn_info?.uses_snntorch?'snntorch (LIF)':'PyTorch MLP'}</div></div>
            <div className="result-detail-item"><div className="result-detail-label">Trainable Params</div><div className="result-detail-value">{metrics.snn_info?.trainable_params?.toLocaleString()}</div></div>
          </div>
        </div>
      )}

      {tab==='comparison' && comparison && (
        <div className="animate-fade-in">
          <div className="chart-card" style={{marginBottom:20}}>
            <div className="card-header"><h3 className="card-title">Actual vs Predicted Prices</h3></div>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={compData}><CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)"/>
                <XAxis dataKey="idx" tick={{fill:'var(--text-muted)',fontSize:11}} label={{value:'Test Samples',position:'bottom',fill:'var(--text-muted)',fontSize:11}}/>
                <YAxis tickFormatter={fmt} tick={{fill:'var(--text-muted)',fontSize:11}}/>
                <Tooltip contentStyle={ttStyle} formatter={(v) => fmt(v)}/><Legend wrapperStyle={{fontSize:12}}/>
                <Bar dataKey="actual" name="Actual" fill="#3b82f6" radius={[4,4,0,0]} opacity={0.8}/>
                <Bar dataKey="predicted" name="Predicted" fill="#10b981" radius={[4,4,0,0]} opacity={0.8}/>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="card-header"><h3 className="card-title">Test Set Results</h3></div>
            <div className="table-wrapper"><table className="data-table"><thead><tr><th>Sample</th><th>Actual</th><th>Predicted</th><th>Error</th><th>Accuracy</th></tr></thead><tbody>
              {compData.map(r => {const acc = r.actual>0 ? Math.max(0,100-Math.abs(r.actual-r.predicted)/r.actual*100) : 0; return (
                <tr key={r.idx}><td>{r.idx}</td><td>{fmt(r.actual)}</td><td>{fmt(r.predicted)}</td><td>{fmt(r.error)}</td>
                  <td><span className={`card-badge ${acc>80?'success':acc>50?'warning':'danger'}`}>{acc.toFixed(1)}%</span></td></tr>
              );})}
            </tbody></table></div>
          </div>
        </div>
      )}
    </div>
  );
}
