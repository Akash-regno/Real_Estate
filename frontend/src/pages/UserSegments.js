import React, { useState, useEffect } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend, LineChart, Line
} from 'recharts';
import { Users, Layers, MapPin } from 'lucide-react';
import { getSegmentationResults } from '../api';

const COLORS = ['#3b82f6','#10b981','#f59e0b','#f43f5e','#8b5cf6','#06b6d4','#ec4899','#14b8a6'];
const fmt = v => v>=1e7?`₹${(v/1e7).toFixed(1)}Cr`:v>=1e5?`₹${(v/1e5).toFixed(1)}L`:`₹${v.toLocaleString()}`;

export default function UserSegments() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sel, setSel] = useState(null);

  useEffect(() => {
    getSegmentationResults().then(r => setData(r.data)).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-container"><div className="loading-container"><div className="spinner"></div><span className="loading-text">Loading segments...</span></div></div>;
  if (!data) return <div className="page-container"><div className="empty-state"><Users size={64} className="empty-state-icon"/><h3 className="empty-state-title">No Data</h3></div></div>;

  const profiles = data.cluster_profiles;
  const pca = data.pca_data;
  const scatter = pca.x.map((x,i) => ({ x:+x.toFixed(3), y:+pca.y[i].toFixed(3), cluster:pca.labels[i] }));
  const sizes = Object.entries(profiles).map(([id,p]) => ({ name:p.segment_label, value:p.size, id:+id }));
  const elbow = data.elbow_data.k_range.map((k,i) => ({ k, inertia:+data.elbow_data.inertias[i].toFixed(2), silhouette:+data.elbow_data.silhouette_scores[i].toFixed(4) }));
  const sp = sel!==null ? profiles[sel.toString()] : null;
  const ttStyle = { background:'var(--bg-card)', border:'1px solid var(--border-default)', borderRadius:'8px', color:'var(--text-primary)' };

  return (
    <div className="page-container">
      <div className="page-header animate-fade-in-up">
        <div style={{display:'flex',alignItems:'center',gap:'12px'}}>
          <div className="stat-icon green"><Users size={22}/></div>
          <div><h1 className="page-title">User Segmentation</h1><p className="page-subtitle">K-Means clustering with PCA visualization</p></div>
        </div>
      </div>

      <div className="stats-grid">
        {Object.entries(profiles).map(([id,p],i) => (
          <div key={id} className="stat-card animate-fade-in-up" style={{cursor:'pointer', borderColor:sel===+id?COLORS[i]:undefined, boxShadow:sel===+id?`0 0 20px ${COLORS[i]}30`:undefined}} onClick={() => setSel(sel===+id?null:+id)}>
            <div style={{width:42,height:42,borderRadius:10,background:`${COLORS[i]}20`,display:'flex',alignItems:'center',justifyContent:'center',color:COLORS[i],fontWeight:800,fontSize:16,fontFamily:'Space Grotesk',flexShrink:0}}>{+id+1}</div>
            <div className="stat-info"><div className="stat-label">{p.segment_label}</div><div className="stat-value" style={{fontSize:22}}>{p.size}</div><div className="stat-change" style={{color:COLORS[i]}}>{p.percentage.toFixed(1)}%</div></div>
          </div>
        ))}
      </div>

      <div className="charts-grid">
        <div className="chart-card animate-fade-in-up">
          <div className="card-header"><h3 className="card-title">PCA Clusters</h3><span className="card-badge info">2D</span></div>
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart><CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)"/>
              <XAxis dataKey="x" name="PC1" tick={{fill:'var(--text-muted)',fontSize:11}}/>
              <YAxis dataKey="y" name="PC2" tick={{fill:'var(--text-muted)',fontSize:11}}/>
              <Tooltip contentStyle={ttStyle}/>
              <Scatter data={scatter}>{scatter.map((p,i) => <Cell key={i} fill={COLORS[p.cluster]} opacity={0.75}/>)}</Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card animate-fade-in-up">
          <div className="card-header"><h3 className="card-title">Elbow & Silhouette</h3><span className="card-badge success">K={data.n_clusters}</span></div>
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={elbow}><CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)"/>
              <XAxis dataKey="k" tick={{fill:'var(--text-secondary)',fontSize:11}}/>
              <YAxis yAxisId="l" tick={{fill:'#3b82f6',fontSize:11}}/><YAxis yAxisId="r" orientation="right" tick={{fill:'#10b981',fontSize:11}}/>
              <Tooltip contentStyle={ttStyle}/><Legend wrapperStyle={{fontSize:12}}/>
              <Line yAxisId="l" type="monotone" dataKey="inertia" name="Inertia" stroke="#3b82f6" strokeWidth={2} dot={{r:4}}/>
              <Line yAxisId="r" type="monotone" dataKey="silhouette" name="Silhouette" stroke="#10b981" strokeWidth={2} dot={{r:4}}/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-card animate-fade-in-up">
          <div className="card-header"><h3 className="card-title">Segment Sizes</h3></div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart><Pie data={sizes} cx="50%" cy="50%" innerRadius={55} outerRadius={100} paddingAngle={4} dataKey="value">{sizes.map((_,i)=><Cell key={i} fill={COLORS[i]} stroke="none"/>)}</Pie><Tooltip contentStyle={ttStyle}/><Legend wrapperStyle={{fontSize:12}}/></PieChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card animate-fade-in-up">
          <div className="card-header"><h3 className="card-title">{sp?`${sp.segment_label} Profile`:'Select a Segment'}</h3></div>
          {sp ? (
            <div style={{display:'flex',flexDirection:'column',gap:14}}>
              {sp.price_stats && <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:12}}>
                {['mean','median','min','max'].map(k => <div key={k} className="result-detail-item"><div className="result-detail-label">{k} Price</div><div className="result-detail-value">{fmt(sp.price_stats[k])}</div></div>)}
              </div>}
              {sp.top_locations && <div><div className="form-label" style={{marginBottom:8}}>Top Locations</div><div style={{display:'flex',flexWrap:'wrap',gap:6}}>{Object.entries(sp.top_locations).map(([l,c])=><span key={l} className="toggle-chip active" style={{cursor:'default'}}><MapPin size={11}/> {l} ({c})</span>)}</div></div>}
            </div>
          ) : <div className="empty-state" style={{padding:'40px 20px'}}><Layers size={48} style={{color:'var(--text-muted)',opacity:0.4}}/><p style={{color:'var(--text-muted)',fontSize:14}}>Click a segment card above</p></div>}
        </div>
      </div>

      <div className="card animate-fade-in-up">
        <div className="card-header"><h3 className="card-title">Properties</h3></div>
        <div className="table-wrapper">
          <table className="data-table"><thead><tr><th>#</th><th>Location</th><th>Price</th><th>Area</th><th>BHK</th><th>Segment</th></tr></thead>
            <tbody>{data.properties.filter(p=>sel===null||p.cluster===sel).map(p=>(
              <tr key={p.index}><td>{p.index+1}</td><td style={{color:'var(--text-primary)',fontWeight:500}}>{p.location}</td><td>{fmt(p.price)}</td><td>{p.area} sqft</td><td>{p.bedrooms}</td>
                <td><span className="cluster-badge" style={{background:`${COLORS[p.cluster]}18`,color:COLORS[p.cluster]}}>{profiles[p.cluster.toString()]?.segment_label}</span></td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
