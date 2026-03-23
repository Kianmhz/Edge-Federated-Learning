import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

const SERVER_URL = process.env.REACT_APP_SERVER_URL || 'http://127.0.0.1:9000';

function App() {
  const [status, setStatus] = useState({
    current_round: 0,
    active_clients: 0,
    pending_updates: 0,
    clients: {},
    history: [],
    connected_clients: [],
    aggregation_in_progress: false
  });
  
  const [metrics, setMetrics] = useState({
    rounds: [],
    accuracy: [],
    loss: [],
    last_updated: null
  });
  
  const [config, setConfig] = useState({
    num_clients: 0,
    participation_prob: 0.5,
    updates_per_round: 2,
    aggregation_timeout: 45,
    current_round: 0,
    connected_clients: 0
  });
  
  const [selectedClient, setSelectedClient] = useState(null);
  const [clientHistory, setClientHistory] = useState([]);
  const [roundDetails, setRoundDetails] = useState({});
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Track last metrics update to prevent unnecessary re-renders
  const lastMetricsUpdate = useRef(null);

  useEffect(() => {
    fetchStatus();
    fetchMetrics();
    fetchConfig();
    
    const statusInterval = setInterval(fetchStatus, 2000);
    const metricsInterval = setInterval(fetchMetricsIfNeeded, 3000);
    const configInterval = setInterval(fetchConfig, 5000);
    
    return () => {
      clearInterval(statusInterval);
      clearInterval(metricsInterval);
      clearInterval(configInterval);
    };
  }, []);
  
  const fetchStatus = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/status`);
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      
      const data = await response.json();
      setStatus(data);
      setIsLoading(false);
      setError(null);
      
    } catch (err) {
      console.error('Failed to fetch status:', err);
      setError(err.message);
      setIsLoading(false);
    }
  };

  const fetchMetricsIfNeeded = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/metrics`);
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      
      const data = await response.json();
      
      // Only update if last_updated timestamp changed (new round completed)
      if (data.last_updated !== lastMetricsUpdate.current) {
        setMetrics(data);
        lastMetricsUpdate.current = data.last_updated;
        console.log("📊 Metrics updated after round completion");
      }
      
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/metrics`);
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      
      const data = await response.json();
      setMetrics(data);
      lastMetricsUpdate.current = data.last_updated;
      
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };

  const fetchConfig = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/config`);
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      
      const data = await response.json();
      setConfig(data);
      
    } catch (err) {
      console.error('Failed to fetch config:', err);
    }
  };

  const fetchClientHistory = async (clientId) => {
    try {
      const response = await fetch(`${SERVER_URL}/client_history/${clientId}`);
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      
      const data = await response.json();
      setClientHistory(data.history);
      setSelectedClient(clientId);
      
    } catch (err) {
      console.error('Failed to fetch client history:', err);
    }
  };

  const fetchRoundDetails = async (roundNum) => {
    try {
      const response = await fetch(`${SERVER_URL}/round_details/${roundNum}`);
      if (!response.ok) return;
      
      const data = await response.json();
      setRoundDetails(prev => ({ ...prev, [roundNum]: data }));
      
    } catch (err) {
      console.error('Failed to fetch round details:', err);
    }
  };

  // Fetch round details for displayed rounds
  useEffect(() => {
    if (status.history && status.history.length > 0) {
      status.history.forEach(h => {
        if (!roundDetails[h.round]) {
          fetchRoundDetails(h.round);
        }
      });
    }
  }, [status.history]);

  // Data processing - stable, only updates when metrics change
  const accuracyData = (metrics.rounds && metrics.rounds.length > 0) 
    ? metrics.rounds.map((round, idx) => ({
        round,
        accuracy: parseFloat((metrics.accuracy[idx] * 100).toFixed(2))
      }))
    : [];

  const lossData = (metrics.rounds && metrics.rounds.length > 0)
    ? metrics.rounds.map((round, idx) => ({
        round,
        loss: parseFloat((metrics.loss[idx] || 0).toFixed(4))
      }))
    : [];

  const latestAccuracy = accuracyData.length > 0 
    ? accuracyData[accuracyData.length - 1].accuracy 
    : 0;

  const clientsByStatus = groupClientsByStatus(status.clients);
  const timedOutCount = clientsByStatus['timed_out']?.length || 0;

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner">⏳</div>
        <div className="loading-text">Connecting to server...</div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="error-container">
        <h2>❌ Connection Error</h2>
        <p><strong>Could not connect to server:</strong> {error}</p>
        <p>Make sure the server is running:</p>
        <code className="code-block">python server/main.py</code>
      </div>
    );
  }

  return (
    <div className="container">
      
      {/* Header */}
      <div className="header">
        <h1 className="title">
          <span className="emoji">🤖</span> Federated Learning Dashboard
        </h1>
        <div className="subtitle">
          Real-time Distributed ML Monitoring
          {status.aggregation_in_progress && <span className="aggregating"> • Aggregating...</span>}
        </div>
      </div>
      
      {/* Top Stats */}
      <div className="stats-grid">
        <StatCard 
          title="Round" 
          value={status.current_round} 
          color="#4CAF50"
          icon="🔄"
        />
        <StatCard 
          title="Connected" 
          value={config.connected_clients} 
          color="#2196F3"
          icon="📱"
        />
        <StatCard 
          title="Participation" 
          value={`${(config.participation_prob * 100).toFixed(0)}%`}
          color="#FF9800"
          icon="🎲"
        />
        <StatCard 
          title="Accuracy" 
          value={`${latestAccuracy.toFixed(1)}%`} 
          color="#9C27B0"
          icon="🎯"
        />
      </div>

      {/* Charts Row - Only updates after round completion */}
      <div className="charts-grid">
        
        {/* Accuracy Chart */}
        <div className="chart-card">
          <h2 className="card-title">📈 Model Accuracy</h2>
          {accuracyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[0, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => `${value}%`} />
                <Line type="monotone" dataKey="accuracy" stroke="#4CAF50" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState message="Waiting for training data..." />
          )}
        </div>
        
        {/* Loss Chart */}
        <div className="chart-card">
          <h2 className="card-title">📉 Training Loss</h2>
          {lossData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => value.toFixed(4)} />
                <Line type="monotone" dataKey="loss" stroke="#F44336" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState message="Waiting for training data..." />
          )}
        </div>
      </div>

      {/* Active Clients Table */}
      <div className="table-card">
        <h2 className="card-title">
          👥 Connected Clients
          <span className="badge">{Object.keys(status.clients).length}</span>
        </h2>
        
        {Object.keys(status.clients).length === 0 ? (
          <EmptyState message="No clients connected. Start clients to see them here." />
        ) : (
          <div className="table-wrapper">
            <table className="table">
              <thead>
                <tr>
                  <th>Client ID</th>
                  <th>Status</th>
                  <th>Data Size</th>
                  <th>Round</th>
                  <th>Last Activity</th>
                  <th>History</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(status.clients)
                  .sort((a, b) => {
                    const idA = parseInt(a[0]) || 0;
                    const idB = parseInt(b[0]) || 0;
                    return idA - idB;
                  })
                  .map(([clientId, clientInfo]) => (
                    <tr key={clientId}>
                      <td>
                        <span className="client-badge">Client {clientId}</span>
                      </td>
                      <td>
                        <StatusBadge status={clientInfo.status} />
                      </td>
                      <td>
                        {clientInfo.data_size 
                          ? `${clientInfo.data_size.toLocaleString()} samples`
                          : 'N/A'
                        }
                      </td>
                      <td>
                        <span className="round-badge">Round {clientInfo.round}</span>
                      </td>
                      <td className="timestamp">
                        {new Date(clientInfo.timestamp).toLocaleTimeString()}
                      </td>
                      <td>
                        <button 
                          className="history-button"
                          onClick={() => fetchClientHistory(clientId)}
                        >
                          📜 View
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Round Participation History */}
      <div className="table-card">
        <h2 className="card-title">📋 Round Participation History</h2>
        
        {status.history && status.history.length > 0 ? (
          <div className="table-wrapper">
            <table className="table">
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Accuracy</th>
                  <th>Selected Clients</th>
                  <th>Submitted</th>
                  <th>Timed Out</th>
                  <th>Rate</th>
                </tr>
              </thead>
              <tbody>
                {status.history.slice().reverse().map((h) => {
                  const details = roundDetails[h.round] || {};
                  const selected = details.selected_clients || h.selected || [];
                  const submitted = details.submitted_clients || h.submitted || [];
                  const timedOut = details.timed_out_clients || h.timed_out || [];
                  const rate = (submitted.length / Math.max(selected.length, 1)) * 100;
                  
                  return (
                    <tr key={h.round}>
                      <td><span className="round-badge">Round {h.round}</span></td>
                      <td><strong>{(h.accuracy * 100).toFixed(2)}%</strong></td>
                      <td>
                        <div className="client-list">
                          {selected.map(cid => (
                            <span key={cid} className="mini-badge">{cid}</span>
                          ))}
                        </div>
                      </td>
                      <td>
                        <div className="client-list">
                          {submitted.map(cid => (
                            <span key={cid} className="mini-badge success">{cid}</span>
                          ))}
                        </div>
                      </td>
                      <td>
                        <div className="client-list">
                          {timedOut.map(cid => (
                            <span key={cid} className="mini-badge danger">{cid}</span>
                          ))}
                        </div>
                      </td>
                      <td>
                        <span className={`rate-badge ${rate === 100 ? 'perfect' : rate > 50 ? 'good' : 'low'}`}>
                          {rate.toFixed(0)}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState message="No rounds completed yet" />
        )}
      </div>

      {/* Client History Modal */}
      {selectedClient && (
        <div className="modal-overlay" onClick={() => setSelectedClient(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Client {selectedClient} History</h3>
            <button className="close-button" onClick={() => setSelectedClient(null)}>✕</button>
            
            <div className="history-timeline">
              {clientHistory.length > 0 ? (
                clientHistory.slice().reverse().map((event, idx) => (
                  <div key={idx} className="timeline-event">
                    <div className="event-time">{new Date(event.timestamp).toLocaleString()}</div>
                    <div className="event-content">
                      <strong>{event.event}</strong> - Round {event.round}
                      {event.details && Object.keys(event.details).length > 0 && (
                        <div className="event-details">
                          {JSON.stringify(event.details)}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <p>No history available</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Fault Tolerance Alert */}
      {timedOutCount > 0 && (
        <div className="alert-card">
          <h3 className="alert-title">⚠️ Fault Tolerance Active</h3>
          <p>
            <strong>{timedOutCount} client(s)</strong> timed out. 
            System continued with available updates.
          </p>
        </div>
      )}

      {/* System Info */}
      <div className="footer">
        <div className="footer-item">
          <strong>Configuration:</strong> {config.connected_clients} clients connected, 
          {' '}{(config.participation_prob * 100).toFixed(0)}% participation rate, 
          {' '}{config.aggregation_timeout}s timeout
        </div>
      </div>
    </div>
  );
}

// Helper Functions
function groupClientsByStatus(clients) {
  const grouped = {};
  Object.entries(clients).forEach(([clientId, info]) => {
    const status = info.status || 'unknown';
    if (!grouped[status]) {
      grouped[status] = [];
    }
    grouped[status].push({ id: clientId, ...info });
  });
  return grouped;
}

// Components
function StatCard({ title, value, color, icon }) {
  return (
    <div className="stat-card" style={{ borderLeftColor: color }}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-content">
        <div className="stat-title">{title}</div>
        <div className="stat-value" style={{ color }}>{value}</div>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const config = {
    'submitted': { color: '#4CAF50', label: '✓ Submitted' },
    'selected': { color: '#FF9800', label: '🎯 Selected' },
    'waiting': { color: '#2196F3', label: '○ Waiting' },
    'timed_out': { color: '#F44336', label: '✗ Timed Out' },
    'training': { color: '#FF9800', label: '⚙ Training' }
  }[status] || { color: '#9E9E9E', label: status };
  
  return (
    <span className="status-badge" style={{ backgroundColor: config.color }}>
      {config.label}
    </span>
  );
}

function EmptyState({ message }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">📭</div>
      <p>{message}</p>
    </div>
  );
}

export default App;