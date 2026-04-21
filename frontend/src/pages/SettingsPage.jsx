import React, { useEffect, useState } from 'react';
import { RefreshCw, Server } from 'lucide-react';
import { getHealth } from '../utils/api';

export default function SettingsPage() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const [h] = await Promise.all([getHealth()]);
      setHealth(h);
    } catch {}
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Settings</h1>
        <button onClick={load} className="text-forge-muted hover:text-forge-accent">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* System info */}
      <div className="bg-forge-surface border border-forge-border rounded-lg p-6">
        <h2 className="text-sm font-medium mb-4 flex items-center gap-2">
          <Server className="w-4 h-4 text-forge-accent" /> System
        </h2>
        {health ? (
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-forge-muted">CUDA</span>
              <span className={`ml-2 font-mono ${health.cuda_available ? 'text-forge-success' : 'text-forge-error'}`}>
                {health.cuda_available ? 'Available' : 'Not Available'}
              </span>
            </div>
            <div>
              <span className="text-forge-muted">GPU</span>
              <span className="ml-2 font-mono">{health.gpu_name || 'N/A'}</span>
            </div>
            <div>
              <span className="text-forge-muted">GPU Count</span>
              <span className="ml-2 font-mono">{health.gpu_count}</span>
            </div>
          </div>
        ) : (
          <p className="text-sm text-forge-muted">Loading...</p>
        )}
      </div>
    </div>
  );
}
