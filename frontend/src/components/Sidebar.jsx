import React from 'react';
import { Flame, Play, Settings, FileJson, Layers, Activity, Search, Box, Database } from 'lucide-react';
const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: Activity },
  { id: 'builder', label: 'Model Builder', icon: Box },
  { id: 'train', label: 'Train', icon: Play },
  { id: 'configs', label: 'Configs', icon: FileJson },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'layers', label: 'Layer Inspector', icon: Layers },
  { id: 'optuna', label: 'Hyperparam Search', icon: Search },
  { id: 'settings', label: 'Settings', icon: Settings },
];
import { tensorboardUrl } from '../utils/helpers';

export default function Sidebar({ active, onNavigate }) {
  return (
    <aside className="w-56 h-screen bg-forge-surface border-r border-forge-border flex flex-col fixed left-0 top-0 z-10">
      <div className="flex items-center gap-2 px-5 py-5 border-b border-forge-border">
        <Flame className="w-6 h-6 text-forge-accent" />
        <span className="font-bold text-2xl tracking-tight">Deep Forge</span>
      </div>
      <nav className="flex-1 py-3">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => onNavigate(id)}
            className={`w-full flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
              active === id
                ? 'text-forge-accent bg-forge-accent/10 border-r-2 border-forge-accent'
                : 'text-forge-muted hover:text-forge-text hover:bg-white/[0.03]'
            }`}>
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </nav>
      <div className="px-5 py-3 border-t border-forge-border">
        <a href="https://ko-fi.com/xposed" target="_blank" rel="noreferrer"
          className="flex items-center justify-center gap-2 w-full px-3 py-2 rounded-lg bg-forge-accent/10 hover:bg-forge-accent/20 border border-forge-accent/20 hover:border-forge-accent/50 text-forge-accent hover:text-forge-accent transition-all text-xs font-medium">
          🧡<span>Support the project</span>
        </a>
      </div>
      <div className="px-5 py-4 border-t border-forge-border space-y-2">
        <a href={tensorboardUrl(6006)} target="_blank" rel="noreferrer"
          className="flex items-center gap-2 text-xs text-forge-muted hover:text-forge-accent transition-colors">
          <Activity className="w-3.5 h-3.5" /> TensorBoard (6006)
        </a>
      </div>
    </aside>
  );
}
