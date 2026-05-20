import React, { useState, useRef } from 'react';
import { Flame, Eye, EyeOff, LogIn, AlertCircle } from 'lucide-react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import TrainPage from './pages/TrainPage';
import ConfigsPage from './pages/ConfigsPage';
import LayerInspector from './pages/LayerInspector';
import OptunaPage from './pages/OptunaPage';
import SettingsPage from './pages/SettingsPage';
import ModelBuilder from './pages/ModelBuilder';
import DatasetsPage from './pages/DatasetsPage';

const PAGES = {
  dashboard: Dashboard,
  builder: ModelBuilder,
  train: TrainPage,
  configs: ConfigsPage,
  datasets: DatasetsPage,
  layers: LayerInspector,
  optuna: OptunaPage,
  settings: SettingsPage,
};

export default function App() {
  const [page, setPage] = useState('dashboard');
  const mounted = useRef(new Set(['dashboard'])); // track what's been visited
  const [showPw, setShowPw] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [authed, setAuthed] = useState(() => !!sessionStorage.getItem('forge_pw'));
  const [pwInput, setPwInput] = useState('');
  const [pwError, setPwError] = useState('');
  
  const handleLogin = async () => {
    setIsChecking(true);
    const res = await fetch('/api/health', { headers: { 'X-Forge-Password': pwInput } });
    setIsChecking(false);
    if (res.ok) { sessionStorage.setItem('forge_pw', pwInput); setAuthed(true); }
    else setPwError('Incorrect password — try again');
  };
  
    // Gate the whole UI:
  if (!authed) return (
    <div className="flex min-h-screen items-center justify-center bg-forge-bg p-8">
      <div className="bg-forge-surface border border-forge-border rounded-2xl p-10 w-full max-w-sm flex flex-col items-center">
  
        {/* Icon */}
        <div className="w-14 h-14 rounded-xl bg-forge-bg border border-forge-border flex items-center justify-center mb-5">
          <Flame className="w-7 h-7 text-forge-accent" />
        </div>
  
        <h1 className="text-xl font-semibold tracking-tight mb-1">Deep Forge</h1>
        <p className="text-sm text-forge-muted mb-6">Enter your access password to continue</p>
  
        <div className="w-full">
          <label className="text-xs text-forge-muted mb-1.5 block">Password</label>
          <div className="relative">
            <input
              type={showPw ? 'text' : 'password'}
              className={`w-full h-10 bg-forge-bg border rounded-lg px-3 pr-10 text-sm font-mono tracking-widest
                focus:outline-none focus:ring-2 focus:ring-forge-accent/30
                ${pwError ? 'border-forge-error' : 'border-forge-border'}`}
              placeholder="••••••••••••"
              value={pwInput}
              onChange={e => { setPwInput(e.target.value); setPwError(''); }}
              onKeyDown={e => e.key === 'Enter' && handleLogin()}
            />
            <button
              type="button"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text"
              onClick={() => setShowPw(v => !v)}
            >
              {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
          {pwError && (
            <p className="text-forge-error text-xs mt-1.5 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" /> {pwError}
            </p>
          )}
        </div>
  
        <button
          className="mt-4 w-full h-10 bg-forge-text text-forge-bg rounded-lg text-sm font-medium
            flex items-center justify-center gap-2 hover:opacity-80 active:scale-[0.98] transition-all
            disabled:opacity-40 disabled:cursor-not-allowed"
          onClick={handleLogin}
          disabled={isChecking}
        >
          <LogIn className="w-4 h-4" />
          {isChecking ? 'Checking…' : 'Enter forge'}
        </button>
  
        <span className="mt-5 text-xs text-forge-muted border border-forge-border rounded-full px-3 py-0.5">
          Deep Forge v1
        </span>
  
        <div className="w-full h-px bg-forge-border mt-5 mb-4" />
        <p className="text-xs text-forge-muted text-center">Access is restricted to authorized users only</p>
      </div>
    </div>
  );

  const handleNavigate = (p) => {
    mounted.current.add(p); // mount it once, never unmount
    setPage(p);
  };

  return (
    <div className="flex min-h-screen bg-forge-bg">
      <Sidebar active={page} onNavigate={handleNavigate} />
      <main className="ml-56 flex-1 p-4 min-w-0 overflow-x-hidden">
        {Object.entries(PAGES).map(([key, PageComponent]) => (
          mounted.current.has(key) && (
            <div key={key} style={{ display: key === page ? 'block' : 'none' }}>
              <PageComponent onNavigate={handleNavigate} isActive={key === page} />
            </div>
          )
        ))}
      </main>
    </div>
  );
} 
