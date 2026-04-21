import React, { useState } from 'react';
import { RotateCcw, CheckCircle, XCircle, Clock, ChevronDown, ChevronRight } from 'lucide-react';

const STATUS_ICON = {
  completed: <CheckCircle className="w-3.5 h-3.5 text-forge-success" />,
  error:     <XCircle className="w-3.5 h-3.5 text-forge-error" />,
  training:  <Clock className="w-3.5 h-3.5 text-forge-accent animate-pulse" />,
  stopping:  <Clock className="w-3.5 h-3.5 text-forge-warning" />,
};

function CheckpointRow({ ckpt, onResume }) {
  return (
    <div className="flex items-center justify-between px-3 py-1.5 hover:bg-forge-bg/50 rounded text-xs">
      <div className="flex items-center gap-3">
        <span className="font-mono text-forge-muted w-16">step {ckpt.step}</span>
        <span className="text-forge-muted">epoch {ckpt.epoch}</span>
        <span className="text-forge-muted/60">{ckpt.tag}</span>
      </div>
      <button
        onClick={() => onResume(ckpt)}
        className="flex items-center gap-1 px-2 py-0.5 bg-forge-accent/10 text-forge-accent 
                   border border-forge-accent/30 rounded hover:bg-forge-accent/20 transition-colors"
      >
        <RotateCcw className="w-3 h-3" /> Resume
      </button>
    </div>
  );
}

function RunRow({ run, onResume, isCurrentRun }) {
  const [expanded, setExpanded] = useState(false);
  const [checkpoints, setCheckpoints] = useState(run.checkpoints || []);
  const [loadingCkpts, setLoadingCkpts] = useState(false);

  const handleExpand = async () => {
    if (!expanded && checkpoints.length === 0) {
      setLoadingCkpts(true);
      try {
        const res = await fetch(`/api/training/runs/${run.run_name}/checkpoints`);
        const data = await res.json();
        setCheckpoints(data.checkpoints || []);
      } catch {}
      setLoadingCkpts(false);
    }
    setExpanded(e => !e);
  };

  return (
    <div className={`border rounded-lg overflow-hidden transition-colors ${
      isCurrentRun ? 'border-forge-accent/40' : 'border-forge-border'
    }`}>
      {/* Run header */}
      <div
        className="flex items-center gap-3 px-3 py-2.5 bg-forge-surface cursor-pointer hover:bg-white/[0.02]"
        onClick={handleExpand}
      >
        <span className="shrink-0">{STATUS_ICON[run.status] || STATUS_ICON.error}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-mono truncate">{run.run_name}</span>
            {isCurrentRun && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-forge-accent/10 text-forge-accent shrink-0">
                active
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 mt-0.5 text-xs text-forge-muted">
            <span>{run.mode}</span>
            {run.config_name && <span>· {run.config_name}</span>}
            <span>· step {run.total_steps}</span>
            {run.checkpoints?.length > 0 && (
              <span>· {run.checkpoints.length} checkpoints</span>
            )}
          </div>
        </div>
        {/* Quick resume from last checkpoint */}
        {run.last_checkpoint && run.status !== 'training' && (
          <button
            onClick={e => { e.stopPropagation(); onResume(run.run_name, run.last_checkpoint); }}
            className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent text-black 
                       rounded text-xs font-medium hover:bg-forge-accent/90"
          >
            <RotateCcw className="w-3 h-3" /> Resume
          </button>
        )}
        <span className="text-forge-muted shrink-0">
          {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </span>
      </div>

      {/* Checkpoints list */}
      {expanded && (
        <div className="bg-forge-bg/30 border-t border-forge-border">
          {loadingCkpts && (
            <p className="text-xs text-forge-muted px-3 py-2">Loading checkpoints...</p>
          )}
          {!loadingCkpts && checkpoints.length === 0 && (
            <p className="text-xs text-forge-muted px-3 py-2">No checkpoints found.</p>
          )}
          {checkpoints.map((ckpt, i) => (
            <CheckpointRow
              key={i}
              ckpt={ckpt}
              onResume={ckpt => onResume(run.run_name, ckpt.path)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function RunsPanel({ runs, currentRunName, configName, mode, onResumeComplete }) {
  const [msg, setMsg] = useState('');
  const [resuming, setResuming] = useState(false);

  const handleResume = async (runName, checkpointPath) => {
    setResuming(true);
    try {
      const res = await fetch('/api/training/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config_name: configName,   // need the config to rebuild model
          mode,
          run_name: runName,
          checkpoint_path: checkpointPath,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Resume failed');
      setMsg(`Resuming from step ${checkpointPath}`);
      onResumeComplete?.();
    } catch (e) {
      setMsg(`Error: ${e.message}`);
    }
    setResuming(false);
    setTimeout(() => setMsg(''), 4000);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-forge-muted uppercase tracking-wide">Training Runs</span>
        {msg && <span className="text-xs text-forge-warning">{msg}</span>}
      </div>
      {runs.length === 0 && (
        <p className="text-sm text-forge-muted py-4 text-center">No runs yet.</p>
      )}
      {runs.map(run => (
        <RunRow
          key={run.run_name}
          run={run}
          isCurrentRun={run.run_name === currentRunName}
          onResume={handleResume}
        />
      ))}
    </div>
  );
}
