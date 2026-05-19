import React, { useState, useRef, useEffect } from 'react';
import { RotateCcw, CheckCircle, XCircle, Clock, ChevronDown, ChevronRight, Download, Loader } from 'lucide-react';

const STATUS_ICON = {
  completed: <CheckCircle className="w-3.5 h-3.5 text-forge-success" />,
  error:     <XCircle    className="w-3.5 h-3.5 text-forge-error" />,
  training:  <Clock      className="w-3.5 h-3.5 text-forge-accent animate-pulse" />,
  stopping:  <Clock      className="w-3.5 h-3.5 text-forge-warning" />,
};

// ── Download popover ──────────────────────────────────────────────────────────

function DownloadPopover({ checkpointPath, runName, onClose }) {
  const [formats, setFormats]   = useState(null);   // null = loading
  const [error, setError]       = useState('');
  const [downloading, setDl]    = useState('');     // format id being downloaded
  const ref = useRef(null);

  // Fetch available formats for this checkpoint on mount
  useEffect(() => {
    const params = new URLSearchParams({ checkpoint_path: checkpointPath });
    fetch(`/api/export/formats?${params}`)
      .then(r => r.json())
      .then(d => {
        if (d.formats) setFormats(d.formats);
        else setError(d.detail || 'Failed to load formats');
      })
      .catch(() => setError('Network error'));
  }, [checkpointPath]);

  // Close on outside click
  useEffect(() => {
    const handler = e => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const handleDownload = async (fmt) => {
    setDl(fmt.id);
    setError('');
    try {
      const params = new URLSearchParams({
        checkpoint_path: checkpointPath,
        format: fmt.id,
        ...(runName ? { run_name: runName } : {}),
      });
      const res = await fetch(`/api/export/download?${params}`);
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `HTTP ${res.status}`);
      }
      // Derive filename from Content-Disposition or fall back
      const cd = res.headers.get('Content-Disposition') || '';
      const match = cd.match(/filename="?([^"]+)"?/);
      const filename = match?.[1] || `${runName || 'model'}_download${fmt.ext}`;

      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      onClose();
    } catch (e) {
      setError(e.message);
    }
    setDl('');
  };

  return (
    <div
      ref={ref}
      className="absolute right-0 top-full mt-1 z-50 w-64
                 bg-forge-surface border border-forge-border rounded-lg shadow-xl
                 overflow-hidden"
      style={{ minWidth: '220px' }}
    >
      {/* Header */}
      <div className="px-3 py-2 border-b border-forge-border">
        <p className="text-xs font-medium text-forge-text">Download as</p>
        <p className="text-xs text-forge-muted font-mono truncate mt-0.5">
          {checkpointPath.split('/').pop()}
        </p>
      </div>

      {/* Body */}
      <div className="py-1">
        {!formats && !error && (
          <div className="flex items-center gap-2 px-3 py-3 text-xs text-forge-muted">
            <Loader className="w-3 h-3 animate-spin" /> Loading formats…
          </div>
        )}

        {error && (
          <p className="px-3 py-2 text-xs text-forge-error">{error}</p>
        )}

        {formats && formats.map(fmt => (
          <button
            key={fmt.id}
            onClick={() => handleDownload(fmt)}
            disabled={!!downloading}
            className="w-full flex items-start gap-2.5 px-3 py-2 hover:bg-forge-bg/60
                       text-left transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="mt-0.5 flex-shrink-0">
              {downloading === fmt.id
                ? <Loader className="w-3.5 h-3.5 animate-spin text-forge-accent" />
                : <Download className="w-3.5 h-3.5 text-forge-muted" />
              }
            </span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium truncate">{fmt.label}</span>
                {fmt.badge && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded
                                   bg-forge-accent/10 text-forge-accent flex-shrink-0">
                    {fmt.badge}
                  </span>
                )}
              </div>
              <p className="text-[11px] text-forge-muted mt-0.5 leading-snug">{fmt.desc}</p>
            </div>
          </button>
        ))}

        {formats && formats.length === 0 && (
          <p className="px-3 py-2 text-xs text-forge-muted">No formats available.</p>
        )}
      </div>
    </div>
  );
}

// ── Checkpoint row ────────────────────────────────────────────────────────────

function CheckpointRow({ ckpt, runName, onResume }) {
  const [popoverOpen, setPopoverOpen] = useState(false);

  return (
    <div className="flex items-center justify-between px-3 py-1.5 hover:bg-forge-bg/50 rounded text-xs group">
      {/* Info */}
      <div className="flex items-center gap-3 min-w-0">
        <span className="font-mono text-forge-muted w-16 flex-shrink-0">
          step {ckpt.step ?? ckpt.global_step}
        </span>
        <span className="text-forge-muted flex-shrink-0">epoch {ckpt.epoch}</span>
        <span className="text-forge-muted/60 truncate">{ckpt.tag}</span>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1.5 flex-shrink-0 ml-2">
        {/* Resume */}
        <button
          onClick={() => onResume(ckpt)}
          className="flex items-center gap-1 px-2 py-0.5
                     bg-forge-accent/10 text-forge-accent border border-forge-accent/30
                     rounded hover:bg-forge-accent/20 transition-colors"
        >
          <RotateCcw className="w-3 h-3" /> Resume
        </button>

        {/* Download — with popover */}
        <div className="relative">
          <button
            onClick={() => setPopoverOpen(o => !o)}
            className={`flex items-center gap-1 px-2 py-0.5 rounded border transition-colors
                        ${popoverOpen
                          ? 'bg-forge-text/10 border-forge-border text-forge-text'
                          : 'border-forge-border/50 text-forge-muted hover:text-forge-text hover:border-forge-border'
                        }`}
            title="Download checkpoint"
          >
            <Download className="w-3 h-3" />
            <ChevronDown className="w-2.5 h-2.5" />
          </button>

          {popoverOpen && (
            <DownloadPopover
              checkpointPath={ckpt.path}
              runName={runName}
              onClose={() => setPopoverOpen(false)}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Run row ───────────────────────────────────────────────────────────────────

function RunRow({ run, onResume, isCurrentRun }) {
  const [expanded, setExpanded]       = useState(false);
  const [checkpoints, setCheckpoints] = useState(run.checkpoints || []);
  const [loadingCkpts, setLoading]    = useState(false);

  const handleExpand = async () => {
    if (!expanded && checkpoints.length === 0) {
      setLoading(true);
      try {
        const res  = await fetch(`/api/training/runs/${run.run_name}/checkpoints`);
        const data = await res.json();
        setCheckpoints(data.checkpoints || []);
      } catch {}
      setLoading(false);
    }
    setExpanded(e => !e);
  };

  return (
    <div className={`border rounded-lg overflow-visible transition-colors ${
      isCurrentRun ? 'border-forge-accent/40' : 'border-forge-border'
    }`}>
      {/* Run header */}
      <div
        className="flex items-center gap-3 px-3 py-2.5 bg-forge-surface cursor-pointer
                   hover:bg-white/[0.02] rounded-lg"
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
            {checkpoints.length > 0 && (
              <span>· {checkpoints.length} checkpoint{checkpoints.length !== 1 ? 's' : ''}</span>
            )}
          </div>
        </div>

        {/* Quick resume from last checkpoint */}
        {run.last_checkpoint && run.status !== 'training' && (
          <button
            onClick={e => { e.stopPropagation(); onResume(run.run_name, run.last_checkpoint); }}
            className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent
                       text-black rounded text-xs font-medium hover:bg-forge-accent/90"
          >
            <RotateCcw className="w-3 h-3" /> Resume
          </button>
        )}

        <span className="text-forge-muted shrink-0">
          {expanded
            ? <ChevronDown  className="w-4 h-4" />
            : <ChevronRight className="w-4 h-4" />
          }
        </span>
      </div>

      {/* Checkpoints list */}
      {expanded && (
        <div className="bg-forge-bg/30 border-t border-forge-border">
          {loadingCkpts && (
            <p className="text-xs text-forge-muted px-3 py-2">Loading checkpoints…</p>
          )}
          {!loadingCkpts && checkpoints.length === 0 && (
            <p className="text-xs text-forge-muted px-3 py-2">No checkpoints found.</p>
          )}
          {checkpoints.map((ckpt, i) => (
            <CheckpointRow
              key={i}
              ckpt={ckpt}
              runName={run.run_name}
              onResume={ckpt => onResume(run.run_name, ckpt.path)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Panel ─────────────────────────────────────────────────────────────────────

export default function RunsPanel({ runs, currentRunName, configName, mode, onResumeComplete }) {
  const [msg, setMsg]         = useState('');
  const [resuming, setResuming] = useState(false);

  const handleResume = async (runName, checkpointPath) => {
    setResuming(true);
    try {
      const res = await fetch('/api/training/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          config_name: configName,
          mode,
          run_name: runName,
          checkpoint_path: checkpointPath,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Resume failed');
      setMsg(`Resuming from ${checkpointPath}`);
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
