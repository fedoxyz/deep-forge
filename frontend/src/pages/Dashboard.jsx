import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Activity, Cpu, Zap, TrendingDown, Folder } from 'lucide-react';
import { getHealth, getOptunaStatus } from '../utils/api';
import { useTrainingStatus } from '../hooks/useTraining';

// ─── LTTB downsampler ──────────────────────────────────────────────────────────
function lttb(data, threshold) {
  const len = data.length;
  if (len <= threshold) return data;

  const sampled = [data[0]];
  let a = 0;
  const bucketSize = (len - 2) / (threshold - 2);

  for (let i = 0; i < threshold - 2; i++) {
    const rangeStart = Math.floor((i + 1) * bucketSize) + 1;
    const rangeEnd   = Math.min(Math.floor((i + 2) * bucketSize) + 1, len);
    const nextRangeEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, len);

    // Average of next bucket for triangle apex
    let avgX = 0, avgY = 0, count = 0;
    for (let j = rangeEnd; j < nextRangeEnd; j++) {
      avgX += data[j].step; avgY += data[j].loss; count++;
    }
    if (count === 0) { avgX = data[rangeEnd - 1].step; avgY = data[rangeEnd - 1].loss; }
    else { avgX /= count; avgY /= count; }

    // Pick point with max triangle area in current bucket
    let maxArea = -1, maxIdx = rangeStart;
    const ax = data[a].step, ay = data[a].loss;
    for (let j = rangeStart; j < rangeEnd; j++) {
      const area = Math.abs(
        (ax - avgX) * (data[j].loss - ay) - (ax - data[j].step) * (avgY - ay),
      );
      if (area > maxArea) { maxArea = area; maxIdx = j; }
    }
    sampled.push(data[maxIdx]);
    a = maxIdx;
  }
  sampled.push(data[len - 1]);
  return sampled;
}

// ─── Stat card ─────────────────────────────────────────────────────────────────
function Stat({ icon: I, label, value, sub, color = 'text-forge-accent' }) {
  return (
    <div className="bg-forge-surface border border-forge-border rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <I className={`w-4 h-4 ${color}`} />
        <span className="text-xs text-forge-muted uppercase tracking-wide">{label}</span>
      </div>
      <div className="text-2xl font-bold font-mono">{value}</div>
      {sub && <div className="text-xs text-forge-muted mt-1">{sub}</div>}
    </div>
  );
}

// ─── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const { status } = useTrainingStatus(1000);
  const [health, setHealth]             = useState(null);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [optunaStatus, setOptunaStatus] = useState(null);
  const [isLoading, setIsLoading]       = useState(true);

  // windowSize: number of steps to show; null = "all"
  const [windowSize, setWindowSize] = useState(null);
  // scrollPos: 0–100 percentage; 100 = pinned to latest
  const [scrollPos, setScrollPos]   = useState(100);

  // Track whether the user has explicitly set windowSize
  const windowSizeSetByUser = useRef(false);

  useEffect(() => {
    Promise.all([
      getHealth().catch(() => null),
      getOptunaStatus().catch(() => null),
    ]).then(([h, o]) => { setHealth(h); setOptunaStatus(o); setIsLoading(false); });

    const iv = setInterval(
      () => getOptunaStatus().then(setOptunaStatus).catch(() => {}),
      1000,
    );
    return () => clearInterval(iv);
  }, []);

  // ── Normalise data source ──────────────────────────────────────────────────
  const isOptunaRunning = optunaStatus?.status === 'training';

  const displayStatus = useMemo(() => {
    if (!isOptunaRunning) return status;
    return {
      ...optunaStatus,
      status: 'training',
      current_step: optunaStatus.current_trial,
      total_steps: optunaStatus.n_trials,
      current_epoch: optunaStatus.current_trial,
      total_epochs: optunaStatus.n_trials,
      loss: optunaStatus.best_value,
      smoothed_loss: optunaStatus.best_value,
      loss_history: (optunaStatus.trials || []).map(t => ({
        step: t.number + 1,
        loss: t.value,
        smoothed: t.value,
      })),
    };
  }, [isOptunaRunning, optunaStatus, status]);

  const effectiveMode = isOptunaRunning ? 'optuna_search' : status?.mode;

  // ── Raw loss data ──────────────────────────────────────────────────────────
  const lossData = useMemo(
    () => (displayStatus?.loss_history || []).map(d => ({
      step: d.step,
      loss: d.loss,
      smoothed: d.smoothed ?? d.loss,
    })),
    [displayStatus],
  );

  const totalSteps = lossData.length;

  // ── Derived window parameters ──────────────────────────────────────────────
  // effectiveWindow: how many points are actually shown (null windowSize = all)
  const showAll        = windowSize === null || windowSize >= totalSteps;
  const effectiveWindow = showAll ? totalSteps : windowSize;

  // When pinned to latest, always show the tail regardless of new data
  const pinnedToLatest = scrollPos >= 100;

  // visibleStart: index of the first point shown
  const maxStart    = Math.max(0, totalSteps - effectiveWindow);
  const visibleStart = showAll
    ? 0
    : pinnedToLatest
      ? maxStart
      : Math.round((scrollPos / 100) * maxStart);

  // ── Slice + downsample ─────────────────────────────────────────────────────
  const windowSlice  = lossData.slice(visibleStart, visibleStart + effectiveWindow);
  const windowedData = useMemo(() => lttb(windowSlice, 600), [windowSlice]);

  // ── Scroll label ───────────────────────────────────────────────────────────
  const scrollLabel = useMemo(() => {
    if (totalSteps === 0) return '—';
    if (showAll)          return 'all';
    if (pinnedToLatest)   return 'latest';
    const first = windowedData[0]?.step;
    const last  = windowedData[windowedData.length - 1]?.step;
    return first != null ? `${first}–${last}` : 'latest';
  }, [totalSteps, showAll, pinnedToLatest, windowedData]);

  // ── Window size slider handler ─────────────────────────────────────────────
  const handleWindowSizeChange = useCallback(e => {
    const v = Number(e.target.value);
    // Max position = "all steps"
    if (v >= totalSteps) {
      setWindowSize(null);
    } else {
      setWindowSize(v);
      windowSizeSetByUser.current = true;
    }
  }, [totalSteps]);

  // Slider value: use totalSteps as sentinel for "all"
  const sliderValue = showAll ? totalSteps : windowSize;

  // ── Status colour ──────────────────────────────────────────────────────────
  const sc = {
    idle:      'text-forge-muted',
    training:  'text-forge-success',
    completed: 'text-blue-400',
    error:     'text-forge-error',
    stopping:  'text-forge-warning',
  }[displayStatus?.status] || 'text-forge-muted';

  // ── Loading state ──────────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-forge-muted">Loading…</div>
      </div>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Dashboard</h1>
        <div className="flex items-center gap-2">
          <span className={`inline-block w-2 h-2 rounded-full ${
            displayStatus?.status === 'training' ? 'bg-green-500 animate-pulse' : 'bg-forge-muted'
          }`} />
          <span className={`text-sm font-mono ${sc}`}>{displayStatus?.status || 'idle'}</span>
          {effectiveMode && (
            <span className="text-xs font-mono text-forge-accent px-2 py-0.5 bg-forge-accent/10 rounded">
              {effectiveMode}
            </span>
          )}
        </div>
      </div>

      {/* Stat grid */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <Stat icon={Activity} label="Status"
          value={displayStatus?.status || '—'}
          sub={isOptunaRunning
            ? `Trial ${optunaStatus.current_trial}/${optunaStatus.n_trials}`
            : status?.config_name ? `Config: ${status.config_name}` : null}
        />

        <Stat icon={TrendingDown} label={isOptunaRunning ? 'Best Loss' : 'Loss'} color="text-green-400"
          value={isOptunaRunning
            ? (optunaStatus?.best_value != null ? optunaStatus.best_value.toFixed(5) : '—')
            : (typeof displayStatus?.smoothed_loss === 'number'
                ? displayStatus.smoothed_loss.toFixed(5)
                : typeof displayStatus?.loss === 'number'
                  ? displayStatus.loss.toFixed(5)
                  : '—')}
          sub={displayStatus?.val_loss != null ? `Val: ${displayStatus.val_loss.toFixed(5)}` : null}
        />

        <Stat icon={Zap} label={isOptunaRunning ? 'Trial' : 'Step'}
          value={isOptunaRunning
            ? `${optunaStatus.current_trial}/${optunaStatus.n_trials}`
            : displayStatus?.current_step || 0}
          sub={isOptunaRunning
            ? `${Math.round((optunaStatus.current_trial / optunaStatus.n_trials) * 100)}% complete`
            : `Epoch ${(displayStatus?.current_epoch || 0) + 1} / ${displayStatus?.total_epochs || '?'}`}
        />

        <Stat icon={Cpu} label="Accuracy" color="text-purple-400"
          value={displayStatus?.val_accuracy != null
            ? `${(displayStatus.val_accuracy * 100).toFixed(1)}%`
            : displayStatus?.accuracy != null
              ? `${(displayStatus.accuracy * 100).toFixed(1)}%`
              : '—'}
          sub={displayStatus?.val_accuracy != null
            ? 'validation'
            : displayStatus?.accuracy != null ? 'training' : null}
        />

        <Stat icon={Cpu} label="GPU" color="text-purple-400"
          value={health?.gpu_name?.split(' ').slice(-2).join(' ') || 'CPU'}
          sub={health?.cuda_available ? `${health.gpu_count} device(s)` : 'CPU mode'}
        />
      </div>

      {/* Optuna best params */}
      {isOptunaRunning && optunaStatus?.best_params && (
        <div className="bg-forge-surface border border-forge-border rounded-lg p-4">
          <h3 className="text-sm font-medium mb-2">Best Parameters So Far</h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
            {Object.entries(optunaStatus.best_params).map(([k, v]) => (
              <div key={k} className="text-xs">
                <span className="text-forge-muted">{k}:</span>{' '}
                <span className="font-mono text-forge-text">
                  {typeof v === 'number' ? v.toFixed(4) : v}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Run dir */}
      {displayStatus?.run_dir && (
        <div className="flex items-center gap-2 text-xs text-forge-muted bg-forge-surface border border-forge-border rounded px-3 py-2">
          <Folder className="w-3.5 h-3.5" />
          Run directory: <span className="font-mono text-forge-text">{displayStatus.run_dir}</span>
        </div>
      )}

      {/* Loss chart */}
      <div className="bg-forge-surface border border-forge-border rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-forge-muted">
            {isOptunaRunning ? 'Optuna Trial Losses' : 'Training Loss'}
          </h2>
          <label className="flex items-center gap-2 text-xs text-forge-muted cursor-pointer">
            <input
              type="checkbox"
              checked={showSmoothed}
              onChange={e => setShowSmoothed(e.target.checked)}
              className="accent-forge-accent"
            />
            Show smoothed
          </label>
        </div>

        {/* Window controls */}
        {totalSteps > 0 && (
          <div className="space-y-2 mb-4">
            {/* Window size */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-forge-muted w-24 shrink-0">Window size</span>
              <input
                type="range"
                min={Math.min(10, totalSteps)}
                max={totalSteps}
                value={sliderValue}
                step={1}
                onChange={handleWindowSizeChange}
                className="flex-1 accent-forge-accent"
              />
              <span className="text-xs font-mono text-forge-text w-20 text-right">
                {showAll ? `all (${totalSteps})` : `${effectiveWindow} steps`}
              </span>
            </div>

            {/* Scroll — disabled when showing all steps */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-forge-muted w-24 shrink-0">Scroll</span>
              <input
                type="range"
                min={0}
                max={100}
                value={showAll ? 100 : scrollPos}
                step={1}
                onChange={e => setScrollPos(Number(e.target.value))}
                className="flex-1 accent-forge-accent"
                disabled={showAll || totalSteps <= effectiveWindow}
              />
              <span className="text-xs font-mono text-forge-text w-20 text-right">
                {scrollLabel}
              </span>
            </div>
          </div>
        )}

        {/* Chart */}
        {totalSteps >= 1 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={windowedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
              <XAxis
                dataKey="step"
                stroke="#71717a"
                tick={{ fontSize: 11 }}
                label={{
                  value: isOptunaRunning ? 'Trial' : 'Step',
                  position: 'insideBottom',
                  offset: -5,
                  fill: '#71717a',
                }}
              />
              <YAxis stroke="#71717a" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: '#12121a',
                  border: '1px solid #1e1e2e',
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              {/* Raw loss line (faint) — only when smoothed overlay is active */}
              {showSmoothed && (
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#f9731644"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                />
              )}
              {/* Primary line */}
              <Line
                type="monotone"
                dataKey={showSmoothed ? 'smoothed' : 'loss'}
                stroke="#f97316"
                strokeWidth={2}
                dot={windowedData.length < 10}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-forge-muted text-sm">
            {isOptunaRunning
              ? `Waiting for first trial… (Trial ${optunaStatus?.current_trial || 0}/${optunaStatus?.n_trials || 0})`
              : 'Start a run to see loss curves.'}
          </div>
        )}
      </div>
    </div>
  );
}
