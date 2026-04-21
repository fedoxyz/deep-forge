import React, { useState, useEffect } from 'react';
import { Play, Square, RefreshCw } from 'lucide-react';
import { getConfigs, getConfig, startTraining, stopTraining, getLatestSamples} from '../utils/api';
import TrainingControls from '../components/Training/TrainingControls';
import SampleConfigPanel from '../components/Training/SampleConfigPanel';
import SampleViewer from '../components/Training/SampleViewer';
import ResumeButton from '../components/Training/ResumeButton';
import { useTrainingStatus } from '../hooks/useTraining';
import { useRuns } from '../hooks/useRuns';
import RunsPanel from '../components/Training/RunsPanel';
import TrainingTerminal from '../components/Training/TrainingTerminal';
import { tensorboardUrl } from '../utils/helpers';

export default function TrainPage({ onNavigate = null, isActive }) {
  const { status, refresh } = useTrainingStatus(1000);
  const [configs, setConfigs] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState('');
  const [configDetail, setConfigDetail] = useState(null);
  const [mode, setMode] = useState('lora');
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);

  const [latestSamples, setLatestSamples] = useState(null);
  const [showSamplePanel, setShowSamplePanel] = useState(false);
  const [samplePrompt, setSamplePrompt] = useState('');
  const [samplerType, setSamplerType] = useState('euler');
  const [sampleSteps, setSampleSteps] = useState(8);
  const [sampleSeed, setSampleSeed] = useState(42);
  const [sampleMsg, setSampleMsg] = useState('');

  const { runs, refresh: refreshRuns } = useRuns();
  const currentRunName = status?.run_name;

  useEffect(() => {
    if (!isActive) return;
  
    getConfigs()
      .then(({ configs: c }) => setConfigs(c))
      .catch(() => {});
  }, [isActive]);

  // When a config is selected, load its details to read the mode and model_spec
  useEffect(() => {
    if (!selectedConfig) { setConfigDetail(null); return; }
    getConfig(selectedConfig).then(({ config: c }) => {
      setConfigDetail(c);
      // Auto-set mode from config if it has a general.mode
      if (c?.general?.mode) setMode(c.general.mode);
    }).catch(() => setConfigDetail(null));
  }, [selectedConfig]);

  useEffect(() => {
    if (status?.status === 'completed' || status?.status === 'idle') {
      refreshRuns();
    }
  }, [status?.status]);

  const hasPipeline = !!configDetail?.pipeline?.name;
  const hasSamplingConfig = configDetail?.sampling?.enabled;
  const showSampling = hasPipeline || hasSamplingConfig;

  const handleStart = async () => {
    if (!selectedConfig) { setMsg('Select a config first'); return; }
    setLoading(true);
    try {
      await startTraining(selectedConfig, null, mode);
      setMsg(`${mode} training started!`);
      refresh();
    } catch (e) { setMsg(`Error: ${e.message}`); }
    setLoading(false);
  };

  const handleStop = async () => {
    setLoading(true);
    try { await stopTraining(); setMsg('Stopping...'); refresh(); }
    catch (e) { setMsg(`Error: ${e.message}`); }
    setLoading(false);
  };

  const isTraining = status?.status === 'training';
  const isStopping = status?.status === 'stopping';

  useEffect(() => {
    if (!isTraining || !showSampling) return;
    const interval = setInterval(async () => {
      try {
        const { sample } = await getLatestSamples();
        if (sample) setLatestSamples(sample);
      } catch {}
    }, 5000);
    return () => clearInterval(interval);
  }, [isTraining, showSampling]);

  // Derive info from loaded config
  const configModelSpec = configDetail?.model_spec;
  const configModelPath = configDetail?.model?.path;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Training</h1>

      <div className="bg-forge-surface border border-forge-border rounded-lg p-6 space-y-4">
        {/* Mode selector */}
        <div>
          <label className="text-xs text-forge-muted uppercase tracking-wide block mb-2">Training Mode</label>
          <div className="flex gap-2">
            {[
              { id: 'lora', label: 'LoRA', desc: 'Train adapters on pretrained model' },
              { id: 'full_finetune', label: 'Full Fine-tune', desc: 'Fine-tune model layers' },
              { id: 'train_custom', label: 'Custom Model', desc: 'Train model spec from scratch' },
            ].map(m => (
              <button key={m.id} onClick={() => setMode(m.id)} disabled={isTraining}
                className={`flex-1 p-3 rounded-lg border text-left transition-colors ${
                  mode === m.id
                    ? 'border-forge-accent bg-forge-accent/10 text-forge-accent'
                    : 'border-forge-border hover:border-forge-accent/50'
                }`}>
                <div className="text-sm font-medium">{m.label}</div>
                <div className="text-xs text-forge-muted mt-0.5">{m.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Config selector + start/stop */}
        <div className="flex items-end gap-4">
          <div className="flex-1">
            <label className="text-xs text-forge-muted uppercase tracking-wide block mb-1.5">Config</label>
            <select value={selectedConfig} onChange={e => setSelectedConfig(e.target.value)} disabled={isTraining}
              className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2.5 text-sm font-mono focus:border-forge-accent focus:outline-none disabled:opacity-50">
              <option value="">Select a config...</option>
              {configs.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </div>
          {!isTraining ? (
            <button onClick={handleStart} disabled={loading || !selectedConfig}
              className="flex items-center gap-2 px-6 py-2.5 bg-forge-success text-black rounded-lg text-sm font-semibold hover:bg-forge-success/90 disabled:opacity-50">
              <Play className="w-4 h-4" /> Start
            </button>
          ) : (
            <>
              <button onClick={handleStop} disabled={loading || isStopping}
                className="flex items-center gap-2 px-6 py-2.5 bg-forge-error text-white rounded-lg text-sm font-semibold hover:bg-forge-error/90 disabled:opacity-50">
                <Square className="w-4 h-4" /> Stop
              </button>
              <ResumeButton
                status={status} selectedConfig={selectedConfig} mode={mode}
                setMsg={setMsg} setLoading={setLoading} refresh={refresh}
              />
            </>
          )}
        </div>

        {isTraining && (
          <TrainingControls
            showSamplePanel={showSamplePanel} setShowSamplePanel={setShowSamplePanel}
            sampleMsg={sampleMsg} setSampleMsg={setSampleMsg}
            showSamplingControls={hasSamplingConfig}
          />
        )}
        
        {showSamplePanel && isTraining && showSampling && (
          <SampleConfigPanel
            samplePrompt={samplePrompt} setSamplePrompt={setSamplePrompt}
            samplerType={samplerType} setSamplerType={setSamplerType}
            sampleSteps={sampleSteps} setSampleSteps={setSampleSteps}
            sampleSeed={sampleSeed} setSampleSeed={setSampleSeed}
            setSampleMsg={setSampleMsg}
          />
        )}

        {/* Config summary when selected */}
        {configDetail && selectedConfig && (
          <div className="bg-forge-bg border border-forge-border rounded p-3 space-y-1">
            <div className="flex items-center gap-3 text-xs">
              <span className="text-forge-muted">Config mode:</span>
              <span className="font-mono text-forge-accent">{configDetail.general?.mode || 'lora'}</span>
              {configModelSpec && (
                <>
                  <span className="text-forge-border">|</span>
                  <span className="text-forge-muted">Model spec:</span>
                  <span className="font-mono text-green-400">{configModelSpec}</span>
                </>
              )}
              {configModelPath && (
                <>
                  <span className="text-forge-border">|</span>
                  <span className="text-forge-muted">Model:</span>
                  <span className="font-mono text-forge-text truncate max-w-xs">{configModelPath}</span>
                </>
              )}
            </div>
            {configDetail.general?.mode && configDetail.general.mode !== mode && (
              <p className="text-xs text-yellow-400">
                ⚠ Config specifies mode "{configDetail.general.mode}" but you selected "{mode}". The mode selector above will override.
              </p>
            )}
          </div>
        )}

        {mode === 'train_custom' && !configModelSpec && (
          <div className="text-xs text-forge-muted bg-forge-bg rounded p-3 space-y-2">
            <p>
              For custom models, your config needs a <code className="px-1 bg-forge-surface rounded">model_spec</code> reference.
            </p>
            <p>
              {onNavigate ? (
                <>
                  <button onClick={() => onNavigate('builder')} className="text-forge-accent hover:underline">Build a model →</button>
                  {' then '}
                  <button onClick={() => onNavigate('configs')} className="text-forge-accent hover:underline">create a config →</button>
                  {' that references it.'}
                </>
              ) : (
                'Use the Model Builder to design your model, save the spec, then reference it in a config.'
              )}
            </p>
          </div>
        )}

        {msg && <p className="text-sm text-forge-warning">{msg}</p>}
      </div>

      <TrainingTerminal isTraining={isTraining} />

      {/* Live status */}
      {status && status.status !== 'idle' && (
        <div className="bg-forge-surface border border-forge-border rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium">Live Status</h2>
            <div className="flex items-center gap-2">
              {status.mode && <span className="text-xs font-mono text-forge-accent px-2 py-0.5 bg-forge-accent/10 rounded">{status.mode}</span>}
              <button onClick={refresh} className="text-forge-muted hover:text-forge-accent"><RefreshCw className="w-4 h-4" /></button>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              ['Status', status.status, isTraining ? 'text-forge-success' : status.status==='error' ? 'text-forge-error' : 'text-forge-text'],
              ['Step', status.current_step],
              ['Epoch', `${(status.current_epoch||0)+1} / ${status.total_epochs}`],
              ['Loss', typeof status.loss==='number' ? status.loss.toFixed(6) : '—'],
              ['Accuracy', status.val_accuracy != null ? `${(status.val_accuracy*100).toFixed(1)}%` : status.accuracy != null ? `${(status.accuracy*100).toFixed(1)}%` : '—'],
            ].map(([label, val, cls]) => (
              <div key={label}>
                <div className="text-xs text-forge-muted mb-0.5">{label}</div>
                <div className={`font-mono text-sm ${cls || ''}`}>{val}</div>
              </div>
            ))}
          </div>

          {isTraining && status.total_epochs > 0 && (
            <div className="mt-4 w-full bg-forge-bg rounded-full h-2 overflow-hidden">
              <div className="bg-forge-accent h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100,((status.current_epoch||0)/status.total_epochs)*100)}%` }} />
            </div>
          )}

          {status.error && (
            <div className="mt-4 p-3 bg-forge-error/10 border border-forge-error/30 rounded text-sm text-forge-error font-mono break-all">
              {status.error}
            </div>
          )}
        </div>
      )}

      <div className="bg-forge-surface border border-forge-border rounded-lg p-6">
        <RunsPanel
          runs={runs}
          currentRunName={currentRunName}
          configName={selectedConfig}
          mode={mode}
          onResumeComplete={() => { refresh(); refreshRuns(); }}
        />
      </div>

      {showSampling && isTraining && <SampleViewer latestSamples={latestSamples} />}

      {/* TensorBoard links */}
      <div className="bg-forge-surface border border-forge-border rounded-lg p-6">
        <h2 className="text-sm font-medium mb-2">TensorBoard</h2>
        <p className="text-sm text-forge-muted mb-3">Detailed metrics, loss curves, accuracy, learning rate schedules.</p>
        <div className="flex gap-3">
          <a href={tensorboardUrl(6006)} target="_blank" rel="noreferrer"
            className="px-4 py-2 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-sm hover:bg-forge-accent/20">
            TensorBoard (6006)
          </a>
        </div>
      </div>
    </div>
  );
}
