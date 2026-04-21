import React, { useState, useEffect } from 'react';
import { Save, Upload } from 'lucide-react';  // Add to imports
import { 
  getConfigs, getModelSpecs, startOptuna, 
  getDefaultSearchSpace, getBuiltinDatasets,
  getHparamConfigs, getHparamConfig, 
  getOptunaStatus, getOptunaResults,
  saveHparamConfig, deleteHparamConfig 
} from '../utils/api';
import { Search, Plus, Trash2, Play } from 'lucide-react';
import { useTrainingStatus } from '../hooks/useTraining';

function SpaceRow({ name, spec, onChange, onRemove }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <input value={name} readOnly className="w-32 bg-forge-bg border border-forge-border rounded px-2 py-1 font-mono" />
      <select value={spec.type} onChange={e => onChange({ ...spec, type: e.target.value })}
        className="bg-forge-bg border border-forge-border rounded px-2 py-1 font-mono">
        {['float', 'float_log', 'int', 'categorical'].map(t => <option key={t}>{t}</option>)}
      </select>
      {spec.type === 'categorical' ? (
        <input value={(spec.choices || []).join(',')} placeholder="16,32,64"
          onChange={e => onChange({ ...spec, choices: e.target.value.split(',').map(v => isNaN(Number(v)) ? v.trim() : Number(v)) })}
          className="flex-1 bg-forge-bg border border-forge-border rounded px-2 py-1 font-mono" />
      ) : (
        <>
          <input type="number" value={spec.low ?? 0} placeholder="low" onChange={e => onChange({ ...spec, low: parseFloat(e.target.value) })}
            className="w-20 bg-forge-bg border border-forge-border rounded px-2 py-1 font-mono" />
          <input type="number" value={spec.high ?? 1} placeholder="high" onChange={e => onChange({ ...spec, high: parseFloat(e.target.value) })}
            className="w-20 bg-forge-bg border border-forge-border rounded px-2 py-1 font-mono" />
          {(spec.type === 'float' || spec.type === 'int') && (
            <input type="number" value={spec.step ?? ''} placeholder="step" onChange={e => onChange({ ...spec, step: parseFloat(e.target.value) || undefined })}
              className="w-16 bg-forge-bg border border-forge-border rounded px-2 py-1 font-mono" />
          )}
        </>
      )}
      <button onClick={onRemove} className="text-forge-muted hover:text-red-400"><Trash2 className="w-3 h-3" /></button>
    </div>
  );
}

export default function OptunaPage() {
  const { status } = useTrainingStatus(2000);
  const [specs, setSpecs] = useState([]);
  const [datasets, setDatasets] = useState({});
  const [selSpec, setSelSpec] = useState('');
  const [selDataset, setSelDataset] = useState('mnist');
  const [nTrials, setNTrials] = useState(20);
  const [trialEpochs, setTrialEpochs] = useState(3);
  const [space, setSpace] = useState({});
  const [newParam, setNewParam] = useState('');
  const [msg, setMsg] = useState('');
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState('');
  const [configName, setConfigName] = useState('');

  const [optunaStatus, setOptunaStatus] = useState(null);
  const [optunaResults, setOptunaResults] = useState(null);

  useEffect(() => {
    getModelSpecs().then(d => setSpecs(d.specs || [])).catch(() => {});
    getBuiltinDatasets().then(d => setDatasets(d.datasets || {})).catch(() => {});
    getDefaultSearchSpace().then(d => setSpace(d.search_space || {})).catch(() => {});
    loadSavedConfigs();
    
    const interval = setInterval(() => {
      getOptunaStatus().then(setOptunaStatus).catch(() => {});
      getOptunaResults().then(setOptunaResults).catch(() => {});
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const loadSavedConfigs = async () => {
  try {
    const data = await getHparamConfigs();
    setSavedConfigs(data.configs || []);
  } catch (e) {
    console.error('Failed to load saved configs:', e);
  }
};

const handleLoadConfig = async (name) => {
  try {
    const data = await getHparamConfig(name);
    const cfg = data.config;
    setSpace(cfg.search_space || {});
    setSelSpec(cfg.model_spec_name || '');
    setSelDataset(cfg.dataset || 'mnist');
    setNTrials(cfg.n_trials || 20);
    setTrialEpochs(cfg.trial_epochs || 3);
    setMsg(`Loaded config: ${name}`);
  } catch (e) {
    setMsg(`Failed to load: ${e.message}`);
  }
};

const handleSaveConfig = async () => {
  if (!configName.trim()) {
    setMsg('Enter a config name to save');
    return;
  }
  try {
    await saveHparamConfig(configName, {
      search_space: space,
      model_spec_name: selSpec,
      dataset: selDataset,
      n_trials: nTrials,
      trial_epochs: trialEpochs,
    });
    setMsg(`Saved: ${configName}`);
    setConfigName('');
    loadSavedConfigs();
  } catch (e) {
    setMsg(`Save failed: ${e.message}`);
  }
};

  const handleStart = async () => {
    if (!selSpec) { setMsg('Select a saved model spec first'); return; }
    try {
      const { spec } = await (await fetch(`/api/model_specs/${selSpec}`)).json();
      const config = {
        model_spec: spec,
        dataset: { builtin: selDataset, batch_size: 64, validation_split: 0.1, num_workers: 2 },
        training: { epochs: trialEpochs, optuna_epochs: trialEpochs, seed: 42 },
        optimizer: { name: 'adamw', lr: 1e-3 },
        loss: { name: 'cross_entropy' },
        logging: { tensorboard: true },
      };
      await startOptuna(null, config, nTrials, 'train_custom', space);
      setMsg(`Optuna started — ${nTrials} trials, ${trialEpochs} epochs each`);
    } catch (e) { setMsg(e.message); }
  };

  const addParam = () => {
    if (!newParam.trim()) return;
    setSpace(p => ({ ...p, [newParam.trim()]: { type: 'float', low: 0, high: 1 } }));
    setNewParam('');
  };

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Hyperparameter Search</h1>
      <p className="text-sm text-forge-muted">Define your search space, pick a model architecture and dataset, then let Optuna find the best hyperparameters.</p>

      <div className="grid grid-cols-2 gap-6">
        {/* Left: search space editor */}
        <div className="bg-forge-surface border border-forge-border rounded-lg p-5 space-y-3">
          <h3 className="text-sm font-medium">Search Space</h3>
          <div className="space-y-2">
            {Object.entries(space).map(([k, v]) => (
              <SpaceRow key={k} name={k} spec={v}
                onChange={ns => setSpace(p => ({ ...p, [k]: ns }))}
                onRemove={() => setSpace(p => { const c = { ...p }; delete c[k]; return c; })} />
            ))}
          </div>
          <div className="flex gap-2">
            <input value={newParam} onChange={e => setNewParam(e.target.value)} placeholder="param name..."
              className="flex-1 bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono focus:border-forge-accent focus:outline-none" />
            <button onClick={addParam} className="flex items-center gap-1 px-3 py-1 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-xs hover:bg-forge-accent/20">
              <Plus className="w-3 h-3" /> Add
            </button>
          </div>
        </div>

        {/* Right: run config */}
        <div className="bg-forge-surface border border-forge-border rounded-lg p-5 space-y-3">
          <h3 className="text-sm font-medium">Run Config</h3>
          <div>
            <label className="text-xs text-forge-muted">Model Architecture (saved spec)</label>
            <select value={selSpec} onChange={e => setSelSpec(e.target.value)}
              className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-sm font-mono focus:border-forge-accent focus:outline-none">
              <option value="">Select a saved model spec...</option>
              {specs.map(s => <option key={s.name} value={s.name}>{s.name}</option>)}
            </select>
            <p className="text-xs text-forge-muted mt-1">Save a model in the Model Builder first, then select it here.</p>
          </div>
          <div>
            <label className="text-xs text-forge-muted">Dataset</label>
            <select value={selDataset} onChange={e => setSelDataset(e.target.value)}
              className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-sm font-mono focus:border-forge-accent focus:outline-none">
              {Object.entries(datasets).map(([k, v]) => <option key={k} value={k}>{k} — {v.description}</option>)}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-forge-muted">Trials</label>
              <input type="number" value={nTrials} onChange={e => setNTrials(parseInt(e.target.value) || 20)} min={1} max={200}
                className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-sm font-mono focus:border-forge-accent focus:outline-none" />
            </div>
            <div>
              <label className="text-xs text-forge-muted">Epochs per trial</label>
              <input type="number" value={trialEpochs} onChange={e => setTrialEpochs(parseInt(e.target.value) || 3)} min={1} max={50}
                className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-sm font-mono focus:border-forge-accent focus:outline-none" />
            </div>
          </div>

<div className="border-t border-forge-border pt-3 mt-3 space-y-2">
  <label className="text-xs text-forge-muted">Load/Save Config</label>
  <div className="flex gap-2">
    <select value={selectedConfig} onChange={e => {
      setSelectedConfig(e.target.value);
      if (e.target.value) handleLoadConfig(e.target.value);
    }}
      className="flex-1 bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-xs font-mono">
      <option value="">Load saved config...</option>
      {savedConfigs.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
    </select>
  </div>
  <div className="flex gap-2">
    <input value={configName} onChange={e => setConfigName(e.target.value)}
      placeholder="Config name..."
      className="flex-1 bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-xs font-mono" />
    <button onClick={handleSaveConfig}
      className="flex items-center gap-1 px-3 py-1.5 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-xs hover:bg-forge-accent/20">
      <Save className="w-3 h-3" /> Save
    </button>
  </div>
</div>

          <button onClick={handleStart} disabled={status?.status === 'training'}
            className="w-full flex items-center justify-center gap-2 px-6 py-2.5 bg-purple-600 text-white rounded-lg text-sm font-semibold hover:bg-purple-500 disabled:opacity-50">
            <Search className="w-4 h-4" /> Start Optuna Search
          </button>
          {msg && <p className="text-sm text-forge-warning">{msg}</p>}
        </div>
      </div>

      {/* Current search space as JSON */}
      <div className="bg-forge-surface border border-forge-border rounded-lg p-4">
        <h3 className="text-sm font-medium mb-2">Search Space JSON</h3>
        <pre className="text-xs font-mono text-forge-muted max-h-40 overflow-auto">{JSON.stringify(space, null, 2)}</pre>
      </div>
   {/* ADD THIS: Optuna Status & Results */}
      {optunaStatus && optunaStatus.status !== 'idle' && (
        <div className="bg-forge-surface border border-forge-border rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Search Progress</h3>
            <span className={`text-xs font-mono px-2 py-1 rounded ${
              optunaStatus.status === 'training' ? 'bg-green-500/10 text-green-400' :
              optunaStatus.status === 'completed' ? 'bg-blue-500/10 text-blue-400' :
              optunaStatus.status === 'error' ? 'bg-red-500/10 text-red-400' : 'bg-forge-accent/10 text-forge-accent'
            }`}>
              {optunaStatus.status}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-forge-muted mb-1">Progress</div>
              <div className="text-lg font-mono font-bold">{optunaStatus.current_trial} / {optunaStatus.n_trials}</div>
              <div className="w-full bg-forge-bg rounded-full h-2 mt-2">
                <div className="bg-purple-600 h-2 rounded-full transition-all" 
                  style={{ width: `${(optunaStatus.current_trial / optunaStatus.n_trials) * 100}%` }} />
              </div>
            </div>
            <div>
              <div className="text-xs text-forge-muted mb-1">Best Value</div>
              <div className="text-lg font-mono font-bold text-green-400">
                {optunaStatus.best_value != null ? optunaStatus.best_value.toFixed(6) : '—'}
              </div>
            </div>
            <div>
              <div className="text-xs text-forge-muted mb-1">Search Space</div>
              <div className="text-lg font-mono font-bold">{Object.keys(optunaStatus.search_space || {}).length} params</div>
            </div>
          </div>

          {optunaStatus.best_params && (
            <div>
              <div className="text-xs text-forge-muted mb-2">Best Parameters</div>
              <div className="bg-forge-bg border border-forge-border rounded p-3">
                <pre className="text-xs font-mono">{JSON.stringify(optunaStatus.best_params, null, 2)}</pre>
              </div>
            </div>
          )}

          {optunaStatus.error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded p-3">
              <div className="text-xs text-red-400">{optunaStatus.error}</div>
            </div>
          )}
        </div>
      )}

      {/* ADD THIS: Trial History */}
      {optunaStatus?.trials && optunaStatus.trials.length > 0 && (
        <div className="bg-forge-surface border border-forge-border rounded-lg p-5">
          <h3 className="text-sm font-medium mb-3">Trial History</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-forge-border">
                  <th className="text-left py-2 text-forge-muted">Trial</th>
                  <th className="text-left py-2 text-forge-muted">Value</th>
                  <th className="text-left py-2 text-forge-muted">State</th>
                  <th className="text-left py-2 text-forge-muted">Parameters</th>
                </tr>
              </thead>
              <tbody>
                {optunaStatus.trials.slice(-10).reverse().map((trial, idx) => (
                  <tr key={trial.number} className="border-b border-forge-border/50">
                    <td className="py-2">{trial.number}</td>
                    <td className="py-2 text-green-400">{trial.value?.toFixed(6) || '—'}</td>
                    <td className="py-2">
                      <span className={`px-2 py-0.5 rounded text-[10px] ${
                        trial.state === 'complete' ? 'bg-green-500/10 text-green-400' : 'bg-forge-muted/10 text-forge-muted'
                      }`}>
                        {trial.state}
                      </span>
                    </td>
                    <td className="py-2 text-forge-muted truncate max-w-md">{JSON.stringify(trial.params)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
