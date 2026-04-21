import React, { useState, useEffect, useRef } from 'react';
import PipelineSection from '../components/Configs/PipelineSection';
import SamplingSection from '../components/Configs/SamplingSection';
import PatternField from '../components/Configs/PatternField';
import { Save, Trash2, Plus, FileJson, Copy, Layers, ExternalLink, Download, Upload, ArrowUp, ArrowDown } from 'lucide-react';
import { getConfigs, getConfig, getDefaultConfig, saveConfig, deleteConfig, getModelSpecs } from '../utils/api';

function Field({ label, value, onChange, type = 'text', options, help }) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-forge-muted uppercase tracking-wide">{label}</label>
      {type === 'select' ? (
        <select value={value || ''} onChange={(e) => onChange(e.target.value)}
          className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none">
          {(options || []).map((o) => (
            <option key={typeof o === 'object' ? o.value : o} value={typeof o === 'object' ? o.value : o}>
              {typeof o === 'object' ? o.label : o}
            </option>
          ))}
        </select>
      ) : type === 'checkbox' ? (
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} className="accent-forge-accent" />
      ) : (
        <input type={type} value={value ?? ''} onChange={(e) => onChange(type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
          className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none" />
      )}
      {help && <p className="text-xs text-forge-muted/60">{help}</p>}
    </div>
  );
}

function Section({ title, children, defaultOpen = true, badge }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-forge-border rounded-lg overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-forge-surface hover:bg-white/[0.02] transition-colors">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{title}</span>
          {badge && <span className="text-xs px-1.5 py-0.5 rounded bg-forge-accent/10 text-forge-accent">{badge}</span>}
        </div>
        <span className="text-forge-muted text-xs">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div className="px-4 py-4 grid grid-cols-2 gap-4 bg-forge-bg/50">{children}</div>}
    </div>
  );
}

// ── Component update helpers ──
function updateComp(config, i, updater) {
  const comps = [...(config.model?.components || [])];
  comps[i] = typeof updater === 'function' ? updater(comps[i]) : { ...comps[i], ...updater };
  return comps;
}
function updateCompField(config, i, section, key, value) {
  return updateComp(config, i, (c) => ({
    ...c, [section]: { ...(c[section] || {}), [key]: value },
  }));
}

// ═══════════════════════════════════════════════════════════════
// Component Editor
// ═══════════════════════════════════════════════════════════════

function ComponentEditor({ comp, i, total, config, upd, modelSpecs, onNavigate }) {
  const strategy = comp.training?.strategy || '';
  const source = comp.source || 'file';

  // Strategy determines which training fields to show
  const showTrainingParams = strategy && strategy !== 'frozen' && comp.role !== 'training_adapter';
  const showLoraFields = strategy === 'lora' && comp.role !== 'training_adapter';
  const showUnfreezeFields = strategy === 'finetune';

  const strategyColors = {
    frozen: 'text-forge-muted',
    lora: 'text-blue-400',
    finetune: 'text-yellow-400',
    full: 'text-green-400',
    adapter: 'text-purple-400',
  };

  return (
    <div className="border border-forge-border rounded p-3 space-y-2 bg-forge-bg/30">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs text-forge-muted/50 font-mono w-5">{comp.execution_order ?? i}</span>
          <span className="text-sm font-medium text-forge-accent">{comp.name || `Component ${i}`}</span>
          {strategy && (
            <span className={`text-xs px-1.5 py-0.5 rounded bg-forge-surface ${strategyColors[strategy] || 'text-forge-muted'}`}>
              {strategy}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {/* Reorder buttons */}
          <button onClick={() => {
            if (i === 0) return;
            const comps = [...(config.model?.components || [])];
            // Swap execution_order
            const a = comps[i].execution_order ?? i;
            const b = comps[i-1].execution_order ?? (i-1);
            comps[i] = { ...comps[i], execution_order: b };
            comps[i-1] = { ...comps[i-1], execution_order: a };
            [comps[i], comps[i-1]] = [comps[i-1], comps[i]];
            upd('model', 'components', comps);
          }} className="text-forge-muted hover:text-forge-accent p-1" title="Move up">
            <ArrowUp className="w-3 h-3" />
          </button>
          <button onClick={() => {
            if (i >= total - 1) return;
            const comps = [...(config.model?.components || [])];
            const a = comps[i].execution_order ?? i;
            const b = comps[i+1].execution_order ?? (i+1);
            comps[i] = { ...comps[i], execution_order: b };
            comps[i+1] = { ...comps[i+1], execution_order: a };
            [comps[i], comps[i+1]] = [comps[i+1], comps[i]];
            upd('model', 'components', comps);
          }} className="text-forge-muted hover:text-forge-accent p-1" title="Move down">
            <ArrowDown className="w-3 h-3" />
          </button>
          <button onClick={() => {
            const comps = [...(config.model?.components || [])];
            comps.splice(i, 1);
            upd('model', 'components', comps);
          }} className="text-forge-error text-xs hover:underline ml-2">Remove</button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {/* Identity */}
        <Field label="Name" value={comp.name} onChange={(v) =>
          upd('model', 'components', updateComp(config, i, { name: v }))} />

        <Field label="Source" value={source} onChange={(v) =>
          upd('model', 'components', updateComp(config, i, { source: v }))}
          type="select" options={[
            { value: 'file', label: 'File (.safetensors)' },
            { value: 'spec', label: 'Model Spec (Model Builder)' },
            { value: 'empty', label: 'Empty (external)' },
          ]} />

        {/* Source-specific */}
        {source === 'file' && (
          <>
            <Field label="Path" value={comp.path} onChange={(v) =>
              upd('model', 'components', updateComp(config, i, { path: v }))} />
            <Field label="Key Filter (regex)" value={comp.key_filter || ''} onChange={(v) =>
              upd('model', 'components', updateComp(config, i, { key_filter: v || undefined }))}
              help="Extract keys from merged checkpoint" />
          </>
        )}
        {source === 'spec' && (
          <div className="col-span-2">
            <label className="text-xs text-forge-muted uppercase tracking-wide">Model Spec</label>
            <select value={comp.spec_name || ''} onChange={(e) =>
              upd('model', 'components', updateComp(config, i, { spec_name: e.target.value }))}
              className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none">
              <option value="">— Select —</option>
              {modelSpecs.map(s => <option key={s.name} value={s.name}>{s.name}</option>)}
            </select>
            {modelSpecs.length === 0 && (
              <p className="text-xs text-forge-muted mt-1">
                No specs found.{' '}
                {onNavigate && <button onClick={() => onNavigate('builder')} className="text-forge-accent hover:underline">Model Builder →</button>}
              </p>
            )}
          </div>
        )}

        <Field label="Role" value={comp.role || 'generic'} onChange={(v) =>
          upd('model', 'components', updateComp(config, i, { role: v }))}
          type="select" options={['denoiser', 'vae', 'text_encoder', 'adapter', 'training_adapter', 'generic']} />
        <Field label="Dtype" value={comp.dtype || 'float16'} onChange={(v) =>
          upd('model', 'components', updateComp(config, i, { dtype: v }))}
          type="select" options={['float16', 'bfloat16', 'float32']} />
          {comp.role === 'training_adapter' && (
            <p className="text-xs text-forge-muted col-span-2">
              Training adapters are applied during training but excluded from sampling/inference.
            </p>
          )}
      </div>

      {/* ── Strategy ── */}
      <div className="mt-2 border-t border-forge-border pt-2">
        <div className="grid grid-cols-2 gap-2">
          <Field label="Training Strategy" value={strategy} onChange={(v) =>
            upd('model', 'components', updateCompField(config, i, 'training', 'strategy', v || null))}
            type="select" options={[
              { value: '', label: '— Infer from mode —' },
              { value: 'frozen', label: 'Frozen (no training)' },
              { value: 'lora', label: 'LoRA adapters' },
              { value: 'finetune', label: 'Selective unfreeze' },
              { value: 'full', label: 'Full training' },
              { value: 'adapter', label: 'Adapter (custom module)' },
            ]} />

          {/* Only show training params when strategy is trainable */}
          {showTrainingParams && (
            <Field label="Learning Rate (blank = global)" value={comp.training?.lr ?? ''} onChange={(v) =>
              upd('model', 'components', updateCompField(config, i, 'training', 'lr', v === '' ? null : parseFloat(v)))}
              type="number" />
          )}
        </div>

        {showTrainingParams && (
          <div className="grid grid-cols-2 gap-2 mt-2">
            <Field label="Freeze for N epochs" value={comp.training?.freeze_epochs || 0} onChange={(v) =>
              upd('model', 'components', updateCompField(config, i, 'training', 'freeze_epochs', parseInt(v) || 0))}
              type="number" help="Delay training this component" />
            <Field label="Max Grad Norm (blank = global)" value={comp.training?.max_grad_norm ?? ''} onChange={(v) =>
              upd('model', 'components', updateCompField(config, i, 'training', 'max_grad_norm', v === '' ? null : parseFloat(v)))}
              type="number" />
          </div>
        )}

        {/* LoRA fields */}
{showLoraFields && (
  <div className="grid grid-cols-2 gap-2 mt-2 p-2 bg-blue-500/5 rounded border border-blue-500/10">
    <Field label="LoRA Rank" 
      value={comp.training?.lora?.rank ?? ''}
      onChange={(v) => {
        const existing = comp.training?.lora || {};
        upd('model', 'components', updateCompField(config, i, 'training', 'lora',
          v === '' ? null : { ...existing, rank: parseInt(v) }));
      }} 
      type="number"
      help={`Blank = global (${config.lora?.rank ?? 16})`}  // ← shows fallback value
    />
    <Field label="LoRA Alpha"
      value={comp.training?.lora?.alpha ?? ''}
      onChange={(v) => {
        const existing = comp.training?.lora || {};
        upd('model', 'components', updateCompField(config, i, 'training', 'lora',
          v === '' ? null : { ...existing, alpha: parseInt(v) }));
      }}
      type="number"
      help={`Blank = global (${config.lora?.alpha ?? 16})`}  // ← shows fallback value
    />
    <div className="col-span-2">
  <PatternField
    label="Target Patterns"
    value={comp.training?.lora?.target_patterns || []}
    onChange={(newPatterns) => {
      const existing = comp.training?.lora || { rank: config.lora?.rank || 16 };
      upd('model', 'components', updateCompField(config, i, 'training', 'lora', {
        ...existing,
        target_patterns: newPatterns,
      }));
    }}
    fallback={config.lora?.target_patterns || []}
    onOpenInspector={() => onNavigate('layers')}
  />
      <Field label="Conv LoRA Rank"
        value={comp.training?.lora?.conv_rank ?? ''}
        onChange={(v) => {
          const existing = comp.training?.lora || { rank: config.lora?.rank || 16 };
          upd('model', 'components', updateCompField(config, i, 'training', 'lora',
            v === '' ? { ...existing, conv_rank: undefined } : { ...existing, conv_rank: parseInt(v) }));
        }}
        type="number"
        help="Blank = no conv LoRA. Typically half of linear rank."
      />
      <Field label="Conv LoRA Alpha"
        value={comp.training?.lora?.conv_alpha ?? ''}
        onChange={(v) => {
          const existing = comp.training?.lora || {};
          upd('model', 'components', updateCompField(config, i, 'training', 'lora',
            v === '' ? { ...existing, conv_alpha: undefined } : { ...existing, conv_alpha: parseInt(v) }));
        }}
        type="number"
        help="Blank = same as conv_rank"
      />
    </div>
            {onNavigate && (
              <div className="col-span-2">
                <button onClick={() => onNavigate('layers')}
                  className="w-full flex items-center justify-center gap-2 px-3 py-1.5 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-xs hover:bg-forge-accent/20">
                  <Layers className="w-3 h-3" /> Layer Inspector
                </button>
              </div>
            )}
          </div>
        )}

        {/* Finetune fields */}
        {showUnfreezeFields && (
          <div className="grid grid-cols-2 gap-2 mt-2 p-2 bg-yellow-500/5 rounded border border-yellow-500/10">
            <div className="col-span-2">
  <PatternField
    label="Unfreeze Patterns"
    value={comp.training?.unfreeze_patterns || []}
    onChange={(newPatterns) =>
      upd('model', 'components', updateCompField(config, i, 'training', 'unfreeze_patterns', newPatterns))
    }
    help="Layer patterns to unfreeze. Empty = unfreeze all."
    onOpenInspector={() => onNavigate('layers')}
  />
            </div>
            {onNavigate && (
              <div className="col-span-2">
                <button onClick={() => onNavigate('layers')}
                  className="w-full flex items-center justify-center gap-2 px-3 py-1.5 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-xs hover:bg-forge-accent/20">
                  <Layers className="w-3 h-3" /> Layer Inspector
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Forward / Pipeline ── */}
      <div className="mt-2 border-t border-forge-border pt-2">
        <span className="text-xs text-forge-muted uppercase">Pipeline</span>
        <div className="grid grid-cols-2 gap-2 mt-1">
          <Field label="Input Key" value={comp.forward?.input_key || ''} onChange={(v) =>
            upd('model', 'components', updateCompField(config, i, 'forward', 'input_key', v || null))}
            help="Batch key to read (blank = prev output)" />
          <Field label="Output Key" value={comp.forward?.output_key || ''} onChange={(v) =>
            upd('model', 'components', updateCompField(config, i, 'forward', 'output_key', v || null))}
            help="Batch key to write (blank = component name)" />
          <Field label="No Grad" value={comp.forward?.no_grad || false} onChange={(v) =>
            upd('model', 'components', updateCompField(config, i, 'forward', 'no_grad', v))}
            type="checkbox" />
          <Field label="Cache Output" value={comp.forward?.cache_output || false} onChange={(v) =>
            upd('model', 'components', updateCompField(config, i, 'forward', 'cache_output', v))}
            type="checkbox" help="Cache across steps (frozen encoders)" />
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// ConfigsPage
// ═══════════════════════════════════════════════════════════════

export default function ConfigsPage({ onNavigate = null }) {
  const [configs, setConfigs] = useState([]);
  const [selected, setSelected] = useState(null);
  const [config, setConfig] = useState(null);
  const [name, setName] = useState('');
  const [msg, setMsg] = useState('');
  const [modelSpecs, setModelSpecs] = useState([]);
  const fileInputRef = useRef(null);

  const load = async () => {
    const { configs: list } = await getConfigs();
    setConfigs(list);
    try { const { specs } = await getModelSpecs(); setModelSpecs(specs || []); }
    catch { setModelSpecs([]); }
  };

  useEffect(() => { load(); }, []);

  const selectConfig = async (n) => {
    const { config: c } = await getConfig(n);
    setConfig(c); setSelected(n); setName(n);
  };

  const newConfig = async () => {
    const { config: c } = await getDefaultConfig();
    setConfig(c); setSelected(null); setName('new_config');
  };

  const handleSave = async () => {
    if (!name.trim()) return;
    try {
      await saveConfig(name.trim(), config);
      setMsg('Saved!'); setTimeout(() => setMsg(''), 2000);
      load(); setSelected(name.trim());
    } catch (e) { setMsg(`Error: ${e.message}`); }
  };

  const handleDelete = async () => {
    if (!selected || !confirm(`Delete "${selected}"?`)) return;
    await deleteConfig(selected); setConfig(null); setSelected(null); load();
  };

  const upd = (section, key, val) => {
    setConfig(prev => ({ ...prev, [section]: { ...prev[section], [key]: val } }));
  };

  const handleExport = () => {
    if (!config) return;
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `${name || 'config'}.json`;
    a.click(); URL.revokeObjectURL(url);
  };

  const handleImport = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        setConfig(JSON.parse(ev.target.result));
        setSelected(null); setName(file.name.replace(/\.json$/i, ''));
        setMsg(`Imported`); setTimeout(() => setMsg(''), 3000);
      } catch (err) { setMsg(`Error: ${err.message}`); }
    };
    reader.readAsText(file); e.target.value = '';
  };

  const mode = config?.general?.mode || 'lora';
  const components = config?.model?.components || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Configurations</h1>
        <div className="flex items-center gap-2">
          <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImport} />
          <button onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 px-3 py-2 bg-forge-surface border border-forge-border rounded-lg text-sm hover:border-forge-accent transition-colors">
            <Upload className="w-4 h-4" /> Import
          </button>
          <button onClick={handleExport} disabled={!config}
            className="flex items-center gap-1.5 px-3 py-2 bg-forge-surface border border-forge-border rounded-lg text-sm hover:border-forge-accent transition-colors disabled:opacity-40">
            <Download className="w-4 h-4" /> Export
          </button>
          <button onClick={newConfig}
            className="flex items-center gap-2 px-4 py-2 bg-forge-accent text-black rounded-lg text-sm font-medium hover:bg-forge-accent/90">
            <Plus className="w-4 h-4" /> New
          </button>
        </div>
      </div>

      <div className="flex gap-6">
        {/* Config list */}
        <div className="w-56 shrink-0 space-y-1">
          {configs.map((c) => (
            <button key={c.name} onClick={() => selectConfig(c.name)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded text-sm text-left transition-colors ${
                selected === c.name ? 'bg-forge-accent/10 text-forge-accent' : 'text-forge-muted hover:text-forge-text hover:bg-white/[0.03]'
              }`}>
              <FileJson className="w-3.5 h-3.5 shrink-0" /> {c.name}
            </button>
          ))}
          {configs.length === 0 && <p className="text-sm text-forge-muted px-3 py-2">No saved configs</p>}
        </div>

        {/* Editor */}
        {config ? (
          <div className="flex-1 space-y-4">
            {/* Name + actions */}
            <div className="flex items-center gap-3">
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Config name"
                className="bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono w-64 focus:border-forge-accent focus:outline-none" />
              <button onClick={handleSave} className="flex items-center gap-2 px-4 py-2 bg-forge-accent text-black rounded text-sm font-medium">
                <Save className="w-4 h-4" /> Save
              </button>
              {selected && (
                <button onClick={handleDelete} className="flex items-center gap-2 px-4 py-2 bg-forge-error/20 text-forge-error rounded text-sm">
                  <Trash2 className="w-4 h-4" /> Delete
                </button>
              )}
              {msg && <span className="text-sm text-forge-success">{msg}</span>}
            </div>

            {/* General */}
            <Section title="General">
              <Field label="Default Mode" value={mode} onChange={(v) => upd('general', 'mode', v)}
                type="select" options={['lora', 'full_finetune', 'train_custom']}
                help="Sets default strategy for components without one" />
              <Field label="Model Path (shortcut)" value={config.model?.path || ''} onChange={(v) => upd('model', 'path', v)}
                help="Single file shortcut. Leave empty if using components below." />
            </Section>

            <PipelineSection Section={Section} Field={Field} config={config} upd={upd} />
            {config?.pipeline?.name && (
              <SamplingSection Section={Section} Field={Field} config={config} upd={upd} />
            )}

            {/* ── COMPONENTS — the main thing ── */}
            <Section title="Model Components" defaultOpen={true} badge={components.length > 0 ? `${components.length}` : null}>
              <div className="col-span-2 space-y-3">
                {components.length === 0 && !config.model?.path && !config.model_spec && (
                  <div className="p-3 bg-forge-bg border border-forge-border rounded text-xs text-forge-muted text-center">
                    Add components to define your model pipeline. Each component is a model file, a Model Builder spec, or a custom module.
                  </div>
                )}
                {components.length === 0 && (config.model?.path || config.model_spec) && (
                  <div className="p-3 bg-forge-accent/5 border border-forge-accent/20 rounded text-xs text-forge-muted">
                    Using {config.model_spec ? `model spec "${config.model_spec}"` : `model path`} as a single component.
                    Add components here to override.
                  </div>
                )}

                {components.map((comp, i) => (
                  <ComponentEditor key={i} comp={comp} i={i} total={components.length}
                    config={config} upd={upd} modelSpecs={modelSpecs} onNavigate={onNavigate} />
                ))}

                <button onClick={() => {
                  const comps = [...components];
                  comps.push({
                    name: '', source: 'file', path: '', role: 'generic', dtype: 'float16',
                    execution_order: comps.length,
                    training: { strategy: null },
                    forward: {},
                  });
                  upd('model', 'components', comps);
                }} className="w-full py-2 border border-dashed border-forge-border rounded text-sm text-forge-muted hover:text-forge-accent hover:border-forge-accent transition-colors">
                  + Add Component
                </button>
              </div>
            </Section>

            {/* Global LoRA defaults */}
            <Section title="LoRA Defaults" defaultOpen={false}>
              <Field label="Default Rank" value={config.lora?.rank} onChange={(v) => upd('lora', 'rank', parseInt(v))} type="number" />
              <Field label="Default Alpha" value={config.lora?.alpha} onChange={(v) => upd('lora', 'alpha', parseInt(v))} type="number" />
              <Field label="Conv Rank (global)" value={config.lora?.conv_rank ?? ''} 
                onChange={(v) => upd('lora', 'conv_rank', v === '' ? null : parseInt(v))}
                type="number" help="Blank = disable conv LoRA globally" />
              <Field label="Conv Alpha (global)" value={config.lora?.conv_alpha ?? ''}
                onChange={(v) => upd('lora', 'conv_alpha', v === '' ? null : parseInt(v))}
                type="number" help="Blank = same as conv_rank" />
              <Field label="Dropout" value={config.lora?.dropout} onChange={(v) => upd('lora', 'dropout', v)} type="number" />
              <Field label="Init Reversed" value={config.lora?.init_reversed} onChange={(v) => upd('lora', 'init_reversed', v)} type="checkbox" />
              <div className="col-span-2">
  <PatternField
    label="Default Target Patterns"
    value={config.lora?.target_patterns || []}
    onChange={(newPatterns) => upd('lora', 'target_patterns', newPatterns)}
    help="Fallback for components without their own"
    onOpenInspector={() => onNavigate('layers')}
  />
              </div>
            </Section>

            <Section title="Dataset">
              <Field label="Dataset Path" value={config.dataset?.path} onChange={(v) => upd('dataset', 'path', v)} />
              <Field label="Builtin" value={config.dataset?.builtin || ''} onChange={(v) => upd('dataset', 'builtin', v || null)}
                help="Or use a built-in dataset: mnist, cifar10, etc." />
              <Field label="Batch Size" value={config.dataset?.batch_size} onChange={(v) => upd('dataset', 'batch_size', parseInt(v))} type="number" />
              <Field label="Num Workers" value={config.dataset?.num_workers} onChange={(v) => upd('dataset', 'num_workers', parseInt(v))} type="number" />
              <button onClick={() => onNavigate('datasets')}
                className="text-xs text-forge-accent hover:underline">
                Browse Dataset →
              </button>
            </Section>

            <Section title="Training">
              <Field label="Epochs" value={config.training?.epochs} onChange={(v) => upd('training', 'epochs', parseInt(v))} type="number" />
              <Field label="Grad Accumulation" value={config.training?.gradient_accumulation_steps} onChange={(v) => upd('training', 'gradient_accumulation_steps', parseInt(v))} type="number" />
              <Field label="Max Grad Norm" value={config.training?.max_grad_norm} onChange={(v) => upd('training', 'max_grad_norm', v)} type="number" />
              <Field label="Mixed Precision" value={config.training?.mixed_precision} onChange={(v) => upd('training', 'mixed_precision', v)} type="select" options={['fp16', 'bf16', 'none']} />
              <Field label="Gradient Checkpointing" value={config.training?.gradient_checkpointing} onChange={(v) => upd('training', 'gradient_checkpointing', v)} type="checkbox" />
              <Field label="Save Every N Steps" value={config.training?.save_every_n_steps} onChange={(v) => upd('training', 'save_every_n_steps', parseInt(v))} type="number" />
              <Field label="Eval Every N Steps" value={config.training?.eval_every_n_steps} onChange={(v) => upd('training', 'eval_every_n_steps', parseInt(v))} type="number" />
              <Field label="Seed" value={config.training?.seed} onChange={(v) => upd('training', 'seed', parseInt(v))} type="number" />
              <Field label="Low VRAM (CPU Offload)"
                value={config.training?.low_vram || false}
                onChange={(v) => upd('training', 'low_vram', v)}
                type="checkbox"
                help="Offload inactive components to CPU between forward passes." />
            </Section>

            <Section title="Optimizer">
              <Field label="Optimizer" value={config.optimizer?.name} onChange={(v) => upd('optimizer', 'name', v)} type="select" options={['adamw', 'adam8bit', 'sgd', 'prodigy']} />
              <Field label="Learning Rate" value={config.optimizer?.lr} onChange={(v) => upd('optimizer', 'lr', v)} type="number" />
              <Field label="Weight Decay" value={config.optimizer?.weight_decay} onChange={(v) => upd('optimizer', 'weight_decay', v)} type="number" />
            </Section>

            <Section title="Scheduler">
              <Field label="Scheduler" value={config.scheduler?.name} onChange={(v) => upd('scheduler', 'name', v)} type="select" options={['cosine_warmup', 'constant_warmup', 'cosine', 'one_cycle']} />
              <Field label="Warmup Steps" value={config.scheduler?.warmup_steps} onChange={(v) => upd('scheduler', 'warmup_steps', parseInt(v))} type="number" />
            </Section>

            {!config?.pipeline?.name && (
              <Section title="Loss">
                <Field label="Loss Function" value={config.loss?.name} onChange={(v) => upd('loss', 'name', v)}
                  type="select" options={['mse', 'huber', 'snr_weighted', 'v_pred', 'cross_entropy']} />
              </Section>
            )}

            {/* Raw JSON */}
            <div className="border border-forge-border rounded-lg overflow-hidden">
              <div className="px-4 py-3 bg-forge-surface flex items-center justify-between">
                <span className="text-sm font-medium">Raw JSON</span>
                <button onClick={() => navigator.clipboard.writeText(JSON.stringify(config, null, 2))}
                  className="text-xs text-forge-muted hover:text-forge-accent flex items-center gap-1">
                  <Copy className="w-3 h-3" /> Copy
                </button>
              </div>
              <pre className="px-4 py-3 text-xs font-mono text-forge-muted bg-forge-bg/50 max-h-72 overflow-auto">
                {JSON.stringify(config, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-forge-muted text-sm py-20">
            Select a config or create a new one
          </div>
        )}
      </div>
    </div>
  );
}
