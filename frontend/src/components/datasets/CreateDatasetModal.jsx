import React, { useState } from 'react';
import { FolderOpen, X } from 'lucide-react';
import { createDataset } from '../../utils/api';

function Spinner() {
  return <div className="w-4 h-4 border-2 border-forge-accent border-t-transparent rounded-full animate-spin" />;
}

export default function CreateDatasetModal({ onCreated, onClose }) {
  const [name, setName] = useState('');
  const [baseDir, setBaseDir] = useState('/workspace/datasets');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [datasetType, setDatasetType] = useState('caption');

  const handleCreate = async () => {
    if (!name.trim()) return;
    setCreating(true); setError('');
    try {
      const result = await createDataset(name.trim(), baseDir, datasetType);
      onCreated(result);
    } catch (e) { setError(e.message); }
    setCreating(false);
  };

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="bg-forge-surface border border-forge-border rounded-xl p-6 w-96 space-y-4" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold">Create Dataset</h2>
          <button onClick={onClose} className="text-forge-muted hover:text-forge-text"><X className="w-4 h-4" /></button>
        </div>
        <div className="space-y-1">
          <label className="text-xs text-forge-muted uppercase tracking-wide">Dataset Name</label>
          <input value={name} onChange={e => setName(e.target.value)} placeholder="my_dataset"
            onKeyDown={e => e.key === 'Enter' && handleCreate()}
            className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none" />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-forge-muted uppercase tracking-wide">Base Directory</label>
          <input value={baseDir} onChange={e => setBaseDir(e.target.value)}
            className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none" />
          <p className="text-[10px] text-forge-muted/50">Will create: {baseDir}/{name || '…'}</p>
        </div>
        {/* Add after the baseDir input block */}
        <div className="space-y-1">
          <label className="text-xs text-forge-muted uppercase tracking-wide">Dataset Type</label>
          <div className="flex gap-2">
            {[
              { value: 'caption',      label: 'Caption' },
              { value: 'detection',    label: 'Detection' },
              { value: 'segmentation', label: 'Segmentation' },
              { value: 'classification', label: 'Classification' },
            ].map(({ value, label }) => (
              <button key={value} type="button"
                onClick={() => setDatasetType(value)}
                className={`flex-1 py-1.5 text-xs rounded border transition-colors ${
                  datasetType === value
                    ? 'border-forge-accent bg-forge-accent/10 text-forge-accent'
                    : 'border-forge-border text-forge-muted hover:text-forge-text'
                }`}>
                {label}
              </button>
            ))}
          </div>
        </div>
        {error && <p className="text-xs text-red-400">{error}</p>}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 text-sm text-forge-muted hover:text-forge-text">Cancel</button>
          <button onClick={handleCreate} disabled={creating || !name.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-forge-accent text-black rounded text-sm font-medium disabled:opacity-40">
            {creating ? <Spinner /> : <FolderOpen className="w-4 h-4" />} Create
          </button>
        </div>
      </div>
    </div>
  );
}
