import React, { useState } from 'react';
import { Grid, Check } from 'lucide-react';

function Spinner() {
  return <div className="w-4 h-4 border-2 border-forge-accent border-t-transparent rounded-full animate-spin" />;
}

const PRESETS = {
  sd15:    'SD 1.5 (512–768px)',
  sdxl:    'SDXL (768–1344px)',
  flux:    'Flux (1024–2048px)',
  ziturbo: 'Z-Image-Turbo (1024–2048px)',
  xl_fine: 'SDXL fine-tune (32-step)',
  custom:  'Custom…',
};

export default function CropBar({ totalCount, selectedCount, cropping, cropResult, onCropAll, onCropSelected }) {
  const [preset, setPreset] = useState('sdxl');
  const [showCustom, setShowCustom] = useState(false);
  const [minSize, setMinSize] = useState(768);
  const [maxSize, setMaxSize] = useState(1344);
  const [step, setStep] = useState(64);

  const cropParams = preset === 'custom'
    ? { preset: 'custom', minSize, maxSize, step }
    : { preset };

  return (
    <div className="border border-forge-border rounded-lg bg-forge-surface/50 overflow-hidden">
      <div className="flex items-center gap-3 px-3 py-2 flex-wrap">
        <span className="text-forge-muted flex items-center gap-1 text-xs shrink-0">
          <Grid className="w-3 h-3" /> Bucket Crop
        </span>
        <select value={preset}
          onChange={e => { setPreset(e.target.value); setShowCustom(e.target.value === 'custom'); }}
          className="bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs focus:border-forge-accent focus:outline-none">
          {Object.entries(PRESETS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <button onClick={() => onCropAll(cropParams)} disabled={cropping}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-forge-surface border border-forge-border rounded hover:border-forge-accent text-forge-muted hover:text-forge-accent disabled:opacity-40 text-xs transition-colors">
          {cropping ? <Spinner /> : <Grid className="w-3 h-3" />} Crop All
        </button>
        {selectedCount > 0 && (
          <button onClick={() => onCropSelected(cropParams)} disabled={cropping}
            className="flex items-center gap-1.5 px-2.5 py-1 bg-forge-surface border border-forge-border rounded hover:border-forge-accent text-forge-muted hover:text-forge-accent disabled:opacity-40 text-xs transition-colors">
            <Grid className="w-3 h-3" /> Crop Selected ({selectedCount})
          </button>
        )}
        {cropResult && (
          <span className="text-green-400 flex items-center gap-1 text-xs">
            <Check className="w-3 h-3" />
            {cropResult.cropped?.length} cropped · {cropResult.skipped?.length} already correct
            {cropResult.errors?.length > 0 && (
              <span className="text-red-400 ml-1">{cropResult.errors.length} errors</span>
            )}
            {cropResult.buckets_used?.length > 0 && (
              <span className="text-forge-muted ml-1">→ {cropResult.buckets_used.join(', ')}</span>
            )}
          </span>
        )}
      </div>
      {showCustom && (
        <div className="grid grid-cols-3 gap-2 px-3 pb-3 border-t border-forge-border pt-2">
          {[['Min Size', minSize, setMinSize], ['Max Size', maxSize, setMaxSize], ['Step', step, setStep]].map(([label, val, setter]) => (
            <div key={label}>
              <label className="text-[10px] text-forge-muted uppercase">{label}</label>
              <input type="number" step="64" value={val} onChange={e => setter(+e.target.value)}
                className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
