import React from 'react';
import { Save, Camera, ChevronDown, ImageIcon } from 'lucide-react';
import { requestSaveNow, requestSample } from '../../utils/api';

export default function TrainingControls({ showSamplePanel, setShowSamplePanel, sampleMsg, setSampleMsg,
showSamplingControls = true}) {
  return (
    <div className="flex items-center gap-2 mt-3 border-t border-forge-border pt-3">
      {/* Manual Save */}
      <button
        onClick={async () => {
          try {
            await requestSaveNow();
            setSampleMsg('Save requested — will save at next step');
            setTimeout(() => setSampleMsg(''), 3000);
          } catch (e) { setSampleMsg('Error: ' + e.message); }
        }}
        className="flex items-center gap-1.5 px-3 py-2 bg-forge-surface border border-forge-border rounded-lg text-sm hover:border-forge-accent transition-colors"
      >
        <Save className="w-4 h-4" /> Save Now
      </button>

      {/* Manual Sample (toggle panel) */}
      <button
        onClick={() => setShowSamplePanel(!showSamplePanel)}
        className="flex items-center gap-1.5 px-3 py-2 bg-forge-surface border border-forge-border rounded-lg text-sm hover:border-blue-400/50 transition-colors"
      >
        <Camera className="w-4 h-4" /> Sample
        <ChevronDown className={`w-3 h-3 transition-transform ${showSamplePanel ? 'rotate-180' : ''}`} />
      </button>

      {/* Quick Sample */}
      <button
        onClick={async () => {
          try {
            await requestSample({});
            setSampleMsg('Sample requested — generating...');
            setTimeout(() => setSampleMsg(''), 5000);
          } catch (e) { setSampleMsg('Error: ' + e.message); }
        }}
        className="flex items-center gap-1.5 px-3 py-2 bg-blue-500/10 text-blue-400 border border-blue-500/30 rounded-lg text-sm hover:bg-blue-500/20 transition-colors"
      >
        <ImageIcon className="w-4 h-4" /> Quick Sample
      </button>
      {sampleMsg && <span className="text-xs text-forge-muted ml-2">{sampleMsg}</span>}
    </div>
  );
}
