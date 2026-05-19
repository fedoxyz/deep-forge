import React, { useState } from 'react';
import { X, Save, ChevronLeft, ChevronRight, Image as ImageIcon } from 'lucide-react';
import { updateCaptionsBatch } from '../../utils/api';

function Spinner() {
  return <div className="w-4 h-4 border-2 border-forge-accent border-t-transparent rounded-full animate-spin" />;
}

export default function BatchCaptionModal({ entries, thumbnails, datasetId, onDone, onClose }) {
  const [captions, setCaptions] = useState(
    () => Object.fromEntries(entries.map(e => [e.index, e.caption || '']))
  );
  const [saving, setSaving] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(0);

  const current = entries[currentIdx];
  const thumbData = thumbnails[String(current?.index)]?.thumbnail;

  const handleSaveAll = async () => {
    setSaving(true);
    try {
      const updates = {};
      for (const [idx, cap] of Object.entries(captions)) updates[idx] = cap;
      await updateCaptionsBatch(datasetId, updates);
      onDone();
    } catch (e) { console.error('Batch save failed:', e); }
    setSaving(false);
  };

  if (!current) return null;

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-forge-surface border border-forge-border rounded-xl w-[700px] max-h-[85vh] flex flex-col overflow-hidden"
        onClick={e => e.stopPropagation()}>
        
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-forge-border">
          <h2 className="text-base font-bold">Caption {entries.length} Image{entries.length > 1 ? 's' : ''}</h2>
          <div className="flex items-center gap-2">
            <button onClick={handleSaveAll} disabled={saving}
              className="flex items-center gap-1.5 px-4 py-2 bg-forge-accent text-black rounded text-sm font-medium disabled:opacity-40">
              {saving ? <Spinner /> : <Save className="w-4 h-4" />} Save All
            </button>
            <button onClick={onClose} className="text-forge-muted hover:text-forge-text p-1">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Nav bar */}
        <div className="flex items-center justify-between px-5 py-2 bg-forge-bg/50 border-b border-forge-border">
          <button onClick={() => setCurrentIdx(i => Math.max(0, i - 1))} disabled={currentIdx === 0}
            className="p-1 text-forge-muted hover:text-forge-accent disabled:opacity-30">
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-xs text-forge-muted font-mono">
            {currentIdx + 1} / {entries.length} — {current.filename}
          </span>
          <button onClick={() => setCurrentIdx(i => Math.min(entries.length - 1, i + 1))}
            disabled={currentIdx >= entries.length - 1}
            className="p-1 text-forge-muted hover:text-forge-accent disabled:opacity-30">
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="flex gap-4">
            <div className="w-64 shrink-0">
              {thumbData
                ? <img src={`data:image/jpeg;base64,${thumbData}`} alt={current.filename}
                    className="w-full rounded-lg object-contain max-h-64" />
                : <div className="w-full h-48 bg-forge-bg rounded-lg flex items-center justify-center">
                    <ImageIcon className="w-8 h-8 text-forge-muted/20" />
                  </div>
              }
              <p className="text-[10px] text-forge-muted mt-1 font-mono text-center">
                {current.width}×{current.height}
              </p>
            </div>
            <div className="flex-1">
              <label className="text-xs text-forge-muted uppercase tracking-wide mb-1 block">Caption</label>
              <textarea value={captions[current.index] || ''}
                onChange={e => setCaptions(prev => ({ ...prev, [current.index]: e.target.value }))}
                placeholder="Enter caption..."
                className="w-full h-40 bg-forge-bg border border-forge-border rounded-lg px-3 py-2 text-sm font-mono resize-none focus:border-forge-accent focus:outline-none" />
              <p className="text-[10px] text-forge-muted/50 mt-1">
                {(captions[current.index] || '').split(/\s+/).filter(Boolean).length} words
              </p>
            </div>
          </div>
        </div>

        {/* Filmstrip */}
        <div className="px-5 py-3 border-t border-forge-border bg-forge-bg/50 overflow-x-auto flex gap-1.5">
          {entries.map((e, i) => {
            const t = thumbnails[String(e.index)]?.thumbnail;
            return (
              <button key={e.index} onClick={() => setCurrentIdx(i)}
                className={`w-10 h-10 rounded overflow-hidden border-2 shrink-0 transition-colors ${
                  i === currentIdx ? 'border-forge-accent' : 'border-transparent hover:border-forge-accent/40'
                }`}>
                {t
                  ? <img src={`data:image/jpeg;base64,${t}`} alt="" className="w-full h-full object-cover" />
                  : <div className="w-full h-full bg-forge-surface" />
                }
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
