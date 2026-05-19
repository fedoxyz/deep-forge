import React, { useState, useEffect, } from 'react';
import {
  Save, X, Check,
  Image as ImageIcon,
  MessageSquare, Zap, Maximize2,
} from 'lucide-react';
import {
  updateCaption,
  captionSingle,
} from '../../utils/api';

export default function CaptionPanel({ entry, thumbnail, datasetId, onUpdate, onPreview }) {
  const [caption, setCaption] = useState(entry?.caption || '');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [previewPinned, setPreviewPinned] = useState(false);
  const [hovering, setHovering] = useState(false);
  useEffect(() => { setCaption(entry?.caption || ''); setSaved(false); setPreviewPinned(false); }, [entry?.index]);
  const handleSave = async () => {
    if (!entry) return; setSaving(true);
    try { await updateCaption(datasetId, entry.filename, caption); onUpdate(entry.index, caption); setSaved(true); setTimeout(() => setSaved(false), 2000); } catch (e) { console.error('Save failed:', e); }
    setSaving(false);
  };
  const handleKeyDown = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); handleSave(); } };
  if (!entry) return <div className="h-full flex items-center justify-center text-forge-muted text-sm">Select an image to edit its caption</div>;
  const showLargePreview = previewPinned || hovering;
  const thumbSrc = thumbnail ? `data:image/jpeg;base64,${thumbnail}` : null;
  return (
    <div className="h-full flex flex-col">
      <div className="relative mb-3">
        <div className={`bg-forge-surface rounded-lg overflow-hidden cursor-pointer transition-all duration-200 ${showLargePreview ? 'max-h-[400px]' : 'max-h-36'}`}
          onMouseEnter={() => setHovering(true)} onMouseLeave={() => setHovering(false)} onClick={() => setPreviewPinned(!previewPinned)}>
          {thumbSrc ? <img src={thumbSrc} alt={entry.filename} className={`w-full transition-all duration-200 ${showLargePreview ? 'object-contain max-h-[400px]' : 'object-cover h-36'}`} />
            : <div className="h-36 flex items-center justify-center"><ImageIcon className="w-10 h-10 text-forge-muted/20" /></div>}
        </div>
        {previewPinned && <button onClick={() => setPreviewPinned(false)} className="absolute top-2 right-2 bg-black/60 text-white p-1 rounded hover:bg-black/80 transition-colors" title="Unpin"><X className="w-3 h-3" /></button>}
        {!showLargePreview && thumbSrc && <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity pointer-events-none"><span className="bg-black/60 text-white text-[10px] px-2 py-1 rounded">Hover to expand · Click to pin</span></div>}
        {thumbSrc && <button onClick={() => onPreview?.(entry)} className="absolute bottom-2 right-2 bg-black/60 text-white p-1 rounded hover:bg-black/80 transition-colors" title="Full size"><Maximize2 className="w-3 h-3" /></button>}
      </div>
      <div className="flex items-center justify-between text-xs text-forge-muted mb-2"><span className="font-mono truncate">{entry.filename}</span><span>{entry.width}x{entry.height}</span></div>
      <div className="flex-1 flex flex-col min-h-0">
      {entry.image_path && (
        <button
          onClick={async () => {
            try {
              const { caption: generated } = await captionSingle(entry.image_path);
              setCaption(generated); setSaved(false);
            } catch (e) { console.error('Auto-caption failed:', e); }
          }}
          className="flex items-center gap-1.5 px-2.5 py-1 mb-2 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-[10px] hover:bg-forge-accent/20 transition-colors w-full justify-center"
        >
          <Zap className="w-3 h-3" /> Auto-Caption This Image
        </button>
      )}
        <label className="text-xs text-forge-muted uppercase tracking-wide mb-1 flex items-center gap-1.5"><MessageSquare className="w-3 h-3" /> Caption</label>
        <textarea value={caption} onChange={(e) => { setCaption(e.target.value); setSaved(false); }} onKeyDown={handleKeyDown} placeholder="Enter caption for this image..."
          className="flex-1 min-h-[100px] bg-forge-bg border border-forge-border rounded-lg px-3 py-2 text-sm font-mono resize-none focus:border-forge-accent focus:outline-none focus:ring-1 focus:ring-forge-accent/20" />
        <div className="flex items-center justify-between mt-2">
          <span className="text-[10px] text-forge-muted/50">{caption.split(/\s+/).filter(Boolean).length} words · Ctrl+S</span>
          <div className="flex items-center gap-2">
            {saved && <span className="text-xs text-green-400 flex items-center gap-1"><Check className="w-3 h-3" /> Saved</span>}
            <button onClick={handleSave} disabled={saving || caption === (entry.caption || '')} className="flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent text-black rounded text-xs font-medium disabled:opacity-40 hover:bg-forge-accent/90 transition-colors">
              {saving ? <Spinner size="sm" /> : <Save className="w-3 h-3" />} Save</button>
          </div>
        </div>
      </div>
    </div>
  );
}

