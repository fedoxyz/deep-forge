import React, { useState, useEffect, memo } from 'react';
import { Trash2, Check, Save, ChevronDown, ChevronUp, Image as ImageIcon, Square } from 'lucide-react';
import { updateCaption } from '../../utils/api';

function Spinner() {
  return <div className="w-3 h-3 border border-forge-accent border-t-transparent rounded-full animate-spin" />;
}

export default memo(function ImageCard({
  entry, thumbnail, isSelected, hasActiveSelection,
  onSelect, onDelete, onPreview, onAnnotate, datasetId, onCaptionUpdate, dsType, maskBust
}) {
  const hasCaption = entry.has_caption_file && entry.caption;
  const [showCaption, setShowCaption] = useState(false);
  const [editing, setEditing] = useState(false);
  const [caption, setCaption] = useState(entry.caption || '');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => { setCaption(entry.caption || ''); }, [entry.caption]);

  const handleSave = async (e) => {
    e?.stopPropagation();
    setSaving(true);
    try {
      await updateCaption(datasetId, entry.filename, caption);
      onCaptionUpdate?.(entry.filename, caption);
      setSaved(true);
      setTimeout(() => { setSaved(false); setEditing(false); }, 1200);
    } catch (err) { console.error('Save failed', err); }
    setSaving(false);
  };

  const handleKeyDown = (e) => {
    e.stopPropagation();
    if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); handleSave(); }
    if (e.key === 'Escape') { setCaption(entry.caption || ''); setEditing(false); }
  };

  return (
    <div className={`relative group rounded-lg overflow-hidden border transition-all flex flex-col ${
      isSelected ? 'border-forge-accent ring-1 ring-forge-accent/30' : 'border-forge-border hover:border-forge-accent/40'
    }`}>
      {/* Image area */}
      <div
        className="aspect-square bg-forge-surface flex items-center justify-center overflow-hidden relative"
        onClick={() => hasActiveSelection ? onSelect(entry) : onPreview?.(entry)}
        style={{ cursor: hasActiveSelection ? 'pointer' : 'zoom-in' }}
      >
        {thumbnail
          ? <img src={`data:image/jpeg;base64,${thumbnail}`} alt={entry.filename}
              className="w-full h-full object-cover" loading="lazy" />
          : <ImageIcon className="w-8 h-8 text-forge-muted/20" />
        }
      {(dsType === 'detection' || dsType === 'segmentation') && entry.has_caption_file && (
        <img
          src={`/api/datasets/${datasetId}/mask/${entry.index}?t=${maskBust ?? 0}`}
          alt="mask"
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
          onError={e => { e.target.style.display = 'none'; }}
        />
      )}

        {!hasCaption && dsType === 'caption' && (
          <span className="absolute top-1.5 left-7 bg-yellow-500/80 text-black text-[9px] font-bold px-1 py-0.5 rounded z-10">
            NO TXT
          </span>
        )}
        {(dsType === 'detection' || dsType === 'segmentation') && !entry.has_caption_file && (
          <span className="absolute top-1.5 left-7 bg-orange-500/80 text-black text-[9px] font-bold px-1 py-0.5 rounded z-10">
            NO LABELS
          </span>
        )}

        {/* Action buttons */}
        <div className="absolute top-1.5 right-1.5 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
          {onAnnotate && (dsType === 'detection' || dsType === 'segmentation') && (
            <button onClick={e => { e.stopPropagation(); onAnnotate(entry); }}
              title="Annotate"
              className="bg-forge-accent/80 text-black p-1 rounded hover:bg-forge-accent transition-colors">
              <Square className="w-3 h-3" />
            </button>
          )}
          <button onClick={e => { e.stopPropagation(); onDelete?.(entry); }}
            className="bg-red-500/70 text-white p-1 rounded hover:bg-red-500/90 transition-colors">
            <Trash2 className="w-3 h-3" />
          </button>
        </div>

        {/* Filename overlay */}
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-1.5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <p className="text-[10px] text-white/80 truncate font-mono">{entry.filename}</p>
          <p className="text-[10px] text-white/50">{entry.width}×{entry.height}</p>
        </div>

        {/* Checkbox */}
        <div onClick={e => { e.stopPropagation(); onSelect(entry); }}
          className={`absolute top-1.5 left-1.5 w-5 h-5 rounded border flex items-center justify-center cursor-pointer transition-all z-10 ${
            isSelected
              ? 'bg-forge-accent border-forge-accent opacity-100'
              : 'border-white/30 bg-black/30 opacity-0 group-hover:opacity-100'
          }`}>
          {isSelected && <Check className="w-3 h-3 text-black" />}
        </div>
      </div>

      {/* Caption section */}
      {/* Caption / Class section */}
      <div className="border-t border-forge-border bg-forge-bg/60 flex flex-col">
        {dsType === 'classification' ? (
          <div className="flex items-center gap-2 px-2 py-1.5">
            <span className="text-[9px] uppercase tracking-wide text-forge-muted/50">class</span>
            <span className="px-1.5 py-0.5 bg-forge-accent/10 border border-forge-accent/20 text-forge-accent text-[10px] font-mono rounded truncate">
              {entry.caption || '—'}
            </span>
          </div>
        ) : (dsType === 'detection' || dsType === 'segmentation') ? (
          // ── Detection / Segmentation: show label count ──
          <div className="flex items-center gap-2 px-2 py-1.5">
            <span className="text-[9px] uppercase tracking-wide text-forge-muted/50">labels</span>
            {entry.has_caption_file ? (
              <span className="px-1.5 py-0.5 bg-forge-accent/10 border border-forge-accent/20 text-forge-accent text-[10px] font-mono rounded">
                {entry.label_count != null ? `${entry.label_count} obj` : 'labeled'}
              </span>
            ) : (
              <span className="px-1.5 py-0.5 bg-orange-500/10 border border-orange-500/20 text-orange-400 text-[10px] font-mono rounded">
                no labels
              </span>
            )}
          </div>
        ) : (
          // ── Caption: existing collapsible editor ──
          <>
            <button onClick={() => setShowCaption(v => !v)}
              className="flex items-center justify-between px-2 py-1 w-full hover:bg-white/[0.03] transition-colors text-left">
              <span className={`text-[10px] truncate flex-1 mr-1 ${hasCaption ? 'text-forge-muted' : 'text-yellow-500/70 italic'}`}>
                {caption ? caption.slice(0, 60) + (caption.length > 60 ? '…' : '') : 'No caption'}
              </span>
              <span className="shrink-0 text-forge-muted/40">
                {showCaption ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </span>
            </button>
            {showCaption && (
              <div className="px-2 pb-2 space-y-1.5" onClick={e => e.stopPropagation()}>
                <textarea value={caption} onChange={e => { setCaption(e.target.value); setEditing(true); }}
                  onKeyDown={handleKeyDown} rows={3} placeholder="Enter caption…"
                  className="w-full bg-forge-surface border border-forge-border rounded px-2 py-1.5 text-[11px] font-mono resize-none focus:border-forge-accent focus:outline-none focus:ring-1 focus:ring-forge-accent/20" />
                <div className="flex items-center justify-between">
                  <span className="text-[9px] text-forge-muted/40">Ctrl+S · Esc to cancel</span>
                  <div className="flex items-center gap-1.5">
                    {saved && <span className="text-[10px] text-green-400 flex items-center gap-1"><Check className="w-2.5 h-2.5" /> Saved</span>}
                    {editing && !saved && (
                      <button onClick={handleSave} disabled={saving}
                        className="flex items-center gap-1 px-2 py-1 bg-forge-accent text-black rounded text-[10px] font-medium disabled:opacity-40">
                        {saving ? <Spinner /> : <Save className="w-2.5 h-2.5" />} Save
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
)
