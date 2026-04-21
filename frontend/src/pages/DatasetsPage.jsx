import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  FolderOpen, Upload, Trash2, Search, Filter, Edit3, Save, X, Check,
  ChevronLeft, ChevronRight, BarChart3, Eye, Image as ImageIcon,
  AlertTriangle, RefreshCw, Sparkles, Tag, Grid, Layers, Plus,
  MessageSquare, ChevronDown, ChevronUp, FileText, Zap, Maximize2,
} from 'lucide-react';
import {
  loadDataset, getLoadedDatasets, unloadDataset, getDatasetEntries,
  getThumbnailsBatch, updateCaption, updateCaptionsBatch,
  analyzeDatasetConcepts, getConceptImages,
  createDataset, uploadDatasetFiles, deleteDatasetFile, deleteDatasetFileBatch,
  loadVisionModel, unloadVisionModel, getVisionModelStatus,
  captionSingle, startCaptionBatch, getCaptionBatchStatus,
  stopCaptionBatch, extractConceptsLLM, getVisionLoadStatus,
  cropDatasetToBuckets
} from '../utils/api';

function Badge({ children, color = 'accent' }) {
  const colors = { accent: 'bg-forge-accent/10 text-forge-accent', warn: 'bg-yellow-500/10 text-yellow-400', error: 'bg-red-500/10 text-red-400', success: 'bg-green-500/10 text-green-400', muted: 'bg-white/5 text-forge-muted' };
  return <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${colors[color] || colors.muted}`}>{children}</span>;
}
function Spinner({ size = 'md' }) { return <RefreshCw className={`${size === 'sm' ? 'w-4 h-4' : 'w-6 h-6'} animate-spin text-forge-accent`} />; }
function EmptyState({ icon: Icon = FolderOpen, title, subtitle, action }) {
  return (<div className="flex flex-col items-center justify-center py-16 text-center"><Icon className="w-12 h-12 text-forge-muted/30 mb-4" /><p className="text-sm text-forge-muted mb-1">{title}</p>{subtitle && <p className="text-xs text-forge-muted/60 mb-4">{subtitle}</p>}{action}</div>);
}

function ImagePreviewModal({ imageSrc, filename, onClose }) {
  if (!imageSrc) return null;
  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8" onClick={onClose}>
      <div className="relative max-w-[90vw] max-h-[90vh]" onClick={e => e.stopPropagation()}>
        <button onClick={onClose} className="absolute -top-3 -right-3 z-10 bg-forge-surface border border-forge-border rounded-full p-1.5 hover:bg-forge-accent/20 transition-colors"><X className="w-4 h-4" /></button>
        <img src={imageSrc} alt={filename} className="max-w-full max-h-[85vh] object-contain rounded-lg" />
        <p className="text-center text-xs text-forge-muted mt-2 font-mono">{filename}</p>
      </div>
    </div>
  );
}

function CreateDatasetModal({ onCreated, onClose }) {
  const [name, setName] = useState('');
  const [baseDir, setBaseDir] = useState('/workspace/datasets');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const handleCreate = async () => {
    if (!name.trim()) return; setCreating(true); setError('');
    try { const result = await createDataset(name.trim(), baseDir); onCreated(result); } catch (e) { setError(e.message); }
    setCreating(false);
  };
  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="bg-forge-surface border border-forge-border rounded-xl p-6 w-96 space-y-4" onClick={e => e.stopPropagation()}>
        <h2 className="text-lg font-bold">Create Dataset</h2>
        <div className="space-y-1"><label className="text-xs text-forge-muted uppercase tracking-wide">Dataset Name</label>
          <input value={name} onChange={e => setName(e.target.value)} placeholder="my_dataset" onKeyDown={e => e.key === 'Enter' && handleCreate()} className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none" /></div>
        <div className="space-y-1"><label className="text-xs text-forge-muted uppercase tracking-wide">Base Directory</label>
          <input value={baseDir} onChange={e => setBaseDir(e.target.value)} className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none" />
          <p className="text-[10px] text-forge-muted/50">Will create: {baseDir}/{name || '...'}</p></div>
        {error && <p className="text-xs text-red-400">{error}</p>}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 text-sm text-forge-muted hover:text-forge-text">Cancel</button>
          <button onClick={handleCreate} disabled={creating || !name.trim()} className="flex items-center gap-2 px-4 py-2 bg-forge-accent text-black rounded text-sm font-medium disabled:opacity-40">
            {creating ? <Spinner size="sm" /> : <FolderOpen className="w-4 h-4" />} Create</button>
        </div>
      </div>
    </div>
  );
}

function UploadZone({ datasetId, onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const inputRef = useRef(null);
  const handleFiles = async (fileList) => {
    if (!fileList || !fileList.length || !datasetId) return;
    setUploading(true); setProgress(0); setResult(null);
    try { const res = await uploadDatasetFiles(datasetId, Array.from(fileList), setProgress); setResult(res); onUploaded?.(); }
    catch (e) { setResult({ errors: [e.message], uploaded: [] }); }
    setUploading(false); setTimeout(() => setResult(null), 5000);
  };
  return (
    <div className="space-y-2">
      <div onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={(e) => { e.preventDefault(); setDragging(false); }}
        onClick={() => !uploading && inputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg px-4 py-4 text-center cursor-pointer transition-all ${dragging ? 'border-forge-accent bg-forge-accent/5 scale-[1.01]' : uploading ? 'border-forge-border opacity-70 cursor-wait' : 'border-forge-border hover:border-forge-accent/40'}`}>
        <input ref={inputRef} type="file" multiple accept="image/*,.txt,.png,.jpg,.jpeg,.webp,.bmp" className="hidden" onChange={e => { handleFiles(e.target.files); e.target.value = ''; }} />
        <div className="flex items-center justify-center gap-3">
          <Upload className="w-5 h-5 text-forge-muted/40" />
          <div className="text-left">
            <p className="text-sm text-forge-muted">{uploading ? `Uploading... ${progress}%` : 'Drop images & caption files here, or click to browse'}</p>
            <p className="text-[10px] text-forge-muted/40">.png .jpg .jpeg .webp .bmp — pair with .txt files for captions (same filename)</p>
          </div>
        </div>
      </div>
      {uploading && <div className="h-1 bg-forge-surface rounded-full overflow-hidden"><div className="h-full bg-forge-accent transition-all rounded-full" style={{ width: `${progress}%` }} /></div>}
      {result && <div className="text-xs flex items-center gap-3">
        {result.uploaded?.length > 0 && <span className="text-green-400 flex items-center gap-1"><Check className="w-3 h-3" /> {result.uploaded.length} file(s) uploaded</span>}
        {result.errors?.length > 0 && <span className="text-red-400 flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> {result.errors.length} error(s)</span>}
      </div>}
    </div>
  );
}

function ImageCard({ entry, thumbnail, isSelected, onSelect, onDelete, onPreview, datasetId, onCaptionUpdate, hasActiveSelection }) {
  const hasCaption = entry.has_caption_file && entry.caption;
  const [showCaption, setShowCaption] = useState(false);
  const [editing, setEditing] = useState(false);
  const [caption, setCaption] = useState(entry.caption || '');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Sync caption if entry changes externally
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
        onClick={() => {
          if (hasActiveSelection) {
            onSelect(entry);
          } else {
            onPreview?.(entry);
          }
        }}
        style={{ cursor: hasActiveSelection ? 'pointer' : 'zoom-in' }}
      >
        {thumbnail
          ? <img src={`data:image/jpeg;base64,${thumbnail}`} alt={entry.filename} className="w-full h-full object-cover" loading="lazy" />
          : <ImageIcon className="w-8 h-8 text-forge-muted/20" />}

        {/* Top badges */}
        {!hasCaption && (
          <span className="absolute top-1.5 left-7 bg-yellow-500/80 text-black text-[9px] font-bold px-1 py-0.5 rounded z-10">NO TXT</span>
        )}

        {/* Delete button */}
        <div className="absolute top-1.5 right-1.5 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
          <button onClick={(e) => { e.stopPropagation(); onDelete?.(entry); }}
            className="bg-red-500/70 text-white p-1 rounded hover:bg-red-500/90 transition-colors">
            <Trash2 className="w-3 h-3" />
          </button>
        </div>

        {/* Filename on hover */}
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-1.5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <p className="text-[10px] text-white/80 truncate font-mono">{entry.filename}</p>
          <p className="text-[10px] text-white/50">{entry.width}×{entry.height}</p>
        </div>

        {/* Checkbox */}
        <div
          onClick={(e) => { e.stopPropagation(); onSelect(entry); }}
          className={`absolute top-1.5 left-1.5 w-5 h-5 rounded border flex items-center justify-center cursor-pointer transition-all z-10 ${
            isSelected ? 'bg-forge-accent border-forge-accent opacity-100' : 'border-white/30 bg-black/30 opacity-0 group-hover:opacity-100'
          }`}
        >
          {isSelected && <Check className="w-3 h-3 text-black" />}
        </div>
      </div>

      {/* Caption section — always visible below image */}
      <div className="border-t border-forge-border bg-forge-bg/60 flex flex-col">
        {/* Caption toggle bar */}
        <button
          onClick={() => setShowCaption(v => !v)}
          className="flex items-center justify-between px-2 py-1 w-full hover:bg-white/[0.03] transition-colors text-left"
        >
          <span className={`text-[10px] truncate flex-1 mr-1 ${hasCaption ? 'text-forge-muted' : 'text-yellow-500/70 italic'}`}>
            {caption ? caption.slice(0, 60) + (caption.length > 60 ? '…' : '') : 'No caption'}
          </span>
          <span className="shrink-0 text-forge-muted/40">
            {showCaption ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </span>
        </button>

        {/* Expanded editor */}
        {showCaption && (
          <div className="px-2 pb-2 space-y-1.5" onClick={e => e.stopPropagation()}>
            <textarea
              value={caption}
              onChange={e => { setCaption(e.target.value); setEditing(true); }}
              onKeyDown={handleKeyDown}
              rows={3}
              placeholder="Enter caption..."
              className="w-full bg-forge-surface border border-forge-border rounded px-2 py-1.5 text-[11px] font-mono resize-none focus:border-forge-accent focus:outline-none focus:ring-1 focus:ring-forge-accent/20"
            />
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-forge-muted/40">Ctrl+S · Esc to cancel</span>
              <div className="flex items-center gap-1.5">
                {saved && <span className="text-[10px] text-green-400 flex items-center gap-1"><Check className="w-2.5 h-2.5" /> Saved</span>}
                {editing && !saved && (
                  <button
                    onClick={handleSave}
                    disabled={saving}
                    className="flex items-center gap-1 px-2 py-1 bg-forge-accent text-black rounded text-[10px] font-medium disabled:opacity-40"
                  >
                    {saving ? <Spinner size="sm" /> : <Save className="w-2.5 h-2.5" />} Save
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function CaptionPanel({ entry, thumbnail, datasetId, onUpdate, onPreview }) {
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

const CATEGORY_COLORS = { attribute: 'bg-blue-500/10 text-blue-400 border-blue-500/20', action: 'bg-green-500/10 text-green-400 border-green-500/20', setting: 'bg-amber-500/10 text-amber-400 border-amber-500/20', style: 'bg-purple-500/10 text-purple-400 border-purple-500/20', composition: 'bg-pink-500/10 text-pink-400 border-pink-500/20', unknown: 'bg-white/5 text-forge-muted border-white/10' };

function ConceptBar({ concept, maxCount, totalImages, onClick, isActive }) {
  const pct = (concept.count / totalImages * 100).toFixed(1);
  const barWidth = (concept.count / maxCount * 100);
  const catColor = CATEGORY_COLORS[concept.category] || CATEGORY_COLORS.unknown;
  const ratio = concept.count / totalImages;
  let barColor = 'bg-forge-accent/40 group-hover:bg-forge-accent/60';
  if (ratio > 0.6) barColor = 'bg-yellow-500/50 group-hover:bg-yellow-500/70';
  if (ratio <= 0.1) barColor = 'bg-forge-muted/30 group-hover:bg-forge-muted/50';
  return (
    <button onClick={() => onClick(concept)} className={`w-full text-left px-2.5 py-1.5 rounded-lg transition-all group ${isActive ? 'bg-forge-accent/10 ring-1 ring-forge-accent/30' : 'hover:bg-white/[0.03]'}`}>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-1.5 min-w-0"><span className={`text-[9px] px-1 py-0.5 rounded border shrink-0 ${catColor}`}>{concept.category}</span><span className="text-xs truncate">{concept.phrase}</span></div>
        <div className="flex items-center gap-1.5 shrink-0 ml-2"><span className="text-[10px] text-forge-muted">{pct}%</span><span className="text-xs font-mono text-forge-accent">{concept.count}</span></div>
      </div>
      <div className="h-1 bg-forge-surface rounded-full overflow-hidden"><div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${barWidth}%` }} /></div>
    </button>
  );
}

function ConceptAnalysisPanel({ datasetId, totalImages, onSelectConcept }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeConcept, setActiveConcept] = useState(null);
  const [filterCat, setFilterCat] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [minFreq, setMinFreq] = useState(2);
  const [maxNgram, setMaxNgram] = useState(3);
  const [showSettings, setShowSettings] = useState(false);
  const runAnalysis = async () => {
    setLoading(true);
    try { const result = await analyzeDatasetConcepts(datasetId, { min_frequency: minFreq, max_ngram: maxNgram }); setAnalysis(result); setActiveConcept(null); }
    catch (e) { console.error('Analysis failed:', e); }
    setLoading(false);
  };
  const filteredConcepts = useMemo(() => {
    if (!analysis) return [];
    let concepts = analysis.concepts;
    if (filterCat !== 'all') concepts = concepts.filter(c => c.category === filterCat);
    if (searchTerm) { const lower = searchTerm.toLowerCase(); concepts = concepts.filter(c => c.phrase.includes(lower)); }
    return concepts;
  }, [analysis, filterCat, searchTerm]);
  const maxCount = useMemo(() => Math.max(1, ...filteredConcepts.map(c => c.count)), [filteredConcepts]);
  const handleConceptClick = (concept) => { const next = concept.phrase === activeConcept?.phrase ? null : concept; setActiveConcept(next); if (next) onSelectConcept?.(next); };
  const catDist = analysis?.category_distribution || {};
  const triggerWords = analysis?.trigger_words_detected || [];
  return (
    <div className="h-full flex flex-col">
      <div className="flex flex-col items-left justify-between mb-1.5 gap-1.5">
        <h3 className="text-sm font-medium flex items-center gap-1.5"><Sparkles className="w-4 h-4 text-forge-accent" /> Concepts</h3>
        <div className="flex items-center gap-1.5">
          <button onClick={() => setShowSettings(!showSettings)} className="text-[10px] text-forge-muted hover:text-forge-accent px-2 py-1 rounded">{showSettings ? 'Hide' : 'Settings'}</button>
          <button onClick={runAnalysis} disabled={loading} className="flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent text-black rounded text-xs font-medium disabled:opacity-40">
            {loading ? <Spinner size="sm" /> : <Zap className="w-3 h-3" />} {analysis ? 'Re-run' : 'Analyze'}</button>
            <button
              onClick={async () => {
                setLoading(true);
                try {
                  const result = await extractConceptsLLM(datasetId, ['attributes', 'actions', 'settings', 'style', 'composition']);
                  setAnalysis(prev => ({
                    ...prev || {},
                    concepts: result.concepts.map(c => ({ ...c, image_indices: c.caption_indices || [] })),
                    total_images: result.total_captions,
                    category_distribution: result.concepts.reduce((acc, c) => { acc[c.category] = (acc[c.category] || 0) + 1; return acc; }, {}),
                    trigger_words_detected: [],
                  }));
                } catch (e) { console.error('LLM extraction failed:', e); }
                setLoading(false);
              }}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/10 text-purple-400 border border-purple-500/30 rounded text-xs disabled:opacity-40 hover:bg-purple-500/20 transition-colors"
            >
              <Sparkles className="w-3 h-3" /> LLM Analyze
            </button>
        </div>
      </div>
      {showSettings && <div className="grid grid-cols-2 gap-2 mb-3 p-2.5 bg-forge-surface rounded-lg border border-forge-border">
        <div><label className="text-[10px] text-forge-muted uppercase">Min Frequency</label><input type="number" value={minFreq} min={1} onChange={(e) => setMinFreq(parseInt(e.target.value) || 2)} className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" /></div>
        <div><label className="text-[10px] text-forge-muted uppercase">Max N-gram</label><input type="number" value={maxNgram} min={1} max={5} onChange={(e) => setMaxNgram(parseInt(e.target.value) || 3)} className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" /></div>
      </div>}
      {!analysis && !loading && <EmptyState icon={BarChart3} title="Run analysis to discover concepts" subtitle="Extracts recurring phrases from your captions" />}
      {loading && <div className="flex-1 flex items-center justify-center"><Spinner /></div>}
      {analysis && !loading && <>
        {triggerWords.length > 0 && <div className="mb-2 px-2.5 py-2 bg-purple-500/5 border border-purple-500/15 rounded-lg">
          <p className="text-[10px] text-purple-400 font-medium mb-1">Trigger word(s) detected & filtered:</p>
          <div className="flex flex-wrap gap-1">{triggerWords.map(tw => <span key={tw} className="text-[10px] px-1.5 py-0.5 bg-purple-500/10 text-purple-300 rounded font-mono">{tw}</span>)}</div>
        </div>}
        <div className="flex flex-wrap gap-1 mb-2">
          <button onClick={() => setFilterCat('all')} className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${filterCat === 'all' ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent' : 'border-forge-border text-forge-muted hover:text-forge-text'}`}>All ({analysis.concepts.length})</button>
          {Object.entries(catDist).map(([cat, count]) => <button key={cat} onClick={() => setFilterCat(filterCat === cat ? 'all' : cat)} className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${filterCat === cat ? CATEGORY_COLORS[cat] : 'border-forge-border text-forge-muted hover:text-forge-text'}`}>{cat} ({count})</button>)}
        </div>
        <div className="relative mb-2"><Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-forge-muted" /><input value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} placeholder="Filter concepts..." className="w-full bg-forge-bg border border-forge-border rounded pl-7 pr-2 py-1.5 text-xs focus:border-forge-accent focus:outline-none" /></div>
        <div className="flex-1 overflow-y-auto space-y-0.5 min-h-0">
          {filteredConcepts.map((concept) => <ConceptBar key={concept.phrase} concept={concept} maxCount={maxCount} totalImages={totalImages} onClick={handleConceptClick} isActive={activeConcept?.phrase === concept.phrase} />)}
          {filteredConcepts.length === 0 && <p className="text-xs text-forge-muted py-4 text-center">No concepts match your filters</p>}
        </div>
        {analysis.concepts.length > 0 && <div className="mt-2 pt-2 border-t border-forge-border space-y-1">
          {analysis.concepts.filter(c => c.count / totalImages > 0.6).length > 0 && <div className="flex items-start gap-1.5 text-[10px] text-yellow-400"><AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" /><span>{analysis.concepts.filter(c => c.count / totalImages > 0.6).length} dominant concept(s) at 60%+ — overfitting risk</span></div>}
        </div>}
      </>}
    </div>
  );
}

function SelectionBar({ count, onClearSelection, onDeleteSelected, onCaptionSelected }) {
  if (count === 0) return null;
  return (
    <div className="flex items-center gap-3 px-3 py-2 bg-forge-accent/5 border border-forge-accent/20 rounded-lg">
      <span className="text-xs text-forge-accent font-medium">{count} selected</span>
      <div className="h-3 w-px bg-forge-border" />
      <button onClick={onCaptionSelected} className="flex items-center gap-1.5 text-xs text-forge-muted hover:text-forge-accent transition-colors"><MessageSquare className="w-3 h-3" /> Caption selected</button>
      <button onClick={onDeleteSelected} className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 transition-colors"><Trash2 className="w-3 h-3" /> Delete selected</button>
      <div className="flex-1" />
      <button onClick={onClearSelection} className="text-xs text-forge-muted hover:text-forge-text transition-colors">Clear</button>
    </div>
  );
}

function CropBar({ totalCount, selectedCount, cropping, cropResult, onCropAll, onCropSelected }) {
  const [preset, setPreset] = useState('sdxl');
  const [showCustom, setShowCustom] = useState(false);
  const [minSize, setMinSize] = useState(768);
  const [maxSize, setMaxSize] = useState(1344);
  const [step, setStep]       = useState(64);

  const presets = {
    sd15:     'SD 1.5 (512–768px)',
    sdxl:     'SDXL (768–1344px)',
    flux:     'Flux (1024–2048px)',
    ziturbo:  'Z-Image-Turbo (1024–2048px)',
    xl_fine:  'SDXL fine-tune (32-step)',
    custom:   'Custom…',
  };

  const cropParams = preset === 'custom'
    ? { preset: 'custom', minSize, maxSize, step }
    : { preset };

  return (
    <div className="border border-forge-border rounded-lg bg-forge-surface/50 overflow-hidden">
      <div className="flex items-center gap-3 px-3 py-2 flex-wrap">
        <span className="text-forge-muted flex items-center gap-1 text-xs shrink-0">
          <Grid className="w-3 h-3" /> Bucket Crop
        </span>

        {/* Preset selector */}
        <select
          value={preset}
          onChange={e => { setPreset(e.target.value); setShowCustom(e.target.value === 'custom'); }}
          className="bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs focus:border-forge-accent focus:outline-none"
        >
          {Object.entries(presets).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>

        <button
          onClick={() => onCropAll(cropParams)}
          disabled={cropping}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-forge-surface border border-forge-border rounded hover:border-forge-accent text-forge-muted hover:text-forge-accent transition-colors disabled:opacity-40 text-xs"
        >
          {cropping ? <Spinner size="sm" /> : <Grid className="w-3 h-3" />}
          Crop All
        </button>

        {selectedCount > 0 && (
          <button
            onClick={() => onCropSelected(cropParams)}
            disabled={cropping}
            className="flex items-center gap-1.5 px-2.5 py-1 bg-forge-surface border border-forge-border rounded hover:border-forge-accent text-forge-muted hover:text-forge-accent transition-colors disabled:opacity-40 text-xs"
          >
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
          <div>
            <label className="text-[10px] text-forge-muted uppercase">Min Size</label>
            <input type="number" step="64" value={minSize} onChange={e => setMinSize(+e.target.value)}
              className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" />
          </div>
          <div>
            <label className="text-[10px] text-forge-muted uppercase">Max Size</label>
            <input type="number" step="64" value={maxSize} onChange={e => setMaxSize(+e.target.value)}
              className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" />
          </div>
          <div>
            <label className="text-[10px] text-forge-muted uppercase">Step</label>
            <input type="number" step="8" value={step} onChange={e => setStep(+e.target.value)}
              className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" />
          </div>
        </div>
      )}
    </div>
  );
}

function BatchCaptionModal({ entries, thumbnails, datasetId, onDone, onClose }) {
  const [captions, setCaptions] = useState(() => Object.fromEntries(entries.map(e => [e.index, e.caption || ''])));
  const [saving, setSaving] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(0);
  const current = entries[currentIdx];
  const thumbData = thumbnails[String(current?.index)]?.thumbnail;
  const handleSaveAll = async () => {
    setSaving(true);
    try { const updates = {}; for (const [idx, cap] of Object.entries(captions)) updates[idx] = cap; await updateCaptionsBatch(datasetId, updates); onDone(); }
    catch (e) { console.error('Batch save failed:', e); }
    setSaving(false);
  };
  if (!current) return null;
  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" style={{marginTop: "none"}}  onClick={onClose}>
      <div className="bg-forge-surface border border-forge-border rounded-xl w-[700px] max-h-[85vh] flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 py-4 border-b border-forge-border">
          <h2 className="text-base font-bold">Caption {entries.length} Image{entries.length > 1 ? 's' : ''}</h2>
          <div className="flex items-center gap-2">
            <button onClick={handleSaveAll} disabled={saving} className="flex items-center gap-1.5 px-4 py-2 bg-forge-accent text-black rounded text-sm font-medium disabled:opacity-40">{saving ? <Spinner size="sm" /> : <Save className="w-4 h-4" />} Save All</button>
            <button onClick={onClose} className="text-forge-muted hover:text-forge-text p-1"><X className="w-5 h-5" /></button>
          </div>
        </div>
        <div className="flex items-center justify-between px-5 py-2 bg-forge-bg/50 border-b border-forge-border">
          <button onClick={() => setCurrentIdx(i => Math.max(0, i - 1))} disabled={currentIdx === 0} className="p-1 text-forge-muted hover:text-forge-accent disabled:opacity-30"><ChevronLeft className="w-4 h-4" /></button>
          <span className="text-xs text-forge-muted font-mono">{currentIdx + 1} / {entries.length} — {current.filename}</span>
          <button onClick={() => setCurrentIdx(i => Math.min(entries.length - 1, i + 1))} disabled={currentIdx >= entries.length - 1} className="p-1 text-forge-muted hover:text-forge-accent disabled:opacity-30"><ChevronRight className="w-4 h-4" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">
          <div className="flex gap-4">
            <div className="w-64 shrink-0">
              {thumbData ? <img src={`data:image/jpeg;base64,${thumbData}`} alt={current.filename} className="w-full rounded-lg object-contain max-h-64" />
                : <div className="w-full h-48 bg-forge-bg rounded-lg flex items-center justify-center"><ImageIcon className="w-8 h-8 text-forge-muted/20" /></div>}
              <p className="text-[10px] text-forge-muted mt-1 font-mono text-center">{current.width}x{current.height}</p>
            </div>
            <div className="flex-1">
              <label className="text-xs text-forge-muted uppercase tracking-wide mb-1 block">Caption</label>
              <textarea value={captions[current.index] || ''} onChange={e => setCaptions(prev => ({ ...prev, [current.index]: e.target.value }))} placeholder="Enter caption..."
                className="w-full h-40 bg-forge-bg border border-forge-border rounded-lg px-3 py-2 text-sm font-mono resize-none focus:border-forge-accent focus:outline-none" />
              <p className="text-[10px] text-forge-muted/50 mt-1">{(captions[current.index] || '').split(/\s+/).filter(Boolean).length} words</p>
            </div>
          </div>
        </div>
        <div className="px-5 py-3 border-t border-forge-border bg-forge-bg/50 overflow-x-auto flex gap-1.5">
          {entries.map((e, i) => { const t = thumbnails[String(e.index)]?.thumbnail; return (
            <button key={e.index} onClick={() => setCurrentIdx(i)} className={`w-10 h-10 rounded overflow-hidden border-2 shrink-0 transition-colors ${i === currentIdx ? 'border-forge-accent' : 'border-transparent hover:border-forge-accent/40'}`}>
              {t ? <img src={`data:image/jpeg;base64,${t}`} alt="" className="w-full h-full object-cover" /> : <div className="w-full h-full bg-forge-surface" />}
            </button>); })}
        </div>
      </div>
    </div>
  );
}

function CaptionToolbar({ datasetId, selectedCount, selectedIndices, entries, onRefresh }) {
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelId, setModelId] = useState('huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated');
  const [captionPrompt, setCaptionPrompt] = useState(
    'Describe this image in 2-5 natural language sentences. Include the subjects, their positions and poses, clothing, the setting/background, lighting conditions, and camera angle. Do not mention artistic style, image quality, or aesthetic judgments. Do not reference skin tone. Use correct gender pronouns. Keep it factual and concise.'
  );
  const [batchStatus, setBatchStatus] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.2);
  const pollRef = useRef(null);
  const loadPollRef = useRef(null);

  // Poll model status on mount
  useEffect(() => {
    checkStatus();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (loadPollRef.current) clearInterval(loadPollRef.current);
    };
  }, []);

  const getBackendName = (id) => {
    if (id.toLowerCase().includes('qwen2.5') || id.toLowerCase().includes('qwen2_5')) return 'qwen25vl';
    return 'qwen3vl';
  };

  const checkStatus = async () => {
    try {
      const s = await getVisionModelStatus();
      setModelStatus(s);
    } catch {
      setModelStatus(null);
    }
    // Resume load polling if a job is already in flight
    try {
      const ls = await getVisionLoadStatus();
      if (ls.status === 'loading') {
        setLoading(true);
        resumeLoadPolling();
      } else if (ls.status === 'loaded') {
        setLoading(false);
      }
    } catch {}
  };
  
  const resumeLoadPolling = () => {
    if (loadPollRef.current) return; // already polling
    loadPollRef.current = setInterval(async () => {
      try {
        const s = await getVisionLoadStatus();
        if (s.status === 'loaded') {
          clearInterval(loadPollRef.current);
          loadPollRef.current = null;
          await checkStatus();
          setLoading(false);
        } else if (s.status === 'error') {
          clearInterval(loadPollRef.current);
          loadPollRef.current = null;
          console.error('Model load failed:', s.error);
          setLoading(false);
        }
      } catch (e) {
        clearInterval(loadPollRef.current);
        loadPollRef.current = null;
        setLoading(false);
      }
    }, 2500);
  };

  const handleLoadModel = async () => {
    setLoading(true);
    try {
      const ls = await getVisionLoadStatus();
      if (ls.status === 'loading') {
        resumeLoadPolling(); // already running, just re-attach
        return;
      }
      await loadVisionModel(getBackendName(modelId), modelId);
      resumeLoadPolling();
    } catch (e) {
      console.error('Load request failed:', e);
      setLoading(false);
    }
  };

  const handleUnloadModel = async () => {
    await unloadVisionModel();
    setModelStatus(null);
  };

  const handleCaptionSelected = async () => {
    if (selectedCount === 0) return;
    const indices = Array.from(selectedIndices);
    try {
      await startCaptionBatch(datasetId, { indices, prompt: captionPrompt, maxNewTokens: maxTokens, temperature });
      startPolling();
    } catch (e) { console.error('Caption start failed:', e); }
  };

  const handleCaptionAll = async () => {
    try {
      await startCaptionBatch(datasetId, { prompt: captionPrompt, maxNewTokens: maxTokens, temperature });
      startPolling();
    } catch (e) { console.error('Caption start failed:', e); }
  };

  const handleCaptionSingle = async (entry) => {
    try {
      const { caption } = await captionSingle(entry.image_path, captionPrompt, maxTokens, temperature);
      await updateCaption(datasetId, entry.index, caption);
      onRefresh?.();
    } catch (e) { console.error('Caption failed:', e); }
  };

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const status = await getCaptionBatchStatus();
        setBatchStatus(status);
        if (!status.running) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          onRefresh?.();
        }
      } catch { clearInterval(pollRef.current); pollRef.current = null; }
    }, 1500);
  };

  const handleStop = async () => {
    await stopCaptionBatch();
  };

  const isModelLoaded = modelStatus?.loaded;
  const isRunning = batchStatus?.running;

  return (
    <div className="border border-forge-border rounded-lg bg-forge-surface/50 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-forge-border">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-forge-accent" />
          <span className="text-sm font-medium">Auto-Caption</span>
          {isModelLoaded && <Badge color="success">Model Loaded</Badge>}
          {!isModelLoaded && <Badge color="muted">No Model</Badge>}
        </div>
        <button onClick={() => setShowSettings(!showSettings)}
          className="text-[10px] text-forge-muted hover:text-forge-accent px-2 py-1">
          {showSettings ? 'Hide Settings' : 'Settings'}
        </button>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="px-3 py-3 border-b border-forge-border bg-forge-bg/50 space-y-3">
          <div>
            <label className="text-[10px] text-forge-muted uppercase tracking-wide">Model ID</label>
            <div className="flex gap-2 mt-0.5">
              <input value={modelId} onChange={e => setModelId(e.target.value)}
                className="flex-1 bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-xs font-mono focus:border-forge-accent focus:outline-none" />
              {!isModelLoaded ? (
                <button onClick={handleLoadModel} disabled={loading}
                  className="px-3 py-1.5 bg-forge-accent text-black rounded text-xs font-medium disabled:opacity-40 shrink-0">
                  {loading ? 'Loading...' : 'Load'}
                </button>
              ) : (
                <button onClick={handleUnloadModel}
                  className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded text-xs shrink-0">Unload</button>
              )}
            </div>
            <div className="text-[10px] text-forge-muted mt-0.5">
              Backend: <span className="text-forge-accent font-mono">{getBackendName(modelId)}</span>
            </div>
          </div>
          <div>
            <label className="text-[10px] text-forge-muted uppercase tracking-wide">Caption Prompt</label>
            <textarea value={captionPrompt} onChange={e => setCaptionPrompt(e.target.value)}
              className="w-full mt-0.5 bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-xs font-mono h-20 resize-none focus:border-forge-accent focus:outline-none" />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-forge-muted uppercase">Max Tokens</label>
              <input type="number" value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value) || 256)}
                className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" />
            </div>
            <div>
              <label className="text-[10px] text-forge-muted uppercase">Temperature</label>
              <input type="number" step="0.1" value={temperature} onChange={e => setTemperature(parseFloat(e.target.value) || 0.7)}
                className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" />
            </div>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="px-3 py-2 flex items-center gap-2 flex-wrap">
        {isRunning ? (
          <>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between text-[10px] text-forge-muted mb-1">
                <span>Captioning {batchStatus.current_file}...</span>
                <span>{batchStatus.progress}/{batchStatus.total}</span>
              </div>
              <div className="h-1 bg-forge-bg rounded-full overflow-hidden">
                <div className="h-full bg-forge-accent rounded-full transition-all"
                  style={{ width: `${(batchStatus.progress / Math.max(batchStatus.total, 1)) * 100}%` }} />
              </div>
            </div>
            <button onClick={handleStop} className="px-3 py-1.5 bg-red-500/20 text-red-400 rounded text-xs shrink-0">Stop</button>
          </>
        ) : (
          <>
            <button onClick={handleCaptionAll} disabled={!isModelLoaded}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-xs disabled:opacity-30 hover:bg-forge-accent/20 transition-colors">
              <Zap className="w-3 h-3" /> Caption All Uncaptioned
            </button>
            {selectedCount > 0 && (
              <button onClick={handleCaptionSelected} disabled={!isModelLoaded}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent/10 text-forge-accent border border-forge-accent/30 rounded text-xs disabled:opacity-30 hover:bg-forge-accent/20 transition-colors">
                <Zap className="w-3 h-3" /> Caption Selected ({selectedCount})
              </button>
            )}
            {!isModelLoaded && <span className="text-[10px] text-forge-muted">Load a model in Settings to enable</span>}
          </>
        )}
      </div>
    </div>
  );
}

export default function DatasetsPage({ onNavigate }) {
  const [loadedDatasets, setLoadedDatasets] = useState({});
  const [activeDatasetId, setActiveDatasetId] = useState(null);
  const [loadPath, setLoadPath] = useState('');
  const [loadError, setLoadError] = useState('');
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [entries, setEntries] = useState([]);
  const [totalEntries, setTotalEntries] = useState(0);
  const [page, setPage] = useState(0);
  const [filterMode, setFilterMode] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [searchTrigger, setSearchTrigger] = useState(0);
  const pageSize = 48;
  const [thumbnails, setThumbnails] = useState({});
  const [selectedIndices, setSelectedIndices] = useState(new Set());
  const [editingFilename, setEditingFilename] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showBatchCaption, setShowBatchCaption] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [rightPanel, setRightPanel] = useState('caption');
  const [showRightPanel, setShowRightPanel] = useState(false);
  const [gridSize, setGridSize] = useState(4);

  const [cropping, setCropping] = useState(false);
  const [cropResult, setCropResult] = useState(null);
  
  const handleCropToBuckets = async (selectedOnly = false, cropParams = { preset: 'sdxl' }) => {
    if (!confirm(
      selectedOnly
        ? `Crop ${selectedIndices.size} selected image(s) to nearest training bucket? Originals are backed up to .originals/`
        : `Crop ALL ${totalEntries} images to nearest training bucket? Originals are backed up to .originals/`
    )) return;
    setCropping(true);
    setCropResult(null);
    try {
      const filenames = selectedOnly
        ? entries.filter(e => selectedIndices.has(e.index)).map(e => e.filename)
        : null;
      const result = await cropDatasetToBuckets(activeDatasetId, { filenames, ...cropParams });
      setCropResult(result);
      fetchEntries();
    } catch (e) {
      console.error('Crop failed:', e);
    }
    setCropping(false);
    setTimeout(() => setCropResult(null), 6000);
  };

  useEffect(() => { refreshLoaded(); }, []);
  const refreshLoaded = async () => { try { const { datasets } = await getLoadedDatasets(); setLoadedDatasets(datasets || {}); } catch {} };

  useEffect(() => { if (activeDatasetId) fetchEntries(); }, [activeDatasetId, page, filterMode, searchTrigger]);

  const fetchEntries = async () => {
    if (!activeDatasetId) return;
    try {
      const filterVal = filterMode === 'search' ? searchText : filterMode;
      const result = await getDatasetEntries(activeDatasetId, { offset: page * pageSize, limit: pageSize, filter: filterVal });
      setEntries(result.entries); setTotalEntries(result.total);
      const filenames = result.entries.map(e => e.filename);
      const { thumbnails: thumbs } = await getThumbnailsBatch(activeDatasetId, filenames, 256);
      setThumbnails(prev => ({ ...prev, ...thumbs }));  // now keyed by filename
    } catch (e) { console.error('Failed to fetch entries:', e); }
  };

  const handleLoad = async () => {
    if (!loadPath.trim()) return; setLoadingDataset(true); setLoadError('');
    try { const result = await loadDataset(loadPath.trim()); setActiveDatasetId(result.dataset_id); setPage(0); setSelectedIndices(new Set()); setLoadPath(''); refreshLoaded(); }
    catch (e) { setLoadError(e.message); }
    setLoadingDataset(false);
  };
  const handleUnload = async (dsId) => { await unloadDataset(dsId); if (activeDatasetId === dsId) { setActiveDatasetId(null); setEntries([]); setSelectedIndices(new Set()); } refreshLoaded(); };
  const handleSelect = (entry) => {
    setSelectedIndices(prev => {
      const next = new Set(prev);
      if (next.has(entry.index)) next.delete(entry.index);
      else next.add(entry.index);
      return next;
    });
    setEditingFilename(entry.filename);
    if (!showRightPanel) { setRightPanel('caption'); setShowRightPanel(true); }
  };
  const handleCaptionUpdate = (filename, newCaption) => {
    setEntries(prev => prev.map(e =>
      e.filename === filename ? { ...e, caption: newCaption, has_caption_file: true } : e
    ));
  };
  const handleDeleteSingle = async (entry) => {
    if (!confirm(`Delete ${entry.filename}?`)) return;
    try {
      await deleteDatasetFile(activeDatasetId, entry.filename);
      setEntries(prev => prev.filter(e => e.index !== entry.index));
      setTotalEntries(prev => prev - 1);
      setSelectedIndices(prev => { const next = new Set(prev); next.delete(entry.index); return next; });
      if (editingFilename === entry.filename) setEditingFilename(null);
      setThumbnails(prev => { const next = {...prev}; delete next[entry.filename]; return next; });
      refreshLoaded();
    } catch (e) { console.error('Delete failed:', e); }
  };
  const handleDeleteSelected = async () => {
    if (selectedIndices.size === 0) return;
    const filenames = entries.filter(e => selectedIndices.has(e.index)).map(e => e.filename);
    if (!confirm(`Delete ${filenames.length} image(s)? This cannot be undone.`)) return;
    try { await deleteDatasetFileBatch(activeDatasetId, filenames); setEntries(prev => prev.filter(e => !selectedIndices.has(e.index))); setTotalEntries(prev => prev - filenames.length); setSelectedIndices(new Set()); setEditingFilename(null); refreshLoaded(); }
    catch (e) { console.error('Batch delete failed:', e); }
    const toDeleteFilenames = entries.filter(e => selectedIndices.has(e.index)).map(e => e.filename);
    setThumbnails(prev => {
      const next = {...prev};
      toDeleteFilenames.forEach(f => delete next[f]);
      return next;
    });
  };
  const handleCaptionSelected = () => { if (selectedIndices.size > 0) setShowBatchCaption(true); };
  const handleSearch = () => {
    if (searchText.trim()) setFilterMode('search');
    else setFilterMode(null);
    setPage(0);
    setSearchTrigger(t => t + 1); // always triggers re-fetch
  };

  const handleConceptSelect = (concept) => {
    setSearchText(concept.phrase);
    setFilterMode('search');
    setPage(0);
  };

  const handlePreview = (entry) => {
    setPreviewImage({
      filename: entry.filename,
      src: `/api/datasets/${activeDatasetId}/image/by-filename/${encodeURIComponent(entry.filename)}`,
    });
  };

  const activeDs = activeDatasetId ? loadedDatasets[activeDatasetId] : null;
  const editEntry = entries.find(e => e.filename === editingFilename);
  const totalPages = Math.ceil(totalEntries / pageSize);
  const selectedEntries = entries.filter(e => selectedIndices.has(e.index));

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Layers className="w-6 h-6 text-forge-accent" /> Datasets
        </h1>
      </div>

      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-forge-muted" />
          <input value={loadPath} onChange={(e) => setLoadPath(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
            placeholder="Path to existing dataset directory..."
            className="w-full bg-forge-surface border border-forge-border rounded-lg pl-10 pr-3 py-2.5 text-sm focus:border-forge-accent focus:outline-none" />
        </div>
        <button onClick={handleLoad} disabled={loadingDataset || !loadPath.trim()}
          className="flex items-center gap-2 px-5 py-2.5 bg-forge-surface border border-forge-border rounded-lg text-sm hover:border-forge-accent transition-colors disabled:opacity-40">
          {loadingDataset ? <Spinner size="sm" /> : <FolderOpen className="w-4 h-4" />} Load
        </button>
        <button onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-5 py-2.5 bg-forge-accent text-black rounded-lg text-sm font-medium hover:bg-forge-accent/90">
          <Plus className="w-4 h-4" /> Create New
        </button>
      </div>
      {loadError && (
        <div className="text-xs text-red-400 flex items-center gap-1.5 -mt-3">
          <AlertTriangle className="w-3 h-3" /> {loadError}
        </div>
      )}

      {Object.keys(loadedDatasets).length > 0 && (
        <div className="flex items-center gap-1 border-b border-forge-border pb-0">
          {Object.entries(loadedDatasets).map(([dsId, ds]) => (
            <div key={dsId} className={`group flex items-center gap-1.5 px-3 py-2 text-sm cursor-pointer border-b-2 transition-colors ${
              activeDatasetId === dsId ? 'border-forge-accent text-forge-accent' : 'border-transparent text-forge-muted hover:text-forge-text'
            }`}>
              <button onClick={() => { setActiveDatasetId(dsId); setPage(0); setSelectedIndices(new Set()); }}>
                <span className="font-mono text-xs truncate max-w-[200px] inline-block">
                  {ds.directory?.split('/').pop() || dsId}
                </span>
                <span className="text-[10px] ml-1.5 text-forge-muted">{ds.total_images} imgs</span>
              </button>
              <button onClick={(e) => { e.stopPropagation(); handleUnload(dsId); }}
                className="opacity-0 group-hover:opacity-100 text-forge-muted hover:text-red-400 transition-all ml-1">
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {!activeDatasetId && (
        <EmptyState icon={FolderOpen} title="No dataset loaded"
          subtitle="Load an existing folder or create a new dataset to get started" />
      )}

      {activeDatasetId && (
        <div className="flex gap-4" style={{ minHeight: '70vh' }}>
          <div className="flex-1 min-w-0 space-y-3">
          {activeDs && (
            <div className="flex items-center gap-1.5 px-1 py-1 text-[11px] text-forge-muted/60 font-mono border-b border-forge-border/40 -mt-2 mb-1">
              <FolderOpen className="w-3 h-3 shrink-0" />
              <span className="truncate" title={activeDs.directory}>{activeDs.directory}</span>
              <button
                onClick={() => navigator.clipboard.writeText(activeDs.directory)}
                className="shrink-0 hover:text-forge-accent transition-colors ml-1"
                title="Copy path"
              >
                <FileText className="w-3 h-3" />
              </button>
            </div>
          )}
            <UploadZone datasetId={activeDatasetId} onUploaded={() => { fetchEntries(); refreshLoaded(); }} />
            <CaptionToolbar
              datasetId={activeDatasetId}
              selectedCount={selectedIndices.size}
              selectedIndices={selectedIndices}
              entries={entries}
              onRefresh={() => { fetchEntries(); refreshLoaded(); }}
            />
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <button onClick={() => { setFilterMode(null); setSearchText(''); setPage(0); }}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    !filterMode ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent' : 'border-forge-border text-forge-muted'
                  }`}>All ({activeDs?.total_images || totalEntries})</button>
                <button onClick={() => { setFilterMode('uncaptioned'); setPage(0); }}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    filterMode === 'uncaptioned' ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400' : 'border-forge-border text-forge-muted'
                  }`}>No Caption</button>
                <button onClick={() => { setFilterMode('captioned'); setPage(0); }}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    filterMode === 'captioned' ? 'bg-green-500/10 border-green-500/30 text-green-400' : 'border-forge-border text-forge-muted'
                  }`}>Captioned</button>
                <button
                  onClick={() => {
                    if (selectedIndices.size === entries.length && entries.length > 0) {
                      setSelectedIndices(new Set()); // deselect all if all selected
                    } else {
                      setSelectedIndices(new Set(entries.map(e => e.index)));
                    }
                  }}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                    selectedIndices.size === entries.length && entries.length > 0
                      ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent'
                      : 'border-forge-border text-forge-muted hover:text-forge-text'
                  }`}
                >
                  {selectedIndices.size === entries.length && entries.length > 0 ? 'Deselect All' : 'Select All'}
                </button>
              </div>
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-forge-muted" />
                  <input value={searchText} onChange={(e) => setSearchText(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Search captions..."
                    className="bg-forge-bg border border-forge-border rounded-lg pl-8 pr-2 py-1.5 text-xs w-48 focus:border-forge-accent focus:outline-none" />
                </div>
                <div className="flex items-center gap-1.5 border border-forge-border rounded px-2 py-1">
                  <Grid className="w-3 h-3 text-forge-muted shrink-0" />
                  <input
                    type="range"
                    min={2}
                    max={10}
                    value={gridSize}
                    onChange={e => setGridSize(Number(e.target.value))}
                    className="w-20 accent-forge-accent cursor-pointer"
                    title={`${gridSize} columns`}
                  />
                  <span className="text-[10px] text-forge-muted w-4 text-center">{gridSize}</span>
                </div>
                <button onClick={() => { setRightPanel('caption'); setShowRightPanel(!showRightPanel || rightPanel !== 'caption'); }}
                  className={`p-1.5 rounded border transition-colors ${
                    showRightPanel && rightPanel === 'caption' ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent' : 'border-forge-border text-forge-muted hover:text-forge-text'
                  }`} title="Caption Editor"><Edit3 className="w-4 h-4" /></button>
                <button onClick={() => { setRightPanel('analysis'); setShowRightPanel(!showRightPanel || rightPanel !== 'analysis'); }}
                  className={`p-1.5 rounded border transition-colors ${
                    showRightPanel && rightPanel === 'analysis' ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent' : 'border-forge-border text-forge-muted hover:text-forge-text'
                  }`} title="Concept Analysis"><BarChart3 className="w-4 h-4" /></button>
              </div>
            </div>

            <SelectionBar count={selectedIndices.size} onClearSelection={() => setSelectedIndices(new Set())}
              onDeleteSelected={handleDeleteSelected} onCaptionSelected={handleCaptionSelected} />
            <CropBar
              totalCount={totalEntries}
              selectedCount={selectedIndices.size}
              cropping={cropping}
              cropResult={cropResult}
              onCropAll={(params) => handleCropToBuckets(false, params)}
              onCropSelected={(params) => handleCropToBuckets(true, params)}
            />

            <div 
              className="grid gap-2"
              style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}
            >
              {entries.map((entry) => (
                <ImageCard
                  key={`${entry.filename}`}
                  entry={entry}
                  thumbnail={thumbnails[entry.filename]?.thumbnail}
                  isSelected={selectedIndices.has(entry.index)}
                  hasActiveSelection={selectedIndices.size > 0}   // ← add this
                  onSelect={handleSelect}
                  onDelete={handleDeleteSingle}
                  onPreview={handlePreview}
                  datasetId={activeDatasetId}
                  onCaptionUpdate={handleCaptionUpdate}
                />
              ))}
            </div>

            {entries.length === 0 && (
              <EmptyState icon={ImageIcon}
                title={filterMode ? 'No images match this filter' : 'No images yet'}
                subtitle={!filterMode ? 'Upload images above to get started' : undefined} />
            )}

            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-3 pt-2">
                <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                  className="p-1.5 rounded border border-forge-border text-forge-muted hover:text-forge-accent disabled:opacity-30 transition-colors">
                  <ChevronLeft className="w-4 h-4" /></button>
                <span className="text-xs text-forge-muted">Page {page + 1} of {totalPages} · {totalEntries} images</span>
                <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}
                  className="p-1.5 rounded border border-forge-border text-forge-muted hover:text-forge-accent disabled:opacity-30 transition-colors">
                  <ChevronRight className="w-4 h-4" /></button>
              </div>
            )}
          </div>

          {showRightPanel && (
            <div className="w-80 shrink-0 border border-forge-border rounded-lg bg-forge-surface/50 p-4 overflow-hidden flex flex-col">
              <div className="flex items-center gap-1 mb-3 border-b border-forge-border pb-2">
                <button onClick={() => setRightPanel('caption')}
                  className={`text-xs px-2.5 py-1 rounded transition-colors ${
                    rightPanel === 'caption' ? 'bg-forge-accent/10 text-forge-accent' : 'text-forge-muted hover:text-forge-text'
                  }`}><Edit3 className="w-3 h-3 inline mr-1" /> Caption</button>
                <button onClick={() => setRightPanel('analysis')}
                  className={`text-xs px-2.5 py-1 rounded transition-colors ${
                    rightPanel === 'analysis' ? 'bg-forge-accent/10 text-forge-accent' : 'text-forge-muted hover:text-forge-text'
                  }`}><BarChart3 className="w-3 h-3 inline mr-1" /> Analysis</button>
                <div className="flex-1" />
                <button onClick={() => setShowRightPanel(false)} className="text-forge-muted hover:text-forge-text">
                  <X className="w-3.5 h-3.5" /></button>
              </div>
              <div className="flex-1 min-h-0 overflow-y-auto">
                {rightPanel === 'caption' && (
                  <CaptionPanel entry={editEntry} 
                  thumbnail={editEntry ? thumbnails[editEntry.filename]?.thumbnail : null}
                    datasetId={activeDatasetId} onUpdate={handleCaptionUpdate} onPreview={handlePreview} />
                )}
                {rightPanel === 'analysis' && (
                  <ConceptAnalysisPanel datasetId={activeDatasetId}
                    totalImages={activeDs?.total_images || totalEntries} onSelectConcept={handleConceptSelect} />
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {showCreateModal && (
        <CreateDatasetModal onCreated={(result) => {
          setActiveDatasetId(result.dataset_id); setShowCreateModal(false);
          setPage(0); setSelectedIndices(new Set()); refreshLoaded();
        }} onClose={() => setShowCreateModal(false)} />
      )}

      {showBatchCaption && selectedEntries.length > 0 && (
        <BatchCaptionModal entries={selectedEntries} thumbnails={thumbnails} datasetId={activeDatasetId}
          onDone={() => { setShowBatchCaption(false); fetchEntries(); }}
          onClose={() => setShowBatchCaption(false)} />
      )}

      {previewImage && (
        <ImagePreviewModal imageSrc={previewImage.src} filename={previewImage.filename}
          onClose={() => setPreviewImage(null)} />
      )}
    </div>
  );
}
