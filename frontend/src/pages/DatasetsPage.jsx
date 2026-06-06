// frontend/src/pages/DatasetsPage.jsx
import React, { useState, useCallback } from 'react';
import { Layers, FolderOpen, Plus, X, AlertTriangle } from 'lucide-react';
import { useDataset } from '../hooks/useDataset';
import { cropDatasetToBuckets } from '../utils/api';
import { inferDatasetType, datasetTypeLabel, DATASET_TYPES } from '../types/dataset';
import ClassificationPanel from '../components/datasets/ClassificationPanel';
import DatasetToolbar from '../components/datasets/DatasetToolbar';
import ImageGrid from '../components/datasets/ImageGrid';
import CaptionPanel from '../components/datasets/CaptionPanel';
import ConceptAnalysisPanel from '../components/datasets/ConceptAnalysisPanel';
import UploadZone from '../components/datasets/UploadZone';
import CaptionToolbar from '../components/datasets/CaptionToolbar';
import CropBar from '../components/datasets/CropBar';
import SelectionBar from '../components/datasets/SelectionBar';
import AnnotationTool from '../components/datasets/AnnotationTool';
import CreateDatasetModal from '../components/datasets/CreateDatasetModal';
import BatchCaptionModal from '../components/datasets/BatchCaptionModal';
import ImagePreviewModal from '../components/datasets/ImagePreviewModal';
import { Edit3, BarChart3 } from 'lucide-react';

export default function DatasetsPage() {
  const ds = useDataset();
  const [loadPath, setLoadPath] = useState('');
  const [rightPanel, setRightPanel] = useState('caption');
  const [showRightPanel, setShowRightPanel] = useState(false);
  const [gridSize, setGridSize] = useState(4);
  const [editingFilename, setEditingFilename] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showBatchCaption, setShowBatchCaption] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [annotatingEntry, setAnnotatingEntry] = useState(null);
  const [cropping, setCropping] = useState(false);
  const [cropResult, setCropResult] = useState(null);

  const editEntry = ds.entries.find(e => e.filename === editingFilename);
  const dsType = inferDatasetType(ds.activeDs);

  const handleSelect = useCallback((entry) => {
    ds.handleSelect(entry);
    setEditingFilename(entry.filename);
    if (!showRightPanel) {
      if (dsType === 'caption') { setRightPanel('caption'); setShowRightPanel(true); }
      else if (dsType === 'classification') { setRightPanel('analysis'); setShowRightPanel(true); }
    }
  }, [ds.handleSelect, showRightPanel, dsType]);
  
  const handlePreview = useCallback((entry) => {
    setPreviewImage({
      filename: entry.filename,
      src: `/api/datasets/${ds.activeDatasetId}/image/by-filename/${encodeURIComponent(entry.filename)}`,
    });
  }, [ds.activeDatasetId]);

  const handleDelete = useCallback(async (e) => {
    if (confirm(`Delete ${e.filename}?`)) await ds.handleDeleteSingle(e);
  }, [ds.handleDeleteSingle]);
  
  const handleAnnotate = useCallback((entry) => setAnnotatingEntry(entry), []);

  const handleCropToBuckets = async (selectedOnly = false, cropParams = { preset: 'sdxl' }) => {
    if (!confirm(
      selectedOnly
        ? `Crop ${ds.selectedIndices.size} selected image(s) to nearest training bucket? Originals are backed up to .originals/`
        : `Crop ALL ${ds.totalEntries} images to nearest training bucket? Originals are backed up to .originals/`
    )) return;
    setCropping(true);
    setCropResult(null);
    try {
      const filenames = selectedOnly
        ? entries.filter(e => selectedIndices.has(e.index)).map(e => e.filename)
        : null;
      const result = await cropDatasetToBuckets(ds.activeDatasetId, { filenames, ...cropParams });
      setCropResult(result);
      fetchEntries();
    } catch (e) {
      console.error('Crop failed:', e);
    }
    setCropping(false);
    setTimeout(() => setCropResult(null), 6000);
  };


  return (
    <div className="space-y-5 overflow-x-hidden">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Layers className="w-6 h-6 text-forge-accent" /> Datasets
        </h1>
      </div>

      {/* Load bar */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-forge-muted" />
          <input value={loadPath} onChange={e => setLoadPath(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && ds.handleLoad(loadPath)}
            placeholder="Path to dataset directory..."
            className="w-full bg-forge-surface border border-forge-border rounded-lg pl-10 pr-3 py-2.5 text-sm focus:border-forge-accent focus:outline-none" />
        </div>
        <button onClick={() => ds.handleLoad(loadPath)} disabled={ds.loadingDataset || !loadPath.trim()}
          className="px-5 py-2.5 bg-forge-surface border border-forge-border rounded-lg text-sm hover:border-forge-accent disabled:opacity-40">
          Load
        </button>
        <button onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-5 py-2.5 bg-forge-accent text-black rounded-lg text-sm font-medium">
          <Plus className="w-4 h-4" /> Create New
        </button>
      </div>

      {ds.loadError && (
        <div className="text-xs text-red-400 flex items-center gap-1.5">
          <AlertTriangle className="w-3 h-3" /> {ds.loadError}
        </div>
      )}

      {/* Dataset tabs */}
      {Object.keys(ds.loadedDatasets).length > 0 && (
        <div className="min-w-0 flex-1">
          <div className="tabs-scroll flex items-center gap-1 border-b border-forge-border overflow-x-auto">
          {Object.entries(ds.loadedDatasets).map(([dsId, info]) => (
            <div key={dsId} className={`group shrink-0 flex items-center gap-1.5 px-3 py-2 text-sm cursor-pointer border-b-2 transition-colors ${
              ds.activeDatasetId === dsId ? 'border-forge-accent text-forge-accent' : 'border-transparent text-forge-muted hover:text-forge-text'
            }`}>
              <button onClick={() => { ds.setActiveDatasetId(dsId); ds.setPage(0); ds.setSelectedIndices(new Set()); }}>
                <span className="font-mono text-xs truncate max-w-[180px] inline-block">
                  {info.directory?.split('/').pop() || dsId}
                </span>
                <span className="text-[10px] ml-1.5 text-forge-muted">{info.total_images} imgs</span>
                <span className="text-[10px] ml-1 opacity-50">· {datasetTypeLabel(inferDatasetType(info))}</span>
              </button>
              <button onClick={e => { e.stopPropagation(); ds.handleUnload(dsId); }}
                className="opacity-0 group-hover:opacity-100 text-forge-muted hover:text-red-400 ml-1">
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
        </div>
      )}

      {!ds.activeDatasetId ? (
        <div className="flex flex-col items-center justify-center py-16 text-forge-muted text-sm">
          <FolderOpen className="w-12 h-12 opacity-20 mb-4" />
          No dataset loaded
        </div>
      ) : (
        <div className="flex gap-4" style={{ minHeight: '70vh' }}>
          {/* Main column */}
          <div className="flex-1 min-w-0 space-y-3">
            <UploadZone datasetId={ds.activeDatasetId}
              dsType={dsType}
              onUploaded={() => { ds.fetchEntries(); ds.refreshLoaded(); }} />
            {dsType === DATASET_TYPES.CAPTION && (
              <CaptionToolbar datasetId={ds.activeDatasetId}
                selectedCount={ds.selectedIndices.size}
                selectedIndices={ds.selectedIndices}
                entries={ds.entries}
                onRefresh={() => { ds.fetchEntries(); ds.refreshLoaded(); }} />
            )}
            <DatasetToolbar
              filterMode={ds.filterMode} setFilterMode={ds.setFilterMode}
              searchText={ds.searchText} setSearchText={ds.setSearchText}
              onSearch={() => { ds.setFilterMode('search'); ds.setPage(0); }}
              totalEntries={ds.totalEntries} totalImages={ds.activeDs?.total_images}
              selectedCount={ds.selectedIndices.size}
              allSelected={ds.selectedIndices.size === ds.entries.length && ds.entries.length > 0}
              onSelectAll={() => {
                if (ds.selectedIndices.size === ds.entries.length && ds.entries.length > 0)
                  ds.setSelectedIndices(new Set());
                else
                  ds.setSelectedIndices(new Set(ds.entries.map(e => e.index)));
              }}
              gridSize={gridSize} setGridSize={setGridSize}
              rightPanel={rightPanel} showRightPanel={showRightPanel}
              dsType={dsType}
              onToggleCaption={() => {
                setRightPanel('caption');
                setShowRightPanel(p => !p || rightPanel !== 'caption');
              }}
              onToggleAnalysis={() => {
                setRightPanel('analysis');
                setShowRightPanel(p => !p || rightPanel !== 'analysis');
              }}
            />
            <SelectionBar count={ds.selectedIndices.size}
              dsType={dsType}
              onClearSelection={() => ds.setSelectedIndices(new Set())}
              onDeleteSelected={async () => {
                if (confirm(`Delete ${ds.selectedIndices.size} image(s)?`))
                  await ds.handleDeleteSelected();
              }}
              onCaptionSelected={() => ds.selectedEntries.length > 0 && setShowBatchCaption(true)} />
            <CropBar totalCount={ds.totalEntries} selectedCount={ds.selectedIndices.size}
              cropping={cropping} cropResult={cropResult}
              onCropAll={(params) => handleCropToBuckets(false, params)}
              onCropSelected={(params) => handleCropToBuckets(true, params)}
              />
            <ImageGrid
              entries={ds.entries} thumbnails={ds.thumbnails}
              selectedIndices={ds.selectedIndices} gridSize={gridSize}
              datasetId={ds.activeDatasetId}
              onSelect={handleSelect}
              onDelete={handleDelete}
              onPreview={handlePreview}
              onAnnotate={handleAnnotate}
              onCaptionUpdate={ds.handleCaptionUpdate} 
              dsType={dsType}
              maskBusts={ds.maskBusts}
            />
            {/* Pagination */}
            {ds.totalPages > 1 && (
              <div className="flex items-center justify-center gap-3 pt-2">
                <button onClick={() => ds.setPage(p => Math.max(0, p - 1))} disabled={ds.page === 0}
                  className="p-1.5 rounded border border-forge-border text-forge-muted hover:text-forge-accent disabled:opacity-30">←</button>
                <span className="text-xs text-forge-muted">Page {ds.page + 1} of {ds.totalPages} · {ds.totalEntries} images</span>
                <button onClick={() => ds.setPage(p => Math.min(ds.totalPages - 1, p + 1))} disabled={ds.page >= ds.totalPages - 1}
                  className="p-1.5 rounded border border-forge-border text-forge-muted hover:text-forge-accent disabled:opacity-30">→</button>
              </div>
            )}
          </div>

          {/* Right panel */}
          {showRightPanel && (
            <div className="w-80 shrink-0 border border-forge-border rounded-lg bg-forge-surface/50 p-4 flex flex-col overflow-hidden">
              <div className="flex items-center gap-1 mb-3 border-b border-forge-border pb-2">
                {dsType === DATASET_TYPES.CAPTION && <>
                  <button onClick={() => setRightPanel('caption')}
                    className={`text-xs px-2.5 py-1 rounded ${rightPanel === 'caption' ? 'bg-forge-accent/10 text-forge-accent' : 'text-forge-muted'}`}>
                    <Edit3 className="w-3 h-3 inline mr-1" />Caption
                  </button>
                  <button onClick={() => setRightPanel('analysis')}
                    className={`text-xs px-2.5 py-1 rounded ${rightPanel === 'analysis' ? 'bg-forge-accent/10 text-forge-accent' : 'text-forge-muted'}`}>
                    <BarChart3 className="w-3 h-3 inline mr-1" />Analysis
                  </button>
                </>}
                {dsType === DATASET_TYPES.CLASSIFICATION && (
                  <span className="text-xs px-2.5 py-1 text-forge-accent">
                    <BarChart3 className="w-3 h-3 inline mr-1" />Classes
                  </span>
                )}
                <div className="flex-1" />
                <button onClick={() => setShowRightPanel(false)} className="text-forge-muted hover:text-forge-text">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
                <div className="flex-1 min-h-0 overflow-y-auto">
                  {dsType === DATASET_TYPES.CAPTION && rightPanel === 'caption' && (
                    <CaptionPanel entry={editEntry}
                      thumbnail={editEntry ? ds.thumbnails[editEntry.filename]?.thumbnail : null}
                      datasetId={ds.activeDatasetId}
                      onUpdate={ds.handleCaptionUpdate}
                      onPreview={handlePreview} />
                  )}
                  {dsType === DATASET_TYPES.CAPTION && rightPanel === 'analysis' && (
                    <ConceptAnalysisPanel datasetId={ds.activeDatasetId}
                      totalImages={ds.activeDs?.total_images || ds.totalEntries}
                      onSelectConcept={(c) => {
                        ds.setSearchText(c.phrase);
                        ds.setFilterMode('search');
                        ds.setPage(0);
                      }} />
                  )}
                  {dsType === DATASET_TYPES.CLASSIFICATION && (
                    <ClassificationPanel datasetId={ds.activeDatasetId}
                      totalImages={ds.activeDs?.total_images || ds.totalEntries} />
                  )}
                  {(dsType === DATASET_TYPES.DETECTION || dsType === DATASET_TYPES.SEGMENTATION || dsType === DATASET_TYPES.UNKNOWN) && (
                    <div className="flex flex-col items-center justify-center h-full text-forge-muted text-xs text-center gap-2 py-8">
                      <span className="text-2xl opacity-30">🚧</span>
                      <span>{datasetTypeLabel(dsType)} panel coming soon</span>
                    </div>
                  )}
                </div>
            </div>
          )}
        </div>
      )}

      {/* Modals */}
{showCreateModal && (
  <CreateDatasetModal
    onCreated={(result) => {
      ds.handleCreated(result);    // ← replaces ds.setActiveDatasetId + ds.refreshLoaded()
      setShowCreateModal(false);
    }}
    onClose={() => setShowCreateModal(false)} />
)}
    {showBatchCaption && dsType === DATASET_TYPES.CAPTION && ds.selectedEntries.length > 0 && (
        <BatchCaptionModal entries={ds.selectedEntries} thumbnails={ds.thumbnails}
          datasetId={ds.activeDatasetId}
          onDone={() => { setShowBatchCaption(false); ds.fetchEntries(); }}
          onClose={() => setShowBatchCaption(false)} />
      )}
      {previewImage && (
        <ImagePreviewModal imageSrc={previewImage.src} filename={previewImage.filename}
          onClose={() => setPreviewImage(null)} />
      )}
      {annotatingEntry && (
        <AnnotationTool
          entry={annotatingEntry}
          datasetId={ds.activeDatasetId}
          dsType={dsType}
          onClose={() => setAnnotatingEntry(null)}
          onSaved={(filename) => {
            ds.bustMask(filename);
            setAnnotatingEntry(null);
            ds.fetchEntries();
          }}
        />
      )}
    </div>
  );
}
