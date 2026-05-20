import { useState, useEffect, useCallback } from 'react';
import {
  getDatasetEntries, getThumbnailsBatch,
  loadDataset, getLoadedDatasets, unloadDataset,
  deleteDatasetFile, deleteDatasetFileBatch,
} from '../utils/api';

const PAGE_SIZE = 48;

export function useDataset() {
  const [loadedDatasets, setLoadedDatasets] = useState({});
  const [activeDatasetId, setActiveDatasetId] = useState(null);
  const [entries, setEntries] = useState([]);
  const [totalEntries, setTotalEntries] = useState(0);
  const [page, setPage] = useState(0);
  const [filterMode, setFilterMode] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [thumbnails, setThumbnails] = useState({});
  const [selectedIndices, setSelectedIndices] = useState(new Set());
  const [loadError, setLoadError] = useState('');
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [maskBusts, setMaskBusts] = useState({});

  const bustMask = useCallback((filename) => {
    setMaskBusts(prev => ({ ...prev, [filename]: Date.now() }));
  }, []);

  const refreshLoaded = useCallback(async () => {
    try {
      const { datasets } = await getLoadedDatasets();
      setLoadedDatasets(datasets || {});
    } catch {}
  }, []);

  useEffect(() => { refreshLoaded(); }, []);

const fetchEntries = useCallback(async () => {
  if (!activeDatasetId) return;
  try {
    const filterVal = filterMode === 'search' ? searchText : filterMode;
    const result = await getDatasetEntries(activeDatasetId, {
      offset: page * PAGE_SIZE, limit: PAGE_SIZE, filter: filterVal,
    });
    setEntries(result.entries);
    setTotalEntries(result.total);
    const filenames = result.entries.map(e => e.filename);
    const { thumbnails: thumbs } = await getThumbnailsBatch(activeDatasetId, filenames, 256);
    setThumbnails(prev => ({ ...prev, ...thumbs }));
  } catch (e) { console.error('fetchEntries failed', e); }
}, [activeDatasetId, page, filterMode, searchText]);

  useEffect(() => { if (activeDatasetId) fetchEntries(); }, [activeDatasetId, page, filterMode]);

  const handleLoad = async (path) => {
    setLoadingDataset(true); setLoadError('');
    try {
      const result = await loadDataset(path);
      setActiveDatasetId(result.dataset_id);
      setPage(0); setSelectedIndices(new Set());
      refreshLoaded();
    } catch (e) { setLoadError(e.message); }
    setLoadingDataset(false);
  };

const handleCreated = useCallback(async (result) => {
  await refreshLoaded();                    // wait — now loadedDatasets has dataset_type
  setActiveDatasetId(result.dataset_id);
  setPage(0);
  setSelectedIndices(new Set());
}, [refreshLoaded]);

  const handleUnload = async (dsId) => {
    await unloadDataset(dsId);
    if (activeDatasetId === dsId) {
      setActiveDatasetId(null); setEntries([]); setSelectedIndices(new Set());
    }
    refreshLoaded();
  };

  const handleSelect = (entry) => {
    setSelectedIndices(prev => {
      const next = new Set(prev);
      if (next.has(entry.index)) next.delete(entry.index);
      else next.add(entry.index);
      return next;
    });
  };

  const handleDeleteSingle = async (entry) => {
    await deleteDatasetFile(activeDatasetId, entry.filename);
    setEntries(prev => prev.filter(e => e.index !== entry.index));
    setTotalEntries(prev => prev - 1);
    setSelectedIndices(prev => { const n = new Set(prev); n.delete(entry.index); return n; });
    setThumbnails(prev => { const n = {...prev}; delete n[entry.filename]; return n; });
    refreshLoaded();
  };

  const handleDeleteSelected = async () => {
    const filenames = entries.filter(e => selectedIndices.has(e.index)).map(e => e.filename);
    await deleteDatasetFileBatch(activeDatasetId, filenames);
    setEntries(prev => prev.filter(e => !selectedIndices.has(e.index)));
    setTotalEntries(prev => prev - filenames.length);
    setSelectedIndices(new Set());
    setThumbnails(prev => {
      const n = {...prev}; filenames.forEach(f => delete n[f]); return n;
    });
    refreshLoaded();
  };

  const handleCaptionUpdate = (filename, newCaption) => {
    setEntries(prev => prev.map(e =>
      e.filename === filename ? { ...e, caption: newCaption, has_caption_file: true } : e
    ));
  };

  return {
    loadedDatasets, activeDatasetId, setActiveDatasetId,
    entries, totalEntries, page, setPage,
    filterMode, setFilterMode, searchText, setSearchText,
    thumbnails, selectedIndices, setSelectedIndices,
    loadError, loadingDataset,
    refreshLoaded, fetchEntries,
    handleLoad, handleUnload, handleSelect,
    handleDeleteSingle, handleDeleteSelected, handleCaptionUpdate,
    pageSize: PAGE_SIZE,
    activeDs: loadedDatasets[activeDatasetId] ?? null,
    selectedEntries: entries.filter(e => selectedIndices.has(e.index)),
    totalPages: Math.ceil(totalEntries / PAGE_SIZE),
    handleCreated,
    maskBusts,
    bustMask,
  };
}
