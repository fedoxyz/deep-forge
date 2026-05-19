import React, { useState, useEffect, useRef} from 'react';
import {
  Sparkles, 
  Zap, Badge
} from 'lucide-react';
import {
  updateCaption, 
  loadVisionModel, unloadVisionModel, getVisionModelStatus,
  captionSingle, startCaptionBatch, getCaptionBatchStatus,
  stopCaptionBatch, getVisionLoadStatus,
} from '../../utils/api';

export default function CaptionToolbar({ datasetId, selectedCount, selectedIndices, entries, onRefresh }) {
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
