const API = '/api';

async function f(path, opts = {}) {
  const pw = sessionStorage.getItem('forge_pw') || '';
  const res = await fetch(`${API}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(pw && { 'X-Forge-Password': pw }),
      ...opts.headers,
    },
    ...opts,
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(e.detail || e.errors?.join(', ') || res.statusText);
  }
  return res.json();
}

// Configs
export const getConfigs = () => f('/configs');
export const getConfig = n => f(`/configs/${n}`);
export const getDefaultConfig = () => f('/configs/default');
export const saveConfig = (n, c) => f('/configs', { method: 'POST', body: JSON.stringify({ name: n, config: c }) });
export const updateConfig = (n, c) => f(`/configs/${n}`, { method: 'PUT', body: JSON.stringify({ config: c }) });
export const deleteConfig = n => f(`/configs/${n}`, { method: 'DELETE' });

// Hyperparam Configs (for Optuna)
export const getHparamConfigs = () => f('/hparam_configs');
export const getHparamConfig = n => f(`/hparam_configs/${n}`);
export const saveHparamConfig = (n, c) => f('/hparam_configs', { method: 'POST', body: JSON.stringify({ name: n, config: c }) });
export const deleteHparamConfig = n => f(`/hparam_configs/${n}`, { method: 'DELETE' });

// Saved model specs (architectures)
export const getModelSpecs = () => f('/model_specs');
export const getModelSpec = n => f(`/model_specs/${n}`);
export const saveModelSpec = (n, spec) => f('/model_specs', { method: 'POST', body: JSON.stringify({ name: n, spec }) });
export const deleteModelSpec = n => f(`/model_specs/${n}`, { method: 'DELETE' });

// Training
export const getTrainingStatus = () => f('/training/status');
export const startTraining = (cn, cfg, mode = 'lora') => f('/training/start', { method: 'POST', body: JSON.stringify({ config_name: cn, config: cfg, mode }) });
export const stopTraining = () => f('/training/stop', { method: 'POST' });
export const getTrainingLogs = () => f('/training/logs');
// Training
export const requestSample = (params = {}) =>
  f('/training/sample', { method: 'POST', body: JSON.stringify(params) });

export const requestSaveNow = () =>
  f('/training/save_now', { method: 'POST' });

export const resumeTraining = (configName, mode, checkpointPath) =>
  f('/training/resume', {
    method: 'POST',
    body: JSON.stringify({ config_name: configName, mode, checkpoint_path: checkpointPath }),
  });

export const getLatestSamples = () => f('/training/samples/latest');

export const getTrainingLogHistory = () => f('/training/logs/history');

export const getPipelines = () => f('/training/pipelines');

export const getPipelinePresets = (name) => f(`/training/pipelines/${name}/presets`);

// Model builder
export const getLayerCatalog = () => f('/builder/catalog');
export const getModelPresets = () => f('/builder/presets');
export const validateModel = (spec, is) => f('/builder/validate', { method: 'POST', body: JSON.stringify({ spec, input_shape: is }) });
export const buildModel = spec => f('/builder/build', { method: 'POST', body: JSON.stringify({ spec }) });
export const modelToCode = spec => f('/builder/to_code', { method: 'POST', body: JSON.stringify({ spec }) });

export const getDatasetCatalog = () => f('/datasets/catalog');
export const getBuiltinDatasets = () => f('/datasets/builtin');

// Dataset loading
export const loadDataset = (directory) =>
  f('/datasets/load', { method: 'POST', body: JSON.stringify({ directory }) });

export const getLoadedDatasets = () => f('/datasets/loaded');

export const unloadDataset = (datasetId) =>
  f(`/datasets/loaded/${datasetId}`, { method: 'DELETE' });

export const getDatasetInfo = (datasetId) => f(`/datasets/${datasetId}`);

export const getDatasetEntries = (datasetId, { offset = 0, limit = 50, filter = null } = {}) => {
  const params = new URLSearchParams({ offset, limit });
  if (filter) params.set('filter', filter);
  return f(`/datasets/${datasetId}/entries?${params}`);
};

export const getClassDistribution = (datasetId) =>
  f(`/datasets/${datasetId}/classes`);

export const cropDatasetToBuckets = (datasetId, { filenames = null, preset = 'sdxl', minSize, maxSize, step, maxAspect } = {}) =>
  f(`/datasets/${datasetId}/crop-to-bucket`, {
    method: 'POST',
    body: JSON.stringify({
      filenames,
      preset,
      min_size: minSize,
      max_size: maxSize,
      step,
      max_aspect: maxAspect,
    }),
  });

export const getBucketPresets = () => f('/datasets/bucket-presets');

export const getImageUrl = (datasetId, filename) =>
  `/api/datasets/${datasetId}/image/by-filename/${encodeURIComponent(filename)}`;

// Thumbnails
export const getThumbnail = (datasetId, imageIndex, size = 256) =>
  f(`/datasets/${datasetId}/thumbnail/${imageIndex}?size=${size}`);

export const getThumbnailsBatch = (datasetId, filenames, size = 192) =>
  f(`/datasets/${datasetId}/thumbnails/by-filenames?filenames=${filenames.join(',')}&size=${size}`);

// Caption editing
export const updateCaption = (datasetId, filename, caption) =>
  f(`/datasets/${datasetId}/caption/by-filename/${encodeURIComponent(filename)}`, {
    method: 'PUT',
    body: JSON.stringify({ caption }),
  });

export const updateCaptionsBatch = (datasetId, updates) =>
  f(`/datasets/${datasetId}/captions`, {
    method: 'PUT',
    body: JSON.stringify({ updates }),
  });

// Concept analysis
export const analyzeDatasetConcepts = (datasetId, params = {}) =>
  f(`/datasets/${datasetId}/analyze`, {
    method: 'POST',
    body: JSON.stringify(params),
  });

export const getConceptImages = (datasetId, phrase) =>
  f(`/datasets/${datasetId}/concept-images?phrase=${encodeURIComponent(phrase)}`);

export const findSimilarPhrases = (datasetId, params = {}) =>
  f(`/datasets/${datasetId}/find-similar`, {
    method: 'POST',
    body: JSON.stringify(params),
  });

export const createDataset = (name, baseDir = '/workspace/dataset', datasetType = 'caption') => {
  const pw = sessionStorage.getItem('forge_pw') || '';
  const form = new FormData();
  form.append('name', name);
  form.append('base_dir', baseDir);
  form.append('dataset_type', datasetType);   // ← this was missing
  return fetch(`${API}/datasets/create`, {
    method: 'POST',
    body: form,
    headers: { ...(pw && { 'X-Forge-Password': pw }) },
  }).then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail); }); return r.json(); });
};

export const uploadDatasetFiles = (datasetId, files, onProgress) => {
  const pw = sessionStorage.getItem('forge_pw') || '';
  const form = new FormData();
  for (const file of files) form.append('files', file);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/datasets/${datasetId}/upload`);
    if (pw) xhr.setRequestHeader('X-Forge-Password', pw);
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round(e.loaded / e.total * 100));
      };
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else { try { reject(new Error(JSON.parse(xhr.responseText)?.detail || 'Upload failed')); } catch { reject(new Error('Upload failed')); } }
    };
    xhr.onerror = () => reject(new Error('Upload failed'));
    xhr.send(form);
  });
};

export const deleteDatasetFile = (datasetId, filename) =>
  f(`/datasets/${datasetId}/file/${encodeURIComponent(filename)}`, { method: 'DELETE' });

export const deleteDatasetFileBatch = (datasetId, filenames) =>
  f(`/datasets/${datasetId}/delete-batch`, {
    method: 'POST',
    body: JSON.stringify({ filenames }),
  });

// ── Vision model & captioning ──

export const loadVisionModel = (backend = 'qwen3vl', modelId = 'huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated', opts = {}) =>
  f('/vision/load', { method: 'POST', body: JSON.stringify({ backend, model_id: modelId, ...opts }) });

export const getVisionLoadStatus = () => f('/vision/load/status');

export const unloadVisionModel = () => f('/vision/unload', { method: 'POST' });
export const getVisionModelStatus = () => f('/vision/status');

export const captionSingle = (imagePath, prompt = '', maxNewTokens = 256, temperature = 0.7) =>
  f('/vision/caption', { method: 'POST', body: JSON.stringify({ image_path: imagePath, prompt, max_new_tokens: maxNewTokens, temperature }) });

export const startCaptionBatch = (datasetId, { indices = null, prompt = '', maxNewTokens = 256, temperature = 0.7, overwrite = false } = {}) =>
  f('/vision/caption-batch', { method: 'POST', body: JSON.stringify({ dataset_id: datasetId, indices, prompt, max_new_tokens: maxNewTokens, temperature, overwrite }) });

export const getCaptionBatchStatus = () => f('/vision/caption-batch/status');
export const getCaptionBatchResults = () => f('/vision/caption-batch/results');
export const stopCaptionBatch = () => f('/vision/caption-batch/stop', { method: 'POST' });

export const extractConceptsLLM = (datasetId, categories, promptTemplate = null) =>
  f('/vision/extract-concepts', { method: 'POST', body: JSON.stringify({ dataset_id: datasetId, categories, prompt_template: promptTemplate }) });

// Model inspection
export const inspectModel = mp => f('/model/inspect', { method: 'POST', body: JSON.stringify({ model_path: mp }) });
export const getPresets = () => f('/model/presets');

// Optuna
export const startOptuna = (cn, cfg, nt = 20, mode = 'train_custom', ss = null) =>
  f('/optuna/start', { method: 'POST', body: JSON.stringify({ config_name: cn, config: cfg, n_trials: nt, mode, search_space: ss }) });
export const getDefaultSearchSpace = () => f('/optuna/default_space');
export const getOptunaStatus = () => f('/optuna/status');
export const getOptunaResults = () => f('/optuna/results');

// System
export const getModules = () => f('/modules');
export const getBuckets = () => f('/buckets');
export const getHealth = () => f('/health');

export function subscribeTraining(onMsg) {
  const pw = sessionStorage.getItem('forge_pw') || '';
  const es = new EventSource(`${API}/training/stream?password=${encodeURIComponent(pw)}`);
  es.onmessage = e => { try { onMsg(JSON.parse(e.data)); } catch {} };
  es.onerror = () => es.close();
  return () => es.close();
}
