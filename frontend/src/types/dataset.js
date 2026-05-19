export const DATASET_TYPES = {
  CAPTION:      'caption',       // must match backend exactly
  SEGMENTATION: 'segmentation',
  DETECTION:    'detection',
  CLASSIFICATION: 'classification',
  UNKNOWN:      'unknown',
};

export function inferDatasetType(dsInfo) {
  if (!dsInfo) return DATASET_TYPES.UNKNOWN;
  if (dsInfo.dataset_type) return dsInfo.dataset_type;  // backend sets this
  // fallback: if entries have polygon labels → segmentation
  if (dsInfo.has_segmentation_labels) return DATASET_TYPES.SEGMENTATION;
  if (dsInfo.has_detection_labels) return DATASET_TYPES.DETECTION;
  return DATASET_TYPES.CAPTION;
}
export function datasetTypeLabel(type) {
  return {
    'caption':      'Caption',
    'segmentation': 'Segmentation',
    'detection':    'Detection',
    'classification': 'Classification',
    'unknown':      'Unknown',
  }[type] ?? 'Unknown';
}
