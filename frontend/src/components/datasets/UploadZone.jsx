// Replace the entire component with this:
import React, { useState, useRef } from 'react';
import { Upload, Check, AlertTriangle, FolderOpen } from 'lucide-react';
import { uploadDatasetFiles } from '../../utils/api';

export default function UploadZone({ datasetId, onUploaded, dsType }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const handleFiles = async (fileList) => {
    if (!fileList?.length || !datasetId) return;
    setUploading(true); setProgress(0); setResult(null);
    try {
      const res = await uploadDatasetFiles(datasetId, Array.from(fileList), setProgress);
      setResult(res);
      onUploaded?.();
    } catch (e) { setResult({ errors: [e.message], uploaded: [] }); }
    setUploading(false);
    setTimeout(() => setResult(null), 5000);
  };
  const isClassification = dsType === 'classification';

  const acceptAttr = isClassification
    ? 'image/*,.png,.jpg,.jpeg,.webp,.bmp'
    : 'image/*,.txt,.png,.jpg,.jpeg,.webp,.bmp';

  const hintText = isClassification
    ? 'Drop a class folder or images — folder name becomes the class label'
    : '.png .jpg .jpeg .webp .bmp — pair with .txt files for captions';

  return (
    <div className="space-y-2">
      <div
        onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={e => { e.preventDefault(); setDragging(false); }}
        className={`border-2 border-dashed rounded-lg px-4 py-4 transition-all ${
          dragging ? 'border-forge-accent bg-forge-accent/5 scale-[1.01]'
          : uploading ? 'border-forge-border opacity-70'
          : 'border-forge-border hover:border-forge-accent/40'
        }`}>

        {/* Hidden inputs */}
        <input ref={fileInputRef} type="file" multiple accept={acceptAttr}
          className="hidden"
          onChange={e => { handleFiles(e.target.files); e.target.value = ''; }} />
        {/* webkitdirectory allows folder selection */}
        <input ref={folderInputRef} type="file"
          // @ts-ignore
          webkitdirectory="true" directory="true" multiple
          className="hidden"
          onChange={e => { handleFiles(e.target.files); e.target.value = ''; }} />

        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 flex-1">
            <Upload className="w-5 h-5 text-forge-muted/40 shrink-0" />
            <div className="text-left">
              <p className="text-sm text-forge-muted">
                {uploading
                  ? `Uploading… ${progress}%`
                  : 'Drop files or folders here'}
              </p>
              <p className="text-[10px] text-forge-muted/40">{hintText}</p>
            </div>
          </div>

          {/* Browse buttons */}
          {!uploading && (
            <div className="flex gap-2 shrink-0">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs border border-forge-border rounded hover:border-forge-accent/50 text-forge-muted hover:text-forge-text transition-colors">
                <Upload className="w-3 h-3" /> Files
              </button>
              <button
                onClick={() => folderInputRef.current?.click()}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs border border-forge-border rounded hover:border-forge-accent/50 text-forge-muted hover:text-forge-text transition-colors">
                <FolderOpen className="w-3 h-3" /> Folder
              </button>
            </div>
          )}
        </div>
      </div>

      {uploading && (
        <div className="h-1 bg-forge-surface rounded-full overflow-hidden">
          <div className="h-full bg-forge-accent transition-all rounded-full" style={{ width: `${progress}%` }} />
        </div>
      )}

      {result && (
        <div className="text-xs flex items-center gap-3">
          {result.uploaded?.length > 0 && (
            <span className="text-green-400 flex items-center gap-1">
              <Check className="w-3 h-3" /> {result.uploaded.length} file(s) uploaded
            </span>
          )}
          {result.errors?.length > 0 && (
            <span className="text-red-400 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> {result.errors.length} error(s)
            </span>
          )}
        </div>
      )}
    </div>
  );
}
