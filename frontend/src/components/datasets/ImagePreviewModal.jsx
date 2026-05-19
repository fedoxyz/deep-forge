import React from 'react';
import { X } from 'lucide-react';

export default function ImagePreviewModal({ imageSrc, filename, onClose }) {
  if (!imageSrc) return null;
  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8" onClick={onClose}>
      <div className="relative max-w-[90vw] max-h-[90vh]" onClick={e => e.stopPropagation()}>
        <button onClick={onClose} className="absolute -top-3 -right-3 z-10 bg-forge-surface border border-forge-border rounded-full p-1.5 hover:bg-forge-accent/20 transition-colors">
          <X className="w-4 h-4" />
        </button>
        <img src={imageSrc} alt={filename} className="max-w-full max-h-[85vh] object-contain rounded-lg" />
        <p className="text-center text-xs text-forge-muted mt-2 font-mono">{filename}</p>
      </div>
    </div>
  );
}
