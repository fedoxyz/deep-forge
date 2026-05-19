import React from 'react';
import { Image as ImageIcon } from 'lucide-react';
import ImageCard from './ImageCard';

export default function ImageGrid({
  entries, thumbnails, selectedIndices, gridSize,
  datasetId, onSelect, onDelete, onPreview, onAnnotate, onCaptionUpdate, dsType,
}) {
  if (entries.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-forge-muted text-sm">
        <ImageIcon className="w-12 h-12 opacity-20 mb-4" />
        No images match this filter
      </div>
    );
  }

  return (
    <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}>
      {entries.map(entry => (
        <ImageCard
          key={entry.filename}
          entry={entry}
          thumbnail={thumbnails[entry.filename]?.thumbnail}
          isSelected={selectedIndices.has(entry.index)}
          hasActiveSelection={selectedIndices.size > 0}
          onSelect={onSelect}
          onDelete={onDelete}
          onPreview={onPreview}
          onAnnotate={onAnnotate}
          datasetId={datasetId}
          onCaptionUpdate={onCaptionUpdate}
          dsType={dsType}
        />
      ))}
    </div>
  );
}
