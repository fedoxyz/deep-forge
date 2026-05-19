import React, { useEffect, useState } from 'react';
import { getClassDistribution } from '../../utils/api';

export default function ClassificationPanel({ datasetId, totalImages }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!datasetId) return;
    setData(null); setError(null);
    getClassDistribution(datasetId)
      .then(setData)
      .catch(e => setError(e.message));
  }, [datasetId]);

  if (error) return <p className="text-xs text-red-400">{error}</p>;
  if (!data) return <p className="text-xs text-forge-muted animate-pulse">Loading classes…</p>;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs text-forge-muted">
        <span>{data.num_classes} classes</span>
        <span>{data.total_images} images</span>
      </div>
      <div className="space-y-1.5">
        {data.classes.map(({ name, count }) => {
          const pct = Math.round((count / data.total_images) * 100);
          return (
            <div key={name}>
              <div className="flex items-center justify-between text-xs mb-0.5">
                <span className="font-mono text-forge-text truncate max-w-[160px]" title={name}>{name}</span>
                <span className="text-forge-muted">{count} <span className="opacity-50">({pct}%)</span></span>
              </div>
              <div className="h-1.5 bg-forge-border rounded-full overflow-hidden">
                <div className="h-full bg-forge-accent rounded-full" style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-forge-muted border-t border-forge-border pt-2 leading-relaxed">
        Class index order (for model output mapping): {data.classes.map(c => c.name).join(', ')}
      </p>
    </div>
  );
}
