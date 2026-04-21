import React from 'react';

export default function SampleViewer({ latestSamples }) {
  if (!latestSamples) return null;

  return (
    <div className="bg-forge-surface border border-forge-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium">Latest Samples</h2>
        <span className="text-xs text-forge-muted">
          Step {latestSamples.step} • {latestSamples.generation_time?.toFixed(1)}s
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {(latestSamples.image_paths || []).map((path, i) => (
          <div key={i} className="space-y-1">
            <img
              src={`/api/files/${encodeURIComponent(path)}`}
              alt={`Sample ${i}`}
              className="w-full rounded border border-forge-border"
            />
            <p className="text-xs text-forge-muted truncate" title={latestSamples.prompts?.[i]}>
              {latestSamples.prompts?.[i]}
            </p>
            <p className="text-xs text-forge-muted/50">
              Seed: {latestSamples.seeds?.[i]} • {latestSamples.sampler}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
