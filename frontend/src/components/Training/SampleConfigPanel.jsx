import React from 'react';
import { requestSample } from '../../utils/api';

export default function SampleConfigPanel({
  samplePrompt, setSamplePrompt,
  samplerType, setSamplerType,
  sampleSteps, setSampleSteps,
  sampleSeed, setSampleSeed,
  setSampleMsg,
}) {
  const handleGenerate = async () => {
    try {
      await requestSample({
        prompts: samplePrompt ? [samplePrompt] : undefined,
        sampler: samplerType,
        num_steps: sampleSteps,
        seed: sampleSeed,
      });
      setSampleMsg('Generating samples...');
      setTimeout(() => setSampleMsg(''), 5000);
    } catch (e) { setSampleMsg('Error: ' + e.message); }
  };

  return (
    <div className="bg-forge-surface border border-forge-border rounded-lg p-4 space-y-3">
      <h3 className="text-sm font-medium">Sample Configuration</h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="col-span-2">
          <label className="text-xs text-forge-muted">Prompt</label>
          <textarea
            value={samplePrompt}
            onChange={(e) => setSamplePrompt(e.target.value)}
            placeholder="Enter prompt for sampling..."
            rows={2}
            className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none"
          />
        </div>
        <div>
          <label className="text-xs text-forge-muted">Sampler</label>
          <select
            value={samplerType}
            onChange={(e) => setSamplerType(e.target.value)}
            className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm focus:border-forge-accent focus:outline-none"
          >
            <option value="euler">Euler</option>
            <option value="euler_a">Euler Ancestral</option>
            <option value="dpm++">DPM++ 2M</option>
          </select>
        </div>
        <div>
          <label className="text-xs text-forge-muted">Steps</label>
          <input
            type="number" value={sampleSteps}
            onChange={(e) => setSampleSteps(parseInt(e.target.value) || 8)}
            className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none"
          />
        </div>
        <div>
          <label className="text-xs text-forge-muted">Seed</label>
          <input
            type="number" value={sampleSeed}
            onChange={(e) => setSampleSeed(parseInt(e.target.value) || 42)}
            className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none"
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={handleGenerate}
            className="px-4 py-2 bg-blue-500 text-white rounded text-sm font-medium hover:bg-blue-600"
          >
            Generate
          </button>
        </div>
      </div>
    </div>
  );
}
