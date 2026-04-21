import React from 'react';

export default function SamplingSection({ Section, Field, config, upd }) {
  return (
    <Section title="Sampling" defaultOpen={false} badge={config.sampling?.enabled ? 'ON' : null}>
      <Field label="Enable Sampling" value={config.sampling?.enabled || false}
        onChange={(v) => upd('sampling', 'enabled', v)} type="checkbox" />
      <Field label="Every N Steps" value={config.sampling?.every_n_steps || 500}
        onChange={(v) => upd('sampling', 'every_n_steps', parseInt(v))} type="number" />
      <Field label="Sampler" value={config.sampling?.sampler || 'euler'}
        onChange={(v) => upd('sampling', 'sampler', v)}
        type="select" options={['euler', 'euler_a', 'dpm++']} />
      <Field label="Num Steps" value={config.sampling?.num_steps || 8}
        onChange={(v) => upd('sampling', 'num_steps', parseInt(v))} type="number" />
      <Field label="Guidance Scale" value={config.sampling?.guidance_scale ?? 0.0}
        onChange={(v) => upd('sampling', 'guidance_scale', parseFloat(v))} type="number"
        help="0.0 for distilled models (Turbo)" />
      <Field label="Seed" value={config.sampling?.seed || 42}
        onChange={(v) => upd('sampling', 'seed', parseInt(v))} type="number" />
      <Field label="Width" value={config.sampling?.width || 1024}
        onChange={(v) => upd('sampling', 'width', parseInt(v))} type="number" />
      <Field label="Height" value={config.sampling?.height || 1024}
        onChange={(v) => upd('sampling', 'height', parseInt(v))} type="number" />
      <div className="col-span-2">
        <label className="text-xs text-forge-muted uppercase tracking-wide">Prompts (one per line)</label>
        <textarea
          value={(config.sampling?.prompts || []).join('\n')}
          onChange={(e) => upd('sampling', 'prompts',
            e.target.value.split('\n').filter(Boolean))}
          rows={4}
          className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2 text-sm font-mono focus:border-forge-accent focus:outline-none"
          placeholder={"portrait of <character> smiling\n<character> walking in park"}
        />
      </div>
    </Section>
  );
}
