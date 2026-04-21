import React from 'react';

export default function PipelineSection({ Section, Field, config, upd }) {
  const pipelineName = config.pipeline?.name;

  const updateParam = (key, value) => {
    const params = { ...(config.pipeline?.params || {}), [key]: value };
    upd('pipeline', 'params', params);
  };

  return (
    <Section title="Pipeline" defaultOpen={false}>
      <Field
        label="Pipeline"
        value={pipelineName || ''}
        onChange={(v) => upd('pipeline', 'name', v || null)}
        type="select"
        options={[
          { value: '', label: '— None (non-diffusion) —' },
          { value: 'zimage_turbo', label: 'Z-Image Turbo (S3-DiT)' },
          { value: 'flux', label: 'Flux (coming soon)' },
          { value: 'sdxl', label: 'SDXL (coming soon)' },
        ]}
        help="Diffusion pipeline for training + sampling"
      />

      {pipelineName === 'zimage_turbo' && (
        <>
          <Field
            label="Time Shift"
            value={config.pipeline?.params?.shift ?? 1.0}
            onChange={(v) => updateParam('shift', parseFloat(v))}
            type="number"
            help="Flow matching time shift (1.0 = uniform)"
          />
          <Field
            label="Loss Type"
            value={config.pipeline?.params?.loss_type || 'mse'}
            onChange={(v) => updateParam('loss_type', v)}
            type="select"
            options={['mse', 'huber']}
          />
          <Field
            label="SNR Gamma"
            value={config.pipeline?.params?.snr_gamma ?? ''}
            onChange={(v) => updateParam('snr_gamma', v === '' ? null : parseFloat(v))}
            type="number"
            help="Min-SNR weighting (blank = disabled, 5.0 recommended)"
          />

          {/* Training Adapter */}
          <div className="col-span-2 border-t border-forge-border pt-3">
            <label className="text-xs text-forge-muted uppercase tracking-wide">Training Adapter</label>
            <p className="text-xs text-forge-muted/60 mt-0.5 mb-2">
              Frozen base LoRA injected into the denoiser before training. Your trainable LoRA stacks on top.
            </p>
            <Field
              label="Path (optional)"
              value={config.pipeline?.training_adapter?.path || ''}
              onChange={(v) => upd('pipeline', 'training_adapter', v ? { path: v } : undefined)}
              help="e.g. loras/my_base_character.safetensors"
            />
          </div>
        </>
      )}
    </Section>
  );
}
