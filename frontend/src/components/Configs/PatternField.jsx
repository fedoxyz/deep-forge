import React, { useState, useRef, useEffect } from 'react';
import { X, Copy, Check, AlertCircle, ExternalLink } from 'lucide-react';

/**
 * PatternField — a tag-input for regex pattern arrays.
 * 
 * Props:
 *   value        — string[]  (the current patterns array)
 *   onChange     — (string[]) => void
 *   label        — string
 *   help         — string | undefined
 *   fallback     — string[]  (global/parent patterns, shown as placeholder when empty)
 *   onOpenInspector — () => void  (optional — shows "Open in Inspector" button)
 */
export default function PatternField({ value = [], onChange, label, help, fallback = [], onOpenInspector }) {
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const patterns = value;

  const validate = (p) => {
    try { new RegExp(p); return null; }
    catch (e) { return e.message; }
  };

  const commit = (raw) => {
    const p = raw.trim();
    if (!p) return;
    const err = validate(p);
    if (err) { setError(err); return; }
    if (!patterns.includes(p)) onChange([...patterns, p]);
    setDraft('');
    setError('');
  };

  const remove = (i) => onChange(patterns.filter((_, idx) => idx !== i));

  const onKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); commit(draft); }
    if (e.key === 'Backspace' && !draft && patterns.length) remove(patterns.length - 1);
    if (e.key === 'Escape') { setDraft(''); setError(''); }
  };

  // Accept patterns pasted from workbench via a custom event
  useEffect(() => {
    const handler = (e) => {
      if (e.detail?.field !== label) return;
      const incoming = e.detail.patterns || [];
      const merged = [...new Set([...patterns, ...incoming])];
      onChange(merged);
    };
    document.addEventListener('forge:inject-patterns', handler);
    return () => document.removeEventListener('forge:inject-patterns', handler);
  }, [patterns, onChange, label]);

  const isEmpty = patterns.length === 0;
  const isUsingFallback = isEmpty && fallback.length > 0;

  return (
    <div className="col-span-2 space-y-1.5">
      {/* label row */}
      <div className="flex items-center justify-between">
        <label className="text-xs text-forge-muted uppercase tracking-wide">{label}</label>
        {onOpenInspector && (
          <button
            type="button"
            onClick={onOpenInspector}
            className="flex items-center gap-1 text-xs text-forge-accent hover:text-forge-accent/80 transition-colors"
          >
            <ExternalLink className="w-3 h-3" />
            Open in Inspector
          </button>
        )}
      </div>

      {/* tag input box */}
      <div
        className={`min-h-[42px] flex flex-wrap gap-1.5 px-2 py-1.5 bg-forge-bg border rounded cursor-text transition-colors focus-within:border-forge-accent ${
          error ? 'border-forge-error/60' : 'border-forge-border'
        }`}
        onClick={() => inputRef.current?.focus()}
      >
        {/* existing pattern tags */}
        {patterns.map((p, i) => (
          <PatternTag key={i} pattern={p} onRemove={() => remove(i)} />
        ))}

        {/* fallback hint when empty */}
        {isUsingFallback && patterns.length === 0 && draft === '' && (
          <div className="flex flex-wrap gap-1 items-center">
            <span className="text-[10px] text-forge-muted/50 mr-1">global fallback:</span>
            {fallback.slice(0, 3).map((p, i) => (
              <span key={i} className="text-[10px] px-1.5 py-0.5 rounded border border-dashed border-forge-border text-forge-muted/40 font-mono">{p}</span>
            ))}
            {fallback.length > 3 && <span className="text-[10px] text-forge-muted/40">+{fallback.length - 3} more</span>}
          </div>
        )}

        {/* draft input */}
        <input
          ref={inputRef}
          value={draft}
          onChange={e => { setDraft(e.target.value); setError(''); }}
          onKeyDown={onKeyDown}
          onBlur={() => { if (draft.trim()) commit(draft); }}
          placeholder={patterns.length === 0 ? 'Type a pattern, press Enter…' : ''}
          className="flex-1 min-w-32 bg-transparent text-xs font-mono text-forge-text placeholder:text-forge-muted/40 outline-none py-0.5"
        />
      </div>

      {/* error */}
      {error && (
        <div className="flex items-center gap-1.5 text-xs text-forge-error">
          <AlertCircle className="w-3 h-3 shrink-0" /> {error}
        </div>
      )}

      {/* help */}
      {help && !error && (
        <div className="text-xs text-forge-muted/60">{help}</div>
      )}

      {/* paste hint */}
      {patterns.length === 0 && !help && (
        <div className="text-xs text-forge-muted/50">
          Use the Layer Inspector to build patterns, then copy them here — or type a regex and press Enter.
        </div>
      )}
    </div>
  );
}

// ── PatternTag ────────────────────────────────────────────────────────────────
function PatternTag({ pattern, onRemove }) {
  const [copied, setCopied] = useState(false);
  const { err } = (() => { try { new RegExp(pattern); return { err: null }; } catch (e) { return { err: e.message }; } })();

  const copy = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(pattern);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <span
      title={err ? `Invalid regex: ${err}` : pattern}
      className={`group inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded font-mono max-w-[280px] border ${
        err
          ? 'bg-forge-error/10 border-forge-error/40 text-forge-error'
          : 'bg-forge-accent/10 border-forge-accent/25 text-forge-accent'
      }`}
    >
      <span className="truncate">{pattern}</span>
      <button onClick={copy} className="shrink-0 opacity-0 group-hover:opacity-60 hover:!opacity-100 transition-opacity">
        {copied ? <Check className="w-2.5 h-2.5" /> : <Copy className="w-2.5 h-2.5" />}
      </button>
      <button onClick={(e) => { e.stopPropagation(); onRemove(); }} className="shrink-0 opacity-0 group-hover:opacity-60 hover:!opacity-100 transition-opacity">
        <X className="w-2.5 h-2.5" />
      </button>
    </span>
  );
}
