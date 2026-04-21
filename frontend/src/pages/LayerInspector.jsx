import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Search, Layers, ChevronRight, Package, Copy, Check, Plus, X, Download, Zap, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { inspectModel, getPresets } from '../utils/api';

// ─── utils ────────────────────────────────────────────────────────────────────

function safeRegex(p) {
  try { return { re: new RegExp(p), err: null }; }
  catch (e) { return { re: null, err: e.message }; }
}

function buildRangeRegex(from, to) {
  if (from > to) return null;
  const nums = Array.from({ length: to - from + 1 }, (_, i) => i + from);
  const singles = nums.filter(n => n < 10);
  const tens = nums.filter(n => n >= 10 && n < 100);
  const parts = [];
  if (singles.length === 1) parts.push(String(singles[0]));
  else if (singles.length > 1) parts.push(`[${singles[0]}-${singles[singles.length - 1]}]`);
  const tenGroups = {};
  tens.forEach(n => { const d = Math.floor(n / 10); (tenGroups[d] = tenGroups[d] || []).push(n % 10); });
  Object.entries(tenGroups).forEach(([d, ds]) => {
    if (ds.length === 1) parts.push(`${d}${ds[0]}`);
    else if (ds[0] === 0 && ds[ds.length - 1] === 9) parts.push(`${d}[0-9]`);
    else if (ds[ds.length - 1] - ds[0] === ds.length - 1) parts.push(`${d}[${ds[0]}-${ds[ds.length - 1]}]`);
    else parts.push(ds.map(x => `${d}${x}`).join('|'));
  });
  return parts.length === 1 ? parts[0] : `(${parts.join('|')})`;
}

// Decompose a layer name into annotated segments so we can explain each part
// e.g. "transformer_blocks.12.attn1.to_q.weight"
// → [{seg:"transformer_blocks", kind:"prefix"}, {seg:"12", kind:"index"}, ...]
function decomposeName(name) {
  const parts = name.split('.');
  return parts.map(p => {
    if (/^\d+$/.test(p)) return { seg: p, kind: 'index', tip: 'Block/layer index — use \\d+ for any, or a range like (8|[9]|1[0-9]) for specific blocks' };
    if (/^(to_q|to_k|to_v|to_out)/.test(p)) return { seg: p, kind: 'proj', tip: 'Attention projection — combine like (to_q|to_k|to_v|to_out) to target multiple' };
    if (/^(weight|bias)$/.test(p)) return { seg: p, kind: 'param', tip: 'Parameter tensor — usually omit this to match both weight and bias' };
    if (/attn|attention/i.test(p)) return { seg: p, kind: 'attn', tip: 'Attention module — use attn\\d? to match attn, attn1, attn2' };
    if (/^(ff|mlp|feed_forward)$/.test(p)) return { seg: p, kind: 'ff', tip: 'Feed-forward / MLP block' };
    if (/conv/.test(p)) return { seg: p, kind: 'conv', tip: 'Convolution layer' };
    if (/norm|ln_/.test(p)) return { seg: p, kind: 'norm', tip: 'Normalization layer — typically frozen, not targeted for LoRA' };
    if (/proj/.test(p)) return { seg: p, kind: 'proj', tip: 'Projection layer' };
    if (/^(net|0|1|2)$/.test(p)) return { seg: p, kind: 'index', tip: 'Sub-module index inside a sequential' };
    return { seg: p, kind: 'prefix', tip: 'Module name — use as literal or escape dots with \\.' };
  });
}

// Given a set of selected layer names, infer a minimal pattern that covers them
function inferPattern(selected, allLayers) {
  if (!selected.length) return '';
  if (selected.length === 1) {
    // exact name minus .weight/.bias suffix
    return selected[0].replace(/\.(weight|bias)$/, '');
  }

  // Find common prefix segments
  const splitAll = selected.map(n => n.split('.'));
  const minLen = Math.min(...splitAll.map(p => p.length));
  const commonSegs = [];
  for (let i = 0; i < minLen; i++) {
    const vals = [...new Set(splitAll.map(p => p[i]))];
    if (vals.length === 1 && !/^\d+$/.test(vals[0])) {
      commonSegs.push(vals[0]);
    } else if (vals.every(v => /^\d+$/.test(v))) {
      const nums = vals.map(Number);
      const min = Math.min(...nums), max = Math.max(...nums);
      const rng = buildRangeRegex(min, max);
      commonSegs.push(rng || '\\d+');
    } else {
      // diverging — collect unique non-index values
      const nonIdx = vals.filter(v => !/^\d+$/.test(v));
      if (nonIdx.length <= 4) commonSegs.push(`(${[...new Set(nonIdx)].join('|')})`);
      else commonSegs.push('[^.]+');
    }
  }
  // Drop trailing .weight/.bias
  while (commonSegs.length && /^\.(weight|bias)$/.test('.' + commonSegs[commonSegs.length - 1])) commonSegs.pop();
  return commonSegs.join('\\.');
}

function layerColor(name) {
  if (/attn|attention/.test(name)) return 'text-forge-accent';
  if (/conv/.test(name)) return 'text-purple-400';
  if (/mlp|ff\./.test(name)) return 'text-blue-400';
  if (/proj/.test(name)) return 'text-green-400';
  return 'text-forge-muted';
}

// ─── CopyButton ───────────────────────────────────────────────────────────────
function CopyButton({ text, label, size = 'sm' }) {
  const [ok, setOk] = useState(false);
  const copy = () => { navigator.clipboard.writeText(text); setOk(true); setTimeout(() => setOk(false), 1400); };
  return (
    <button onClick={copy} className={`flex items-center gap-1 font-mono transition-colors rounded border ${
      size === 'xs' ? 'text-xs px-1.5 py-0.5' : 'text-xs px-2 py-1'
    } ${ok
      ? 'bg-green-500/15 border-green-500/30 text-green-400'
      : 'bg-forge-bg border-forge-border text-forge-muted hover:border-forge-accent hover:text-forge-text'
    }`}>
      {ok ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
      {label && <span>{ok ? 'Copied' : label}</span>}
    </button>
  );
}

// ─── LayerAnnotation — click a layer, get a breakdown + pattern suggestions ──
function LayerAnnotation({ name, allLayers, onAddPattern }) {
  const segments = decomposeName(name);
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const [expandFamily, setExpandFamily] = useState(false);

  // Generate a few useful pattern suggestions based on structure
  const suggestions = useMemo(() => {
    const parts = name.split('.');
    const results = [];

    // 1. Exact layer (no param suffix)
    const noParam = name.replace(/\.(weight|bias)$/, '');
    results.push({
      label: 'Exact layer',
      pattern: noParam,
      desc: 'Targets only this specific layer',
    });

    // 2. All projections of same type in all blocks
    const projMatch = name.match(/\.(to_q|to_k|to_v|to_out|q_proj|k_proj|v_proj|out_proj)(\.|$)/);
    if (projMatch) {
      const proj = projMatch[1];
      results.push({
        label: `All ${proj} across blocks`,
        pattern: `\\.${proj}(\\.weight)?$`,
        desc: `Matches every ${proj} in every block — good for targeting one projection type`,
      });
      results.push({
        label: 'All attn projections',
        pattern: `\\.(to_q|to_k|to_v|to_out|q_proj|k_proj|v_proj|out_proj)`,
        desc: 'All attention projections everywhere — widest attn target',
      });
    }

    // 3. Everything in the same attn/ff module
    const attnMatch = name.match(/(.+?\.(attn\d?|attention\d?))\./);
    if (attnMatch) {
      results.push({
        label: 'All of this attention module',
        pattern: attnMatch[1].replace(/\./g, '\\.'),
        desc: 'All layers inside this specific attn module (one block)',
      });
    }

    // 4. Index-based: same module, middle blocks only
    const idxMatch = parts.findIndex(p => /^\d+$/.test(p));
    if (idxMatch >= 0) {
      const total = [...new Set(allLayers.map(l => {
        const ps = l.split('.');
        return ps[idxMatch];
      }).filter(v => /^\d+$/.test(v)))].map(Number);
      const maxIdx = Math.max(...total);
      const lo = Math.floor(maxIdx * 0.3), hi = Math.floor(maxIdx * 0.75);
      const prefix = parts.slice(0, idxMatch).join('\\.');
      const suffix = parts.slice(idxMatch + 1).filter(p => !/^(weight|bias)$/.test(p)).join('\\.');
      const rng = buildRangeRegex(lo, hi) || '\\d+';
      if (lo < hi) {
        results.push({
          label: `Middle blocks (${lo}–${hi})`,
          pattern: `${prefix}\\.${rng}\\.${suffix}`,
          desc: `Targets only middle-depth blocks — useful for identity preservation (avoids early/late blocks)`,
        });
      }
    }

    // 5. FF layers pattern
    const ffMatch = name.match(/\.(ff|mlp|feed_forward)\./);
    if (ffMatch) {
      results.push({
        label: 'All FF/MLP layers',
        pattern: `\\.(ff|mlp|feed_forward)\\.`,
        desc: 'Every feed-forward layer — combine with attn patterns for style learning',
      });
    }

    return results;
  }, [name, allLayers]);

  // Count how many layers each suggestion matches
  const withCounts = suggestions.map(s => {
    const { re } = safeRegex(s.pattern);
    const count = re ? allLayers.filter(l => re.test(l)).length : 0;
    return { ...s, count };
  });

  // Family: all layers that share the same non-index structure
  const familyPattern = useMemo(() => {
    const parts = name.split('.');
    return parts.map(p => /^\d+$/.test(p) ? '\\d+' : p).join('\\.').replace(/\.(weight|bias)$/, '');
  }, [name]);
  const family = useMemo(() => {
    const { re } = safeRegex(familyPattern);
    return re ? allLayers.filter(l => re.test(l)) : [];
  }, [familyPattern, allLayers]);

  return (
    <div className="bg-forge-bg border border-forge-border rounded-lg overflow-hidden text-xs">
      {/* Decomposed name */}
      <div className="px-4 py-3 border-b border-forge-border">
        <div className="text-forge-muted mb-2 uppercase tracking-wide text-[10px]">Layer structure</div>
        <div className="flex flex-wrap items-center gap-px font-mono">
          {segments.map((seg, i) => (
            <React.Fragment key={i}>
              <button
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
                className={`px-1.5 py-0.5 rounded transition-colors ${
                  seg.kind === 'index' ? 'bg-amber-500/15 text-amber-400 hover:bg-amber-500/25' :
                  seg.kind === 'attn'  ? 'bg-forge-accent/15 text-forge-accent hover:bg-forge-accent/25' :
                  seg.kind === 'proj'  ? 'bg-green-500/15 text-green-400 hover:bg-green-500/25' :
                  seg.kind === 'ff'    ? 'bg-blue-500/15 text-blue-400 hover:bg-blue-500/25' :
                  seg.kind === 'conv'  ? 'bg-purple-500/15 text-purple-400 hover:bg-purple-500/25' :
                  seg.kind === 'param' ? 'bg-forge-muted/10 text-forge-muted hover:bg-forge-muted/20' :
                  seg.kind === 'norm'  ? 'bg-red-500/10 text-red-400/70 hover:bg-red-500/15' :
                  'bg-forge-surface text-forge-text hover:bg-forge-border'
                }`}
              >
                {seg.seg}
              </button>
              {i < segments.length - 1 && <span className="text-forge-muted/40">.</span>}
            </React.Fragment>
          ))}
        </div>
        {hoveredIdx !== null && (
          <div className="mt-2 p-2 bg-forge-surface border border-forge-border rounded text-forge-muted leading-relaxed">
            <span className={`font-mono mr-1 ${
              segments[hoveredIdx].kind === 'index' ? 'text-amber-400' :
              segments[hoveredIdx].kind === 'attn' ? 'text-forge-accent' :
              segments[hoveredIdx].kind === 'proj' ? 'text-green-400' : 'text-forge-text'
            }`}>{segments[hoveredIdx].seg}</span>
            — {segments[hoveredIdx].tip}
          </div>
        )}
      </div>

      {/* Family */}
      <div className="px-4 py-3 border-b border-forge-border">
        <button
          onClick={() => setExpandFamily(v => !v)}
          className="w-full flex items-center justify-between hover:text-forge-text transition-colors"
        >
          <span className="text-forge-muted uppercase tracking-wide text-[10px]">
            Same-structure family
            <span className="ml-2 normal-case text-forge-text font-mono">{family.length} layers</span>
          </span>
          {expandFamily ? <ChevronUp className="w-3 h-3 text-forge-muted" /> : <ChevronDown className="w-3 h-3 text-forge-muted" />}
        </button>
        <div className="mt-1 font-mono text-forge-accent text-[11px] flex items-center gap-2">
          <span className="break-all">{familyPattern}</span>
          <CopyButton text={familyPattern} size="xs" />
          <button onClick={() => onAddPattern(familyPattern)} className="text-[10px] px-1.5 py-0.5 bg-forge-accent/20 text-forge-accent rounded hover:bg-forge-accent/30 whitespace-nowrap">+ Add</button>
        </div>
        {expandFamily && (
          <div className="mt-2 max-h-32 overflow-y-auto space-y-0.5">
            {family.map((l, i) => (
              <div key={i} className="font-mono text-forge-muted pl-2 hover:text-forge-text">{l}</div>
            ))}
          </div>
        )}
      </div>

      {/* Suggestions */}
      <div className="px-4 py-3">
        <div className="text-forge-muted uppercase tracking-wide text-[10px] mb-2">Pattern suggestions</div>
        <div className="space-y-2">
          {withCounts.map((s, i) => (
            <div key={i} className="group flex items-start gap-2 p-2 rounded hover:bg-forge-surface transition-colors">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-forge-text font-medium">{s.label}</span>
                  <span className="text-forge-muted">→ {s.count} layer{s.count !== 1 ? 's' : ''}</span>
                </div>
                <code className="text-forge-accent font-mono text-[11px] break-all">{s.pattern}</code>
                <div className="text-forge-muted/70 mt-0.5 leading-relaxed">{s.desc}</div>
              </div>
              <div className="flex flex-col gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                <CopyButton text={s.pattern} size="xs" />
                <button
                  onClick={() => onAddPattern(s.pattern)}
                  className="text-[10px] px-1.5 py-0.5 bg-forge-accent text-black rounded hover:bg-forge-accent/90 whitespace-nowrap"
                >+ Add</button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── SendButton — fires forge:inject-patterns to a named PatternField ─────────
function SendButton({ label, field, patterns, help }) {
  const [sent, setSent] = useState(false);
  const fire = () => {
    document.dispatchEvent(new CustomEvent('forge:inject-patterns', { detail: { field, patterns } }));
    setSent(true);
    setTimeout(() => setSent(false), 1600);
  };
  return (
    <button
      onClick={fire}
      title={help}
      className={`flex flex-col items-start gap-0.5 px-3 py-2 rounded border text-xs transition-colors text-left ${
        sent
          ? 'bg-green-500/15 border-green-500/30 text-green-400'
          : 'bg-forge-bg border-forge-border text-forge-muted hover:border-forge-accent hover:text-forge-text'
      }`}
    >
      <span className="font-medium">{sent ? '✓ Sent' : label}</span>
      <span className="text-[10px] opacity-60">{help}</span>
    </button>
  );
}


function PatternWorkbench({ layers, patterns = [], onPatternsChange }) {
  const setPatterns = (fn) => onPatternsChange(typeof fn === 'function' ? fn(patterns) : fn);
  const [input, setInput] = useState('');
  const [exportFmt, setExportFmt] = useState('yaml');

  const currentRe = useMemo(() => input.trim() ? safeRegex(input.trim()) : { re: null, err: null }, [input]);

  const compiledAll = useMemo(() => {
    if (!patterns.length) return null;
    try { return new RegExp(patterns.map(p => `(?:${p})`).join('|')); }
    catch { return null; }
  }, [patterns]);

  const previewMatches = useMemo(() => {
    if (!currentRe.re || !layers.length) return [];
    return layers.filter(l => currentRe.re.test(l));
  }, [currentRe.re, layers]);

  const matchedByAll = useMemo(() => {
    if (!compiledAll || !layers.length) return new Set();
    return new Set(layers.filter(l => compiledAll.test(l)));
  }, [compiledAll, layers]);

  const addPattern = (p) => {
    const v = (p || input).trim();
    if (!v || patterns.includes(v)) return;
    setPatterns(prev => [...prev, v]);
    if (!p) setInput('');
  };

  const removePattern = (i) => setPatterns(prev => prev.filter((_, idx) => idx !== i));

  const coverage = layers.length ? Math.round(matchedByAll.size / layers.length * 100) : 0;

  const exportText = useMemo(() => {
    if (!patterns.length) return '# no patterns added yet';
    if (exportFmt === 'yaml') return `lora:\n  target_patterns:\n${patterns.map(p => `    - '${p}'`).join('\n')}`;
    if (exportFmt === 'json') return JSON.stringify(patterns, null, 2);
    return `target_patterns = [\n${patterns.map(p => `    r"${p}",`).join('\n')}\n]`;
  }, [patterns, exportFmt]);

  return (
    <div className="bg-forge-surface border border-forge-border rounded-lg overflow-hidden">
      {/* header + stats */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-forge-border">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-forge-accent" />
          <span className="text-sm font-medium">Pattern Workbench</span>
          <span className="text-xs text-forge-muted">— patterns go into <code className="bg-forge-bg rounded px-1">lora.target_patterns</code></span>
        </div>
        {layers.length > 0 && patterns.length > 0 && (
          <div className="flex items-center gap-3 text-xs">
            <span className="text-forge-muted">{matchedByAll.size}/{layers.length} layers</span>
            <span className="font-mono font-bold text-forge-accent">{coverage}%</span>
            <div className="w-20 h-1.5 bg-forge-bg rounded-full overflow-hidden">
              <div className="h-full bg-forge-accent rounded-full transition-all" style={{ width: `${coverage}%` }} />
            </div>
          </div>
        )}
      </div>

      <div className="p-5 space-y-4">
        {/* input row */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addPattern()}
              placeholder='regex pattern — e.g. transformer_blocks\.\d+\.attn\.(to_q|to_v)'
              className={`w-full bg-forge-bg border rounded px-3 py-2.5 text-sm font-mono focus:outline-none transition-colors ${
                currentRe.err ? 'border-forge-error/60 text-forge-error'
                : input.trim() ? 'border-forge-accent/50' : 'border-forge-border'
              }`}
            />
            {currentRe.err && (
              <div className="absolute top-full left-0 mt-1 z-10 text-xs text-forge-error bg-forge-bg border border-forge-error/30 rounded px-2 py-1 whitespace-nowrap">
                {currentRe.err}
              </div>
            )}
          </div>
          <button
            onClick={() => addPattern()}
            disabled={!input.trim() || !!currentRe.err}
            className="flex items-center gap-1.5 px-4 py-2 bg-forge-accent text-black rounded text-sm font-medium hover:bg-forge-accent/90 disabled:opacity-40 shrink-0"
          >
            <Plus className="w-4 h-4" /> Add
          </button>
        </div>

        {/* live preview */}
        {input.trim() && !currentRe.err && layers.length > 0 && (
          <div className="text-xs">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-forge-muted">Preview:</span>
              <span className={previewMatches.length > 0 ? 'text-green-400 font-mono' : 'text-forge-muted'}>
                {previewMatches.length} match{previewMatches.length !== 1 ? 'es' : ''} of {layers.length}
              </span>
            </div>
            {previewMatches.length > 0 && (
              <div className="max-h-28 overflow-y-auto bg-forge-bg border border-forge-border rounded divide-y divide-forge-border/40">
                {previewMatches.slice(0, 100).map((n, i) => (
                  <div key={i} className="px-3 py-1 font-mono text-forge-accent">{n}</div>
                ))}
                {previewMatches.length > 100 && <div className="px-3 py-1 text-forge-muted/60">…and {previewMatches.length - 100} more</div>}
              </div>
            )}
          </div>
        )}

        {/* pattern chips */}
        {patterns.length > 0 ? (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-forge-muted uppercase tracking-wide">Added patterns ({patterns.length})</span>
              <button onClick={() => setPatterns([])} className="text-xs text-forge-muted hover:text-forge-error">Clear all</button>
            </div>
            <div className="flex flex-wrap gap-2">
              {patterns.map((p, i) => {
                const { re, err } = safeRegex(p);
                const cnt = re && layers.length ? layers.filter(l => re.test(l)).length : null;
                return (
                  <span key={i} className={`inline-flex items-center gap-1.5 text-xs px-2 py-1.5 rounded font-mono border ${
                    err ? 'bg-forge-error/10 border-forge-error/30 text-forge-error'
                    : 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent'
                  }`}>
                    <span className="break-all">{p}</span>
                    {cnt !== null && <span className="text-forge-muted shrink-0">({cnt})</span>}
                    <button onClick={() => removePattern(i)} className="hover:text-forge-text ml-0.5 shrink-0"><X className="w-3 h-3" /></button>
                  </span>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="text-xs text-forge-muted italic">
            No patterns yet. Click a layer below to get suggestions, or type a pattern above.
          </div>
        )}

        {/* send to config + export */}
        {patterns.length > 0 && (
          <div className="border-t border-forge-border pt-4 space-y-3">

            {/* ── send to config fields ── */}
            <div>
              <div className="text-xs text-forge-muted uppercase tracking-wide mb-2">Send to config field</div>
              <div className="grid grid-cols-2 gap-2">
                <SendButton
                  label="→ LoRA Defaults (global)"
                  field="Default Target Patterns"
                  patterns={patterns}
                  help="Sets lora.target_patterns — fallback for all components"
                />
                <SendButton
                  label="→ Active component"
                  field="Target Patterns"
                  patterns={patterns}
                  help="Sets training.lora.target_patterns on the selected component"
                />
              </div>
            </div>

            {/* ── copy as text ── */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex gap-1.5">
                  {['yaml', 'json', 'py'].map(f => (
                    <button key={f} onClick={() => setExportFmt(f)} className={`text-xs px-2.5 py-1 rounded border font-mono transition-colors ${
                      exportFmt === f ? 'bg-forge-accent text-black border-forge-accent' : 'bg-forge-bg border-forge-border text-forge-muted hover:border-forge-accent'
                    }`}>{f}</button>
                  ))}
                </div>
                <CopyButton text={exportText} label="Copy text" />
              </div>
              <pre className="bg-forge-bg border border-forge-border rounded p-3 text-xs font-mono text-forge-text overflow-x-auto whitespace-pre-wrap">{exportText}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function LayerInspector() {
  const [modelPath, setModelPath] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('');
  const [expandedGroups, setExpandedGroups] = useState({});
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [workbenchPatterns, setWorkbenchPatterns] = useState([]);
  const workbenchRef = useRef(null);
  const [workbenchAddFn, setWorkbenchAddFn] = useState(null);

  // We lift addPattern out via a callback ref
  const [addToWorkbench, setAddToWorkbench] = useState(() => () => {});

  const handleInspect = async (path = modelPath) => {
    const p = (path || '').trim();
    if (!p) return;
    setLoading(true);
    setError('');
    setSelectedLayer(null);
    try {
      const data = await inspectModel(p);
      setResult(data);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  const allLayerNames = useMemo(() => [
    ...(result?.linear_layers || []),
    ...(result?.conv_layers || []),
  ], [result]);

  const filteredLayers = useMemo(() => allLayerNames.filter(name =>
    filter ? name?.toLowerCase().includes(filter.toLowerCase()) : true
  ), [allLayerNames, filter]);

  // Workbench state lifted up so LayerAnnotation can add to it
  const [patterns, setPatterns] = useState([]);

  const addPattern = (p) => {
    setPatterns(prev => prev.includes(p) ? prev : [...prev, p]);
    // scroll to workbench
    workbenchRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Layer Inspector</h1>
        <p className="text-sm text-forge-muted mt-1">
          Inspect a model, click any layer to understand its structure and get ready-to-use regex patterns for{' '}
          <code className="bg-forge-surface rounded px-1">lora.target_patterns</code>.
        </p>
      </div>

      {/* Inspect input */}
      <div className="flex items-end gap-3">
        <div className="flex-1">
          <label className="text-xs text-forge-muted uppercase tracking-wide block mb-1.5">Model Path</label>
          <input
            value={modelPath}
            onChange={e => setModelPath(e.target.value)}
            placeholder="/data/models/model.safetensors"
            onKeyDown={e => e.key === 'Enter' && handleInspect()}
            className="w-full bg-forge-bg border border-forge-border rounded px-3 py-2.5 text-sm font-mono focus:border-forge-accent focus:outline-none"
          />
        </div>
        <button
          onClick={() => handleInspect()}
          disabled={loading}
          className="flex items-center gap-2 px-5 py-2.5 bg-forge-accent text-black rounded text-sm font-medium hover:bg-forge-accent/90 disabled:opacity-50"
        >
          <Layers className="w-4 h-4" /> {loading ? 'Inspecting...' : 'Inspect'}
        </button>
      </div>

      {error && <div className="p-3 bg-forge-error/10 border border-forge-error/30 rounded text-sm text-forge-error">{error}</div>}

      {result && (
        <>
          {/* Summary */}
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Total Params', value: result.total_params_human, sub: result.total_params?.toLocaleString(), color: 'text-blue-400' },
              { label: 'Total Keys', value: result.num_keys, color: '' },
              { label: 'Linear Layers', value: result.linear_layers?.length || 0, color: 'text-forge-accent' },
              { label: 'Conv Layers', value: result.conv_layers?.length || 0, color: 'text-purple-400' },
            ].map(({ label, value, sub, color }) => (
              <div key={label} className="bg-forge-surface border border-forge-border rounded-lg p-4">
                <div className="text-xs text-forge-muted mb-1">{label}</div>
                <div className={`text-xl font-bold font-mono ${color}`}>{value}</div>
                {sub && <div className="text-xs text-forge-muted mt-1">{sub}</div>}
              </div>
            ))}
          </div>

          {/* Groups */}
          {result.group_names?.length > 0 && (
            <div className="bg-forge-surface border border-forge-border rounded-lg p-5">
              <h2 className="text-sm font-medium mb-3 flex items-center gap-2">
                <Package className="w-4 h-4" /> Model Groups ({result.group_names.length})
              </h2>
              <div className="space-y-2">
                {result.group_names.map(groupName => {
                  const groupLayers = result.groups?.[groupName] || [];
                  const isExpanded = expandedGroups[groupName] || false;
                  return (
                    <div key={groupName} className="bg-forge-bg border border-forge-border rounded overflow-hidden">
                      <button
                        onClick={() => setExpandedGroups(prev => ({ ...prev, [groupName]: !prev[groupName] }))}
                        className="w-full flex items-center justify-between px-3 py-2 hover:bg-forge-accent/5 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-forge-muted text-xs">{isExpanded ? '▾' : '▸'}</span>
                          <span className="text-xs font-mono text-forge-accent break-all">{groupName}</span>
                        </div>
                        <span className="text-xs text-forge-muted shrink-0 ml-2">{groupLayers.length} layer{groupLayers.length !== 1 ? 's' : ''}</span>
                      </button>
                      {isExpanded && (
                        <div className="px-3 py-2 border-t border-forge-border bg-forge-bg/50">
                          <div className="space-y-0.5 max-h-48 overflow-y-auto">
                            {groupLayers.slice(0, 100).map((item, idx) => {
                              const ln = typeof item === 'string' ? item : item?.name || item?.key || JSON.stringify(item);
                              return (
                                <button
                                  key={idx}
                                  onClick={() => setSelectedLayer(ln)}
                                  className={`w-full text-left text-xs font-mono pl-4 py-0.5 rounded transition-colors ${
                                    selectedLayer === ln ? 'text-forge-accent bg-forge-accent/10' : 'text-forge-muted hover:text-forge-text'
                                  }`}
                                >{ln}</button>
                              );
                            })}
                            {groupLayers.length > 100 && <div className="text-xs text-forge-muted/60 pl-4 py-1">…and {groupLayers.length - 100} more</div>}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Layer Browser + Annotation side-by-side */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {/* browser */}
            <div className="bg-forge-surface border border-forge-border rounded-lg p-5">
              <div className="flex items-center gap-3 mb-4">
                <h2 className="text-sm font-medium shrink-0">Layers ({allLayerNames.length.toLocaleString()})</h2>
                <div className="flex-1 relative">
                  <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-forge-muted" />
                  <input
                    value={filter}
                    onChange={e => setFilter(e.target.value)}
                    placeholder="Filter... (attn, q_proj, conv, mlp)"
                    className="w-full bg-forge-bg border border-forge-border rounded pl-9 pr-3 py-1.5 text-xs font-mono focus:border-forge-accent focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5 mb-3">
                {['attn', 'q_proj', 'v_proj', 'mlp', 'conv', 'ff', 'proj'].map(qf => (
                  <button key={qf} onClick={() => setFilter(qf)}
                    className={`text-xs px-2 py-0.5 rounded font-mono border transition-colors ${
                      filter === qf ? 'bg-forge-accent text-black border-forge-accent'
                      : 'bg-forge-bg border-forge-border text-forge-muted hover:border-forge-accent'
                    }`}>{qf}</button>
                ))}
                {filter && <button onClick={() => setFilter('')} className="text-xs px-2 py-0.5 rounded bg-forge-error/20 text-forge-error border border-forge-error/30">×</button>}
              </div>

              <div className="text-xs text-forge-muted mb-2 flex items-center gap-1">
                <Info className="w-3 h-3" />
                Click any layer to see its structure explained and get pattern suggestions
              </div>

              <div className="max-h-[480px] overflow-y-auto border border-forge-border rounded bg-forge-bg/50">
                {filteredLayers.slice(0, 1000).map((name, idx) => (
                  <button
                    key={`${name}-${idx}`}
                    onClick={() => setSelectedLayer(prev => prev === name ? null : name)}
                    className={`w-full text-left flex items-start gap-2 px-2 py-1.5 transition-colors ${
                      selectedLayer === name
                        ? 'bg-forge-accent/15 border-l-2 border-forge-accent'
                        : 'hover:bg-forge-accent/5 border-l-2 border-transparent'
                    }`}
                  >
                    <ChevronRight className={`w-3 h-3 shrink-0 mt-0.5 transition-colors ${selectedLayer === name ? 'text-forge-accent' : 'text-forge-muted'}`} />
                    <span className={`text-xs font-mono break-all leading-tight ${layerColor(name)}`}>{name}</span>
                  </button>
                ))}
                {filteredLayers.length > 1000 && (
                  <div className="px-3 py-2 text-xs text-forge-muted">Showing 1,000 of {filteredLayers.length.toLocaleString()} — filter to narrow</div>
                )}
              </div>
            </div>

            {/* annotation panel */}
            <div>
              {selectedLayer ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-medium">Layer breakdown</h2>
                    <button onClick={() => setSelectedLayer(null)} className="text-xs text-forge-muted hover:text-forge-text flex items-center gap-1"><X className="w-3 h-3" /> Close</button>
                  </div>
                  <LayerAnnotation
                    name={selectedLayer}
                    allLayers={allLayerNames}
                    onAddPattern={addPattern}
                  />
                </div>
              ) : (
                <div className="h-full min-h-48 flex flex-col items-center justify-center text-forge-muted border-2 border-dashed border-forge-border rounded-lg p-8 text-sm text-center">
                  <ChevronRight className="w-8 h-8 mb-3 opacity-30" />
                  <div className="font-medium mb-1">Select a layer</div>
                  <div className="text-xs opacity-70">Click any layer in the browser to see a breakdown of its segments and get regex patterns you can add directly to your config</div>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Pattern Workbench — always visible */}
      <div ref={workbenchRef}>
        <PatternWorkbench
          layers={allLayerNames}
          patterns={patterns}
          onPatternsChange={setPatterns}
        />
      </div>
    </div>
  );
}

/**
 * Utility: call this from anywhere to push patterns into a named PatternField.
 *   injectPatterns('Default Target Patterns', ['pattern1', 'pattern2'])
 *   injectPatterns('Target Patterns', ['pattern1'])  // component-level field
 */
export function injectPatterns(fieldLabel, patterns) {
  document.dispatchEvent(new CustomEvent('forge:inject-patterns', {
    detail: { field: fieldLabel, patterns }
  }));
}
