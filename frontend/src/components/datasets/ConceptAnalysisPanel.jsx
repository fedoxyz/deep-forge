import React, { useState, useMemo } from 'react';
import {
  Search, 
  BarChart3,
  AlertTriangle, Sparkles,
  Zap,
} from 'lucide-react';
import {
  analyzeDatasetConcepts, extractConceptsLLM
} from '../../utils/api';


export default function ConceptAnalysisPanel({ datasetId, totalImages, onSelectConcept }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeConcept, setActiveConcept] = useState(null);
  const [filterCat, setFilterCat] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [minFreq, setMinFreq] = useState(2);
  const [maxNgram, setMaxNgram] = useState(3);
  const [showSettings, setShowSettings] = useState(false);
  const runAnalysis = async () => {
    setLoading(true);
    try { const result = await analyzeDatasetConcepts(datasetId, { min_frequency: minFreq, max_ngram: maxNgram }); setAnalysis(result); setActiveConcept(null); }
    catch (e) { console.error('Analysis failed:', e); }
    setLoading(false);
  };
  const filteredConcepts = useMemo(() => {
    if (!analysis) return [];
    let concepts = analysis.concepts;
    if (filterCat !== 'all') concepts = concepts.filter(c => c.category === filterCat);
    if (searchTerm) { const lower = searchTerm.toLowerCase(); concepts = concepts.filter(c => c.phrase.includes(lower)); }
    return concepts;
  }, [analysis, filterCat, searchTerm]);
  const maxCount = useMemo(() => Math.max(1, ...filteredConcepts.map(c => c.count)), [filteredConcepts]);
  const handleConceptClick = (concept) => { const next = concept.phrase === activeConcept?.phrase ? null : concept; setActiveConcept(next); if (next) onSelectConcept?.(next); };
  const catDist = analysis?.category_distribution || {};
  const triggerWords = analysis?.trigger_words_detected || [];
  return (
    <div className="h-full flex flex-col">
      <div className="flex flex-col items-left justify-between mb-1.5 gap-1.5">
        <h3 className="text-sm font-medium flex items-center gap-1.5"><Sparkles className="w-4 h-4 text-forge-accent" /> Concepts</h3>
        <div className="flex items-center gap-1.5">
          <button onClick={() => setShowSettings(!showSettings)} className="text-[10px] text-forge-muted hover:text-forge-accent px-2 py-1 rounded">{showSettings ? 'Hide' : 'Settings'}</button>
          <button onClick={runAnalysis} disabled={loading} className="flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent text-black rounded text-xs font-medium disabled:opacity-40">
            {loading ? <Spinner size="sm" /> : <Zap className="w-3 h-3" />} {analysis ? 'Re-run' : 'Analyze'}</button>
            <button
              onClick={async () => {
                setLoading(true);
                try {
                  const result = await extractConceptsLLM(datasetId, ['attributes', 'actions', 'settings', 'style', 'composition']);
                  setAnalysis(prev => ({
                    ...prev || {},
                    concepts: result.concepts.map(c => ({ ...c, image_indices: c.caption_indices || [] })),
                    total_images: result.total_captions,
                    category_distribution: result.concepts.reduce((acc, c) => { acc[c.category] = (acc[c.category] || 0) + 1; return acc; }, {}),
                    trigger_words_detected: [],
                  }));
                } catch (e) { console.error('LLM extraction failed:', e); }
                setLoading(false);
              }}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/10 text-purple-400 border border-purple-500/30 rounded text-xs disabled:opacity-40 hover:bg-purple-500/20 transition-colors"
            >
              <Sparkles className="w-3 h-3" /> LLM Analyze
            </button>
        </div>
      </div>
      {showSettings && <div className="grid grid-cols-2 gap-2 mb-3 p-2.5 bg-forge-surface rounded-lg border border-forge-border">
        <div><label className="text-[10px] text-forge-muted uppercase">Min Frequency</label><input type="number" value={minFreq} min={1} onChange={(e) => setMinFreq(parseInt(e.target.value) || 2)} className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" /></div>
        <div><label className="text-[10px] text-forge-muted uppercase">Max N-gram</label><input type="number" value={maxNgram} min={1} max={5} onChange={(e) => setMaxNgram(parseInt(e.target.value) || 3)} className="w-full bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono mt-0.5 focus:border-forge-accent focus:outline-none" /></div>
      </div>}
      {!analysis && !loading && <EmptyState icon={BarChart3} title="Run analysis to discover concepts" subtitle="Extracts recurring phrases from your captions" />}
      {loading && <div className="flex-1 flex items-center justify-center"><Spinner /></div>}
      {analysis && !loading && <>
        {triggerWords.length > 0 && <div className="mb-2 px-2.5 py-2 bg-purple-500/5 border border-purple-500/15 rounded-lg">
          <p className="text-[10px] text-purple-400 font-medium mb-1">Trigger word(s) detected & filtered:</p>
          <div className="flex flex-wrap gap-1">{triggerWords.map(tw => <span key={tw} className="text-[10px] px-1.5 py-0.5 bg-purple-500/10 text-purple-300 rounded font-mono">{tw}</span>)}</div>
        </div>}
        <div className="flex flex-wrap gap-1 mb-2">
          <button onClick={() => setFilterCat('all')} className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${filterCat === 'all' ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent' : 'border-forge-border text-forge-muted hover:text-forge-text'}`}>All ({analysis.concepts.length})</button>
          {Object.entries(catDist).map(([cat, count]) => <button key={cat} onClick={() => setFilterCat(filterCat === cat ? 'all' : cat)} className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${filterCat === cat ? CATEGORY_COLORS[cat] : 'border-forge-border text-forge-muted hover:text-forge-text'}`}>{cat} ({count})</button>)}
        </div>
        <div className="relative mb-2"><Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-forge-muted" /><input value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} placeholder="Filter concepts..." className="w-full bg-forge-bg border border-forge-border rounded pl-7 pr-2 py-1.5 text-xs focus:border-forge-accent focus:outline-none" /></div>
        <div className="flex-1 overflow-y-auto space-y-0.5 min-h-0">
          {filteredConcepts.map((concept) => <ConceptBar key={concept.phrase} concept={concept} maxCount={maxCount} totalImages={totalImages} onClick={handleConceptClick} isActive={activeConcept?.phrase === concept.phrase} />)}
          {filteredConcepts.length === 0 && <p className="text-xs text-forge-muted py-4 text-center">No concepts match your filters</p>}
        </div>
        {analysis.concepts.length > 0 && <div className="mt-2 pt-2 border-t border-forge-border space-y-1">
          {analysis.concepts.filter(c => c.count / totalImages > 0.6).length > 0 && <div className="flex items-start gap-1.5 text-[10px] text-yellow-400"><AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" /><span>{analysis.concepts.filter(c => c.count / totalImages > 0.6).length} dominant concept(s) at 60%+ — overfitting risk</span></div>}
        </div>}
      </>}
    </div>
  );
}

