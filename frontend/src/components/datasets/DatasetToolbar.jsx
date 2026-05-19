import React from 'react';
import { Search, Grid, Edit3, BarChart3 } from 'lucide-react';

export default function DatasetToolbar({
  filterMode, setFilterMode, searchText, setSearchText, onSearch,
  totalEntries, totalImages, selectedCount, allSelected, onSelectAll,
  gridSize, setGridSize, rightPanel, showRightPanel,
  onToggleCaption, onToggleAnalysis, dsType,
}) {
  const isCaption = dsType === 'caption';
  const hasRightPanel = isCaption || dsType === 'classification';

  const filters = isCaption
    ? [[null, `All (${totalImages ?? totalEntries})`], ['uncaptioned', 'No Caption'], ['captioned', 'Captioned']]
    : [[null, `All (${totalImages ?? totalEntries})`]];

  return (
    <div className="flex items-center justify-between gap-3 flex-wrap">
      <div className="flex items-center gap-2 flex-wrap">
        {filters.map(([val, label]) => (
          <button key={String(val)} onClick={() => setFilterMode(val)}
            className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
              filterMode === val
                ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent'
                : 'border-forge-border text-forge-muted hover:text-forge-text'
            }`}>{label}</button>
        ))}
        <button onClick={onSelectAll}
          className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
            allSelected
              ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent'
              : 'border-forge-border text-forge-muted hover:text-forge-text'
          }`}>
          {allSelected ? 'Deselect All' : 'Select All'}
        </button>
      </div>
      <div className="flex items-center gap-2">
        {isCaption && (
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-forge-muted" />
            <input value={searchText} onChange={e => setSearchText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && onSearch()}
              placeholder="Search captions…"
              className="bg-forge-bg border border-forge-border rounded-lg pl-8 pr-2 py-1.5 text-xs w-48 focus:border-forge-accent focus:outline-none" />
          </div>
        )}
        <div className="flex items-center gap-1.5 border border-forge-border rounded px-2 py-1">
          <Grid className="w-3 h-3 text-forge-muted shrink-0" />
          <input type="range" min={2} max={10} value={gridSize}
            onChange={e => setGridSize(Number(e.target.value))}
            className="w-20 accent-forge-accent cursor-pointer" />
          <span className="text-[10px] text-forge-muted w-4 text-center">{gridSize}</span>
        </div>
        {hasRightPanel && <>
          {isCaption && (
            <button onClick={onToggleCaption}
              className={`p-1.5 rounded border transition-colors ${
                showRightPanel && rightPanel === 'caption'
                  ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent'
                  : 'border-forge-border text-forge-muted hover:text-forge-text'
              }`} title="Caption Editor">
              <Edit3 className="w-4 h-4" />
            </button>
          )}
          <button onClick={onToggleAnalysis}
            className={`p-1.5 rounded border transition-colors ${
              showRightPanel && rightPanel === 'analysis'
                ? 'bg-forge-accent/10 border-forge-accent/30 text-forge-accent'
                : 'border-forge-border text-forge-muted hover:text-forge-text'
            }`} title={isCaption ? 'Concept Analysis' : 'Class Distribution'}>
            <BarChart3 className="w-4 h-4" />
          </button>
        </>}
      </div>
    </div>
  );
}
