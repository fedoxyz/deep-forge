import React from 'react';
import { Trash2, MessageSquare } from 'lucide-react';

export default function SelectionBar({ count, onClearSelection, onDeleteSelected, onCaptionSelected, dsType }) {
  if (count === 0) return null;
  return (
    <div className="flex items-center gap-3 px-3 py-2 bg-forge-accent/5 border border-forge-accent/20 rounded-lg">
      <span className="text-xs text-forge-accent font-medium">{count} selected</span>
      <div className="h-3 w-px bg-forge-border" />
      {dsType === 'caption' && (
        <button onClick={onCaptionSelected}
          className="flex items-center gap-1.5 text-xs text-forge-muted hover:text-forge-accent transition-colors">
          <MessageSquare className="w-3 h-3" /> Caption selected
        </button>
      )}
      <button onClick={onDeleteSelected}
        className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 transition-colors">
        <Trash2 className="w-3 h-3" /> Delete selected
      </button>
      <div className="flex-1" />
      <button onClick={onClearSelection} className="text-xs text-forge-muted hover:text-forge-text transition-colors">
        Clear
      </button>
    </div>
  );
}
