import React, { useEffect, useRef, useState } from 'react';
import { Terminal, Minimize2, Maximize2 } from 'lucide-react';
import { getTrainingLogHistory } from '../../utils/api';

export default function TrainingTerminal({ isTraining }) {
  const [lines, setLines]           = useState([]);
  const [connected, setConnected]   = useState(false);
  const [minimized, setMinimized]   = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const containerRef  = useRef(null);
  const esRef         = useRef(null);
  const autoScrollRef = useRef(true);  // ref so scroll handler never goes stale

  const pw = sessionStorage.getItem('forge_pw') || '';
  const pwParam = pw ? `&password=${encodeURIComponent(pw)}` : '';

  // ── Scroll helpers ──────────────────────────────────────────
  const scrollToBottom = () => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = dist < 60;
    autoScrollRef.current = atBottom;
    setShowScrollBtn(!atBottom);
  };

  // Scroll when new lines arrive — only if already at bottom
  useEffect(() => {
    if (autoScrollRef.current) scrollToBottom();
  }, [lines]);

  // ── Colorizer ───────────────────────────────────────────────
  const colorize = (line) => {
    if (/error|exception|traceback/i.test(line))          return 'text-red-400';
    if (/warning|warn/i.test(line))                        return 'text-yellow-400';
    if (/\[lora\]|\[checkpoint\]|\[sampler\]/i.test(line)) return 'text-blue-400';
    if (/loss=|step \d+|epoch \d+/i.test(line))            return 'text-green-400';
    return 'text-forge-muted/80';
  };

  // ── Single SSE connection ────────────────────────────────────
  useEffect(() => {
    let es;

    // Load history first, then open SSE from that offset
      getTrainingLogHistory()
      .then(d => {
        const history = d.lines || [];
        setLines(history);

        es = new EventSource(`/api/training/logs/stream?since=${history.length}${pwParam}`);
        esRef.current = es;
        es.onopen    = () => setConnected(true);
        es.onerror   = () => setConnected(false);
        es.onmessage = (e) => {
          if (!e.data || e.data.startsWith(':')) return;
          setLines(prev => {
            const next = [...prev, e.data];
            return next.length > 1000 ? next.slice(-1000) : next;
          });
        };
      })
      .catch(() => {
        // History unavailable — connect from 0i
        es = new EventSource(`/api/training/logs/stream?since=0${pwParam}`);
        esRef.current = es;
        es.onopen    = () => setConnected(true);
        es.onerror   = () => setConnected(false);
        es.onmessage = (e) => {
          if (!e.data || e.data.startsWith(':')) return;
          setLines(prev => [...prev, e.data].slice(-1000));
        };
      });

    return () => {
      es?.close();
      esRef.current = null;
      setConnected(false);
    };
  }, []); // mount once — never reconnect

  // ── Render ───────────────────────────────────────────────────
  return (
    <div className="border border-forge-border rounded-lg overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-forge-surface border-b border-forge-border">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-forge-muted" />
          <span className="text-sm font-medium">Console</span>
          <span className={`w-1.5 h-1.5 rounded-full transition-colors ${
            connected ? 'bg-forge-success' : 'bg-forge-muted'
          }`} />
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setLines([])}
            className="text-xs text-forge-muted hover:text-forge-text"
          >
            Clear
          </button>
          <button
            onClick={() => setMinimized(m => !m)}
            className="text-forge-muted hover:text-forge-text"
          >
            {minimized
              ? <Maximize2 className="w-3.5 h-3.5" />
              : <Minimize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Body + scroll button wrapper — position:relative so button anchors here */}
      {!minimized && (
        <div className="relative">
          <div
            ref={containerRef}
            onScroll={handleScroll}
            className="h-72 overflow-y-auto bg-[#0d1117] p-3 font-mono text-xs leading-5"
          >
            {lines.length === 0 && (
              <span className="text-forge-muted/40">Waiting for output…</span>
            )}
            {lines.map((line, i) => (
              <div key={i} className={colorize(line)}>
                {line.trimEnd() || '\u00A0'}
              </div>
            ))}
          </div>

          {/* Scroll-to-bottom button — only when user has scrolled up */}
          {showScrollBtn && (
            <button
              onClick={() => {
                autoScrollRef.current = true;
                setShowScrollBtn(false);
                scrollToBottom();
              }}
              className="absolute bottom-2 right-3 text-xs px-2 py-1
                         bg-forge-surface border border-forge-border rounded
                         text-forge-accent hover:bg-forge-accent/10 transition-colors"
            >
              ↓ bottom
            </button>
          )}
        </div>
      )}
    </div>
  );
}
