// frontend/src/components/datasets/AnnotationTool.jsx
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { X, Square, Pentagon, Trash2, Save, Undo, MousePointer } from 'lucide-react';

const MODES = { SELECT: 'select', BBOX: 'bbox', POLYGON: 'polygon' };
const CLASS_COLORS = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#a855f7','#ec4899'];

function colorForClass(id) { return CLASS_COLORS[id % CLASS_COLORS.length]; }

export default function AnnotationTool({ entry, datasetId, onClose, onSaved }) {
  const canvasRef = useRef(null);
  const [imgEl, setImgEl] = useState(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [mode, setMode] = useState(MODES.BBOX);
  const [annotations, setAnnotations] = useState([]);  // [{type:'bbox'|'polygon', class_id, points:[{x,y}]}]
  const [drawing, setDrawing] = useState(null);  // in-progress shape
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [classId, setClassId] = useState(0);

  // Load existing YOLO labels
  useEffect(() => {
    async function loadLabels() {
      try {
        const r = await fetch(`/api/datasets/${datasetId}/labels/${entry.index}`);
        if (r.ok) {
          const text = await r.text();
          const parsed = parseYOLO(text, entry.width, entry.height);
          setAnnotations(parsed);
        }
      } catch {}
    }
    loadLabels();
  }, [entry.index]);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setImgEl(img);
      setLoading(false);
    };
    img.src = `/api/datasets/${datasetId}/image/by-filename/${encodeURIComponent(entry.filename)}`;
  }, [entry.filename]);

  // Compute scale/offset when image loads or canvas resizes
  useEffect(() => {
    if (!imgEl || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const maxW = canvas.parentElement.clientWidth - 32;
    const maxH = window.innerHeight * 0.72;
    const s = Math.min(maxW / imgEl.width, maxH / imgEl.height, 1);
    setScale(s);
    setOffset({
      x: (maxW - imgEl.width * s) / 2,
      y: 0,
    });
  }, [imgEl]);

  // Render
  useEffect(() => {
    if (!imgEl || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const W = imgEl.width * scale + offset.x * 2;
    const H = imgEl.height * scale + 8;
    canvas.width = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(imgEl, offset.x, 0, imgEl.width * scale, imgEl.height * scale);

    // Draw saved annotations
    annotations.forEach((ann, i) => {
      const color = colorForClass(ann.class_id);
      ctx.strokeStyle = color;
      ctx.fillStyle = color + '33';
      ctx.lineWidth = selected === i ? 3 : 1.5;
      drawAnnotation(ctx, ann, scale, offset);
    });

    // Draw in-progress shape
    if (drawing) {
      ctx.strokeStyle = colorForClass(classId);
      ctx.fillStyle = colorForClass(classId) + '22';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      drawAnnotation(ctx, drawing, scale, offset);
      ctx.setLineDash([]);
    }
  }, [imgEl, scale, offset, annotations, drawing, selected]);

  function drawAnnotation(ctx, ann, s, off) {
    const pts = ann.points;
    if (!pts || pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x * s + off.x, pts[0].y * s);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x * s + off.x, pts[i].y * s);
    }
    if (ann.type === 'polygon' || ann.type === 'bbox') ctx.closePath();
    ctx.fill();
    ctx.stroke();
    // Class label
    ctx.fillStyle = colorForClass(ann.class_id);
    ctx.font = '11px monospace';
    ctx.fillText(`cls:${ann.class_id}`, pts[0].x * s + off.x + 3, pts[0].y * s - 3);
  }

  function canvasToImg(e) {
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = (e.clientX - rect.left) / (rect.width / canvasRef.current.width);
    const cy = (e.clientY - rect.top) / (rect.height / canvasRef.current.height);
    return {
      x: (cx - offset.x) / scale,
      y: cy / scale,
    };
  }

  function handleMouseDown(e) {
    const pt = canvasToImg(e);
    if (mode === MODES.BBOX) {
      setDrawing({ type: 'bbox', class_id: classId, points: [pt, pt] });
    } else if (mode === MODES.POLYGON) {
      if (!drawing) {
        setDrawing({ type: 'polygon', class_id: classId, points: [pt] });
      } else {
        setDrawing(d => ({ ...d, points: [...d.points, pt] }));
      }
    } else if (mode === MODES.SELECT) {
      // Hit test
      const hit = annotations.findLastIndex(ann => hitTest(pt, ann));
      setSelected(hit >= 0 ? hit : null);
    }
  }

  function handleMouseMove(e) {
    if (!drawing) return;
    const pt = canvasToImg(e);
    if (mode === MODES.BBOX) {
      setDrawing(d => ({ ...d, points: [d.points[0], pt] }));
    } else if (mode === MODES.POLYGON) {
      // preview last segment
      setDrawing(d => {
        const pts = [...d.points];
        if (pts.length > 0) {
          const preview = [...pts];
          return { ...d, _preview: pt };
        }
        return d;
      });
    }
  }

  function handleDoubleClick(e) {
    if (mode === MODES.POLYGON && drawing && drawing.points.length >= 3) {
      setAnnotations(a => [...a, { ...drawing, _preview: undefined }]);
      setDrawing(null);
    }
  }

  function handleMouseUp(e) {
    if (mode === MODES.BBOX && drawing) {
      const [p1, p2] = drawing.points;
      if (Math.abs(p1.x - p2.x) > 4 && Math.abs(p1.y - p2.y) > 4) {
        // Normalize to top-left / bottom-right
        const norm = {
          ...drawing,
          points: [
            { x: Math.min(p1.x, p2.x), y: Math.min(p1.y, p2.y) },
            { x: Math.max(p1.x, p2.x), y: Math.min(p1.y, p2.y) },
            { x: Math.max(p1.x, p2.x), y: Math.max(p1.y, p2.y) },
            { x: Math.min(p1.x, p2.x), y: Math.max(p1.y, p2.y) },
          ],
        };
        setAnnotations(a => [...a, norm]);
      }
      setDrawing(null);
    }
  }

  function hitTest(pt, ann) {
    if (!ann.points || ann.points.length < 2) return false;
    const xs = ann.points.map(p => p.x);
    const ys = ann.points.map(p => p.y);
    return pt.x >= Math.min(...xs) && pt.x <= Math.max(...xs) &&
           pt.y >= Math.min(...ys) && pt.y <= Math.max(...ys);
  }

  function deleteSelected() {
    if (selected === null) return;
    setAnnotations(a => a.filter((_, i) => i !== selected));
    setSelected(null);
  }

  async function handleSave() {
    setSaving(true);
    try {
      const yolo = toYOLO(annotations, entry.width, entry.height);
      await fetch(`/api/datasets/${datasetId}/labels/${entry.index}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels: yolo }),
      });
      onSaved();
    } catch (e) { console.error('Save failed', e); }
    setSaving(false);
  }

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-forge-surface border border-forge-border rounded-xl flex flex-col overflow-hidden"
        style={{ width: 'min(95vw, 1100px)', maxHeight: '95vh' }}>
        
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-forge-border shrink-0">
          <span className="text-sm font-medium font-mono truncate">{entry.filename}</span>
          <div className="flex items-center gap-2">
            <button onClick={handleSave} disabled={saving}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent text-black rounded text-xs font-medium disabled:opacity-40">
              <Save className="w-3 h-3" />{saving ? 'Saving…' : 'Save Labels'}
            </button>
            <button onClick={onClose} className="text-forge-muted hover:text-forge-text">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-3 px-4 py-2 border-b border-forge-border bg-forge-bg/50 shrink-0 flex-wrap">
          {/* Mode */}
          <div className="flex gap-1">
            {[
              [MODES.SELECT,  <MousePointer className="w-3.5 h-3.5"/>, 'Select'],
              [MODES.BBOX,    <Square className="w-3.5 h-3.5"/>,       'Bounding Box'],
              [MODES.POLYGON, <Pentagon className="w-3.5 h-3.5"/>,     'Polygon (dbl-click to close)'],
            ].map(([m, icon, label]) => (
              <button key={m} onClick={() => { setMode(m); setDrawing(null); }}
                title={label}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs border transition-colors ${
                  mode === m ? 'bg-forge-accent/10 border-forge-accent/40 text-forge-accent' : 'border-forge-border text-forge-muted hover:text-forge-text'
                }`}>{icon} {label}</button>
            ))}
          </div>
          <div className="h-4 w-px bg-forge-border" />
          {/* Class */}
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-forge-muted uppercase">Class</span>
            <input type="number" min={0} max={99} value={classId}
              onChange={e => setClassId(parseInt(e.target.value) || 0)}
              className="w-14 bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono focus:border-forge-accent focus:outline-none" />
            <div className="w-4 h-4 rounded" style={{ background: colorForClass(classId) }} />
          </div>
          <div className="h-4 w-px bg-forge-border" />
          {/* Actions */}
          {selected !== null && (
            <button onClick={deleteSelected}
              className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 px-2 py-1 rounded border border-red-400/20 hover:border-red-400/40">
              <Trash2 className="w-3 h-3" /> Delete Selected
            </button>
          )}
          <button onClick={() => setAnnotations([])}
            className="text-xs text-forge-muted hover:text-forge-text px-2 py-1">
            Clear All
          </button>
          <span className="ml-auto text-[10px] text-forge-muted">{annotations.length} annotation{annotations.length !== 1 ? 's' : ''}</span>
        </div>

        {/* Canvas */}
        <div className="flex-1 overflow-auto p-4 bg-forge-bg/30 flex items-start justify-center">
          {loading ? (
            <div className="flex items-center justify-center py-20 text-forge-muted text-sm">Loading image…</div>
          ) : (
            <canvas ref={canvasRef}
              className="cursor-crosshair max-w-full"
              style={{ imageRendering: 'pixelated' }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onDoubleClick={handleDoubleClick} />
          )}
        </div>

        {/* Annotation list */}
        {annotations.length > 0 && (
          <div className="border-t border-forge-border px-4 py-2 flex gap-2 flex-wrap shrink-0 bg-forge-bg/50 max-h-24 overflow-y-auto">
            {annotations.map((ann, i) => (
              <button key={i} onClick={() => setSelected(selected === i ? null : i)}
                className={`flex items-center gap-1.5 text-[10px] px-2 py-1 rounded border transition-colors ${
                  selected === i ? 'border-forge-accent bg-forge-accent/10 text-forge-accent' : 'border-forge-border text-forge-muted hover:border-forge-accent/40'
                }`}>
                <div className="w-2.5 h-2.5 rounded-sm" style={{ background: colorForClass(ann.class_id) }} />
                {ann.type} cls:{ann.class_id}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── YOLO serialization helpers ──

function parseYOLO(text, imgW, imgH) {
  if (!text.trim()) return [];
  return text.trim().split('\n').map(line => {
    const parts = line.trim().split(/\s+/).map(Number);
    if (parts.length < 5) return null;
    const class_id = parts[0];
    const coords = parts.slice(1);
    if (coords.length === 4) {
      // bbox: cx cy w h normalized
      const [cx, cy, w, h] = coords;
      const x1 = (cx - w/2) * imgW, y1 = (cy - h/2) * imgH;
      const x2 = (cx + w/2) * imgW, y2 = (cy + h/2) * imgH;
      return { type: 'bbox', class_id, points: [
        {x:x1,y:y1},{x:x2,y:y1},{x:x2,y:y2},{x:x1,y:y2}
      ]};
    } else {
      // polygon: x1 y1 x2 y2 ...
      const points = [];
      for (let i = 0; i < coords.length; i += 2) {
        points.push({ x: coords[i] * imgW, y: coords[i+1] * imgH });
      }
      return { type: 'polygon', class_id, points };
    }
  }).filter(Boolean);
}

function toYOLO(annotations, imgW, imgH) {
  return annotations.map(ann => {
    const { class_id, type, points } = ann;
    if (type === 'bbox') {
      const xs = points.map(p => p.x), ys = points.map(p => p.y);
      const x1 = Math.min(...xs), y1 = Math.min(...ys);
      const x2 = Math.max(...xs), y2 = Math.max(...ys);
      const cx = ((x1+x2)/2/imgW).toFixed(6);
      const cy = ((y1+y2)/2/imgH).toFixed(6);
      const w  = ((x2-x1)/imgW).toFixed(6);
      const h  = ((y2-y1)/imgH).toFixed(6);
      return `${class_id} ${cx} ${cy} ${w} ${h}`;
    } else {
      const coords = points.map(p => `${(p.x/imgW).toFixed(6)} ${(p.y/imgH).toFixed(6)}`).join(' ');
      return `${class_id} ${coords}`;
    }
  }).join('\n');
}
