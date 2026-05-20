import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { X, Square, Save, Undo, Redo, MousePointer, Trash2, Pentagon, Circle, Brush } from 'lucide-react';
import { serializeLabels, parseLabels, rasterizePolygon, rasterizeEllipse, rleToMask, maskToRLE } from '../../utils/labelFormat';

// ── Constants ─────────────────────────────────────────────────────────────────

const TOOLS = {
  SELECT:  'select',   // click = select, drag = move (merged)
  BBOX:    'bbox',
  POLYGON: 'polygon',
  ELLIPSE: 'ellipse',
  PAINT:   'paint',
  ERASE:   'erase',
};

const CLASS_PALETTE = [
  '#ef4444','#3b82f6','#22c55e','#f59e0b','#a855f7',
  '#ec4899','#14b8a6','#f97316','#6366f1','#84cc16',
];

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return [r,g,b];
}
function colorForClass(name, alpha=1) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash<<5)-hash);
  const [r,g,b] = hexToRgb(CLASS_PALETTE[Math.abs(hash) % CLASS_PALETTE.length]);
  return `rgba(${r},${g},${b},${alpha})`;
}
function colorHexForClass(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash<<5)-hash);
  return CLASS_PALETTE[Math.abs(hash) % CLASS_PALETTE.length];
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function AnnotationTool({ entry, datasetId, dsType, onClose, onSaved }) {
  const canvasRef    = useRef(null);
  const overlayRef   = useRef(null); // for paint strokes
  const containerRef = useRef(null);

  const [imgEl,   setImgEl]   = useState(null);
  const [scale,   setScale]   = useState(1);
  const [offset,  setOffset]  = useState({ x: 0, y: 0 });
  const [loading, setLoading] = useState(true);
  const [saving,  setSaving]  = useState(false);

  // ── Tool state
  const isSegmentation = dsType === 'segmentation';
  const [tool,      setTool]      = useState(isSegmentation ? TOOLS.PAINT : TOOLS.BBOX);
  const [className, setClassName] = useState('object');
  const [brushSize, setBrushSize] = useState(20);

  // ── Annotation history (undo/redo)
  const [history, setHistory]   = useState([[]]); // array of annotation snapshots
  const [histIdx, setHistIdx]   = useState(0);

  const rawHistory = history[histIdx];
  const annotations = Array.isArray(rawHistory) ? rawHistory : [];

  // ── In-progress drawing state
  const [drawing,  setDrawing]  = useState(null);  // { tool, ... }
  const [selected, setSelected] = useState(null);  // annotation id

  // ── Paint canvas overlay (Uint8Array per class+name — merged on commit)
  const paintRef = useRef(null); // { mask: Uint8Array, className, dirty }
  const paintCtxRef = useRef(null);

  const paintHistoryRef = useRef([]); // stack of Uint8Array snapshots
  const paintHistoryIdxRef = useRef(-1);

  // ── Load image ───────────────────────────────────────────────────────────────

  useEffect(() => {
    const img = new Image();
    img.onload = () => { setImgEl(img); setLoading(false); };
    img.src = `/api/datasets/${datasetId}/image/by-filename/${encodeURIComponent(entry.filename)}`;
  }, [entry.filename, datasetId]);

  // ── Load existing labels ──────────────────────────────────────────────────────

  useEffect(() => {
    async function load() {
      try {
        const r = await fetch(`/api/datasets/${datasetId}/labels/${entry.index}`);
        if (r.ok) {
          const text = await r.text();
          const parsed = parseLabels(text, entry.width, entry.height);
          // Convert to internal format
          const anns = parsed.map(lb => {
            if (lb.type === 'bbox') return { ...lb, id: crypto.randomUUID() };
            if (lb.type === 'mask') return { ...lb, id: crypto.randomUUID() };
            return null;
          }).filter(Boolean);
          setHistory([anns]);
          setHistIdx(0);
        }
      } catch {}
    }
    load();
  }, [entry.index, datasetId]);

  // ── Compute scale/offset ──────────────────────────────────────────────────────

  useEffect(() => {
    if (!imgEl || !containerRef.current) return;
    const el  = containerRef.current;
    const maxW = el.clientWidth  - 32;
    const maxH = window.innerHeight * 0.68;
    const s = Math.min(maxW / imgEl.width, maxH / imgEl.height, 1);
    setScale(s);
    setOffset({ x: (maxW - imgEl.width * s) / 2, y: 0 });
  }, [imgEl]);

  // ── Render ────────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!imgEl || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');
    const W = Math.round(imgEl.width * scale + offset.x * 2);
    const H = Math.round(imgEl.height * scale + 8);
    canvas.width  = W;
    canvas.height = H;

    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(imgEl, offset.x, 0, imgEl.width * scale, imgEl.height * scale);

    // Draw saved annotations
    annotations.forEach(ann => {
      const isSel = ann.id === selected;
      ctx.save();
      ctx.lineWidth = isSel ? 3 : 1.5;
      if (ann.type === 'bbox') {
        const x = ann.x1 * scale + offset.x;
        const y = ann.y1 * scale;
        const w = (ann.x2 - ann.x1) * scale;
        const h = (ann.y2 - ann.y1) * scale;
        ctx.strokeStyle = colorForClass(ann.className, isSel ? 1 : 0.85);
        ctx.fillStyle   = colorForClass(ann.className, 0.15);
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = colorForClass(ann.className, 1);
        ctx.font = '11px monospace';
        ctx.fillText(ann.className, x + 3, y - 3);
      } else if (ann.type === 'mask') {
        renderMaskAnn(ctx, ann, scale, offset, imgEl.width, imgEl.height, isSel);
      }
      ctx.restore();
    });

    // In-progress drawing
    if (drawing) {
      ctx.save();
      ctx.strokeStyle = colorForClass(className, 0.9);
      ctx.fillStyle   = colorForClass(className, 0.2);
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([5,4]);
      if (drawing.tool === TOOLS.BBOX) {
        const x = Math.min(drawing.x0, drawing.x1) * scale + offset.x;
        const y = Math.min(drawing.y0, drawing.y1) * scale;
        const w = Math.abs(drawing.x1 - drawing.x0) * scale;
        const h = Math.abs(drawing.y1 - drawing.y0) * scale;
        ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
      } else if (drawing.tool === TOOLS.POLYGON && drawing.points.length > 0) {
        ctx.beginPath();
        const [p0,...rest] = drawing.points;
        ctx.moveTo(p0.x*scale+offset.x, p0.y*scale);
        rest.forEach(p => ctx.lineTo(p.x*scale+offset.x, p.y*scale));
        if (drawing.preview) ctx.lineTo(drawing.preview.x*scale+offset.x, drawing.preview.y*scale);
        ctx.stroke();
        // dots
        drawing.points.forEach(p => {
          ctx.beginPath();
          ctx.arc(p.x*scale+offset.x, p.y*scale, 4, 0, Math.PI*2);
          ctx.fillStyle = colorForClass(className,1);
          ctx.setLineDash([]);
          ctx.fill();
          ctx.setLineDash([5,4]);
        });
      } else if (drawing.tool === TOOLS.ELLIPSE) {
        const cx = (drawing.x0+drawing.x1)/2*scale+offset.x;
        const cy = (drawing.y0+drawing.y1)/2*scale;
        const rx = Math.abs(drawing.x1-drawing.x0)/2*scale;
        const ry = Math.abs(drawing.y1-drawing.y0)/2*scale;
        ctx.beginPath();
        ctx.ellipse(cx,cy,Math.max(1,rx),Math.max(1,ry),0,0,Math.PI*2);
        ctx.fill(); ctx.stroke();
      }
      ctx.restore();
    }

    // Also sync overlay canvas size
    if (overlayRef.current) {
      overlayRef.current.width  = W;
      overlayRef.current.height = H;
      if (paintRef.current) redrawPaintOverlay();
    }
  }, [imgEl, scale, offset, annotations, drawing, selected, className]);
  // ── Paint overlay ─────────────────────────────────────────────────────────────

  function initPaint() {
    if (!imgEl) return;
    if (!paintRef.current || paintRef.current.className !== className) {
      paintRef.current = { mask: new Uint8Array(imgEl.width * imgEl.height), className };
      paintHistoryRef.current = [];
      paintHistoryIdxRef.current = -1;
    }
  }

  function snapshotPaint() {
    if (!paintRef.current) return;
    const copy = paintRef.current.mask.slice();
    // Truncate forward history
    paintHistoryRef.current = paintHistoryRef.current.slice(0, paintHistoryIdxRef.current + 1);
    paintHistoryRef.current.push(copy);
    paintHistoryIdxRef.current = paintHistoryRef.current.length - 1;
  }

  function redrawPaintOverlay() {
    if (!overlayRef.current || !paintRef.current || !imgEl) return;
    const ctx = overlayRef.current.getContext('2d');
    ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    const { mask } = paintRef.current;
    const W = imgEl.width, H = imgEl.height;
    const imgData = ctx.createImageData(W, H);
    const [r,g,b] = hexToRgb(colorHexForClass(className));
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) {
        imgData.data[i*4]   = r;
        imgData.data[i*4+1] = g;
        imgData.data[i*4+2] = b;
        imgData.data[i*4+3] = 140;
      }
    }
    // Put scaled
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = W; tmpCanvas.height = H;
    tmpCanvas.getContext('2d').putImageData(imgData, 0, 0);
    ctx.drawImage(tmpCanvas, offset.x, 0, W*scale, H*scale);
  }

  function paintAt(imgX, imgY, erase=false) {
    if (!paintRef.current || !imgEl) return;
    const { mask } = paintRef.current;
    const W = imgEl.width, H = imgEl.height;
    const r = brushSize / 2;
    const x0 = Math.max(0, Math.floor(imgX - r));
    const x1 = Math.min(W-1, Math.ceil(imgX + r));
    const y0 = Math.max(0, Math.floor(imgY - r));
    const y1 = Math.min(H-1, Math.ceil(imgY + r));
    for (let y=y0; y<=y1; y++) {
      for (let x=x0; x<=x1; x++) {
        const dx=x-imgX, dy=y-imgY;
        if (dx*dx+dy*dy <= r*r) mask[y*W+x] = erase ? 0 : 1;
      }
    }
    redrawPaintOverlay();
  }

  function commitPaint() {
    if (!paintRef.current || !imgEl) return;
    const { mask, className: cls } = paintRef.current;
    if (!mask.some(v => v)) { paintRef.current = null; return; }
    const { cropX1, cropY1, cropX2, cropY2, rle } = maskToRLE(mask, imgEl.width, imgEl.height);
    const ann = {
      id: crypto.randomUUID(),
      type: 'mask',
      className: cls,
      cropX1, cropY1, cropX2, cropY2,
      rle,
      _cachedMask: mask.slice(),
    };
    paintRef.current = null;
    paintHistoryRef.current = [];
    paintHistoryIdxRef.current = -1;
    if (overlayRef.current) {
      overlayRef.current.getContext('2d').clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    }
    pushAnnotation(ann);
  }

  // ── Mask rendering helper ─────────────────────────────────────────────────────

  function renderMaskAnn(ctx, ann, s, off, imgW, imgH, isSel) {
    let mask = ann._cachedMask;
    if (!mask) {
      mask = rleToMask(ann.rle, ann.cropX1, ann.cropY1, ann.cropX2, ann.cropY2, imgW, imgH);
      ann._cachedMask = mask;
    }
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = imgW; tmpCanvas.height = imgH;
    const tc = tmpCanvas.getContext('2d');
    const imgData = tc.createImageData(imgW, imgH);
    const [r,g,b] = hexToRgb(colorHexForClass(ann.className));
    const alpha = isSel ? 180 : 120;
    for (let i=0; i<mask.length; i++) {
      if (mask[i]) {
        imgData.data[i*4]=r; imgData.data[i*4+1]=g;
        imgData.data[i*4+2]=b; imgData.data[i*4+3]=alpha;
      }
    }
    tc.putImageData(imgData, 0, 0);
    ctx.drawImage(tmpCanvas, off.x, 0, imgW*s, imgH*s);
    // Outline the crop box when selected
    if (isSel) {
      ctx.strokeStyle = colorForClass(ann.className, 1);
      ctx.setLineDash([4,3]);
      ctx.strokeRect(
        ann.cropX1*s+off.x, ann.cropY1*s,
        (ann.cropX2-ann.cropX1)*s, (ann.cropY2-ann.cropY1)*s
      );
      ctx.setLineDash([]);
    }
    ctx.fillStyle = colorForClass(ann.className, 1);
    ctx.font = '11px monospace';
    ctx.fillText(ann.className, ann.cropX1*s+off.x+3, ann.cropY1*s-3);
  }

  // ── History helpers ────────────────────────────────────────────────────────────

  function pushAnnotation(ann) {
    setHistory(h => {
      const base = h[histIdx];
      const next = [...base, ann];
      return [...h.slice(0, histIdx+1), next];
    });
    setHistIdx(i => i+1);
  }

  function pushAnnotations(anns) {
    setHistory(h => {
      const base = h[histIdx];
      const next = [...base, ...anns];
      return [...h.slice(0, histIdx+1), next];
    });
    setHistIdx(i => i+1);
  }

  function replaceAnnotations(anns) {
    let nextIdx;
    setHistory(h => {
      const sliced = [...h.slice(0, histIdx + 1), anns];
      nextIdx = sliced.length - 1;
      return sliced;
    });
    setHistIdx(() => nextIdx);
  }

  function undo() {
    // If actively painting, undo within paint history first
    if (paintRef.current && paintHistoryIdxRef.current > 0) {
      paintHistoryIdxRef.current -= 1;
      paintRef.current.mask = paintHistoryRef.current[paintHistoryIdxRef.current].slice();
      redrawPaintOverlay();
      return;
    }
    // If at first paint snapshot, clear paint entirely
    if (paintRef.current && paintHistoryIdxRef.current === 0) {
      paintRef.current = null;
      paintHistoryRef.current = [];
      paintHistoryIdxRef.current = -1;
      if (overlayRef.current) {
        overlayRef.current.getContext('2d').clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
      }
      return;
    }
    setHistIdx(i => Math.max(0, i - 1));
    setSelected(null);
  }
  
  function redo() {
    if (paintRef.current && paintHistoryIdxRef.current < paintHistoryRef.current.length - 1) {
      paintHistoryIdxRef.current += 1;
      paintRef.current.mask = paintHistoryRef.current[paintHistoryIdxRef.current].slice();
      redrawPaintOverlay();
      return;
    }
    setHistIdx(i => {
      setHistory(h => { return h; }); // read latest
      return Math.min(history.length - 1, i + 1);
    });
  }

  // ── Coordinate helpers ─────────────────────────────────────────────────────────

  function canvasToImg(e) {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width  / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top)  * scaleY;
    return {
      x: Math.round((cx - offset.x) / scale),
      y: Math.round(cy / scale),
    };
  }

  // ── Mouse handlers ─────────────────────────────────────────────────────────────

  const dragRef = useRef(null); // for move

  function handleMouseDown(e) {
    if (e.button !== 0) return;
    const pt = canvasToImg(e);

    if (tool === TOOLS.SELECT) {
      const hit = [...annotations].reverse().find(ann => hitTest(pt, ann));
      setSelected(hit?.id ?? null);
      // If something was hit, set up for potential drag
      if (hit) {
        dragRef.current = { annId: hit.id, startPt: pt, origAnn: { ...hit }, moved: false };
      } else {
        dragRef.current = null;
      }
      return;
    }

    if ((tool === TOOLS.PAINT || tool === TOOLS.ERASE) && !paintRef.current) {
      const hit = [...annotations].reverse().find(ann => ann.type === 'mask' && hitTest(pt, ann));
      if (hit) {
        let mask = hit._cachedMask;
        if (!mask) mask = rleToMask(hit.rle, hit.cropX1, hit.cropY1, hit.cropX2, hit.cropY2, imgEl.width, imgEl.height);
        paintRef.current = { mask: mask.slice(), className: hit.className };
        replaceAnnotations(annotations.filter(a => a.id !== hit.id));
      } else {
        initPaint();
      }
      paintAt(pt.x, pt.y, tool === TOOLS.ERASE);
      dragRef.current = { painting: true };
      return;
    }

    if (tool === TOOLS.BBOX) {
      setDrawing({ tool: TOOLS.BBOX, x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y });
      return;
    }

    if (tool === TOOLS.ELLIPSE) {
      setDrawing({ tool: TOOLS.ELLIPSE, x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y });
      return;
    }

    if (tool === TOOLS.POLYGON) {
      if (!drawing) {
        setDrawing({ tool: TOOLS.POLYGON, points: [pt] });
      } else {
        const first = drawing.points[0];
        const dx = (pt.x - first.x) * scale, dy = (pt.y - first.y) * scale;
        if (drawing.points.length >= 3 && Math.sqrt(dx*dx+dy*dy) < 12) {
          commitPolygon(drawing.points);
        } else {
          setDrawing(d => ({ ...d, points: [...d.points, pt] }));
        }
      }
      return;
    }
    if (tool === TOOLS.PAINT || tool === TOOLS.ERASE) {
      // For erase, only paint if there's already a mask OR an existing annotation to edit
      if (tool === TOOLS.ERASE && !paintRef.current) {
        // Load selected mask annotation into paintRef so we can erase from it
        if (selected) {
          const ann = annotations.find(a => a.id === selected && a.type === 'mask');
          if (ann) {
            let mask = ann._cachedMask;
            if (!mask) mask = rleToMask(ann.rle, ann.cropX1, ann.cropY1, ann.cropX2, ann.cropY2, imgEl.width, imgEl.height);
            paintRef.current = { mask: mask.slice(), className: ann.className };
            // Remove the annotation from history — it's now being edited
            replaceAnnotations(annotations.filter(a => a.id !== selected));
            setSelected(null);
          }
        }
        // If nothing to erase into, just init empty (erase on empty = no-op visually)
        if (!paintRef.current) initPaint();
      } else {
        initPaint();
      }
      paintAt(pt.x, pt.y, tool === TOOLS.ERASE);
      dragRef.current = { painting: true };
      return;
    }
  }

  function handleMouseMove(e) {
    const pt = canvasToImg(e);

    if (dragRef.current?.annId) {
      // Capture all needed values from ref BEFORE entering setState callback
      const annId   = dragRef.current.annId;
      const startPt = dragRef.current.startPt;
      const orig    = dragRef.current.origAnn;
      const dx = pt.x - startPt.x;
      const dy = pt.y - startPt.y;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
        dragRef.current.moved = true;
      }
      setHistory(h => {
        const anns = h[histIdx].map(a => {
          if (a.id !== annId) return a;
          if (a.type === 'bbox') return { ...a,
            x1: orig.x1+dx, y1: orig.y1+dy,
            x2: orig.x2+dx, y2: orig.y2+dy };
          if (a.type === 'mask') return { ...a,
            cropX1: orig.cropX1+dx, cropY1: orig.cropY1+dy,
            cropX2: orig.cropX2+dx, cropY2: orig.cropY2+dy,
            _cachedMask: null };
          return a;
        });
        const next = [...h];
        next[histIdx] = anns;
        return next;
      });
      return;
    }

    if ((tool === TOOLS.PAINT || tool === TOOLS.ERASE) && dragRef.current?.painting) {
      paintAt(pt.x, pt.y, tool === TOOLS.ERASE);
      return;
    }

    if (drawing) {
      if (drawing.tool === TOOLS.BBOX || drawing.tool === TOOLS.ELLIPSE)
        setDrawing(d => ({ ...d, x1: pt.x, y1: pt.y }));
      else if (drawing.tool === TOOLS.POLYGON)
        setDrawing(d => ({ ...d, preview: pt }));
    }
  }

  function handleMouseUp(e) {
    if (dragRef.current?.annId) {
      if (dragRef.current.moved) {
        setHistory(h => [...h.slice(0, histIdx + 1), h[histIdx].map(a => ({ ...a }))]);
        setHistIdx(i => i + 1);
      }
      dragRef.current = null;
      return;
    }

    if ((tool === TOOLS.PAINT || tool === TOOLS.ERASE) && dragRef.current?.painting) {
      dragRef.current = null;
      snapshotPaint(); // snapshot after each stroke for undo
      return;
    }

    if (!drawing) return;

    if (drawing.tool === TOOLS.BBOX) {
      const x1 = Math.min(drawing.x0, drawing.x1);
      const y1 = Math.min(drawing.y0, drawing.y1);
      const x2 = Math.max(drawing.x0, drawing.x1);
      const y2 = Math.max(drawing.y0, drawing.y1);
      if (x2-x1 > 4 && y2-y1 > 4) {
        pushAnnotation({ id: crypto.randomUUID(), type:'bbox', className, x1,y1,x2,y2 });
      }
      setDrawing(null);
    }

    if (drawing.tool === TOOLS.ELLIPSE) {
      const x0=drawing.x0,y0=drawing.y0,x1=drawing.x1,y1=drawing.y1;
      if (Math.abs(x1-x0)>4 && Math.abs(y1-y0)>4 && imgEl) {
        const cx=(x0+x1)/2, cy=(y0+y1)/2;
        const rx=Math.abs(x1-x0)/2, ry=Math.abs(y1-y0)/2;
        const mask = rasterizeEllipse(cx,cy,rx,ry, imgEl.width, imgEl.height);
        const rleData = maskToRLE(mask, imgEl.width, imgEl.height);
        pushAnnotation({ id:crypto.randomUUID(), type:'mask', className, ...rleData, _cachedMask:mask });
      }
      setDrawing(null);
    }
  }

  function handleDoubleClick(e) {
    if (tool === TOOLS.POLYGON && drawing?.points.length >= 3) {
      commitPolygon(drawing.points);
    }
  }

  function commitPolygon(points) {
    if (!imgEl || points.length < 3) return;
    const mask = rasterizePolygon(points, imgEl.width, imgEl.height);
    const rleData = maskToRLE(mask, imgEl.width, imgEl.height);
    pushAnnotation({ id:crypto.randomUUID(), type:'mask', className, ...rleData, _cachedMask:mask });
    setDrawing(null);
  }

  function hitTest(pt, ann) {
    if (ann.type === 'bbox')
      return pt.x>=ann.x1 && pt.x<=ann.x2 && pt.y>=ann.y1 && pt.y<=ann.y2;
    if (ann.type === 'mask')
      return pt.x>=ann.cropX1 && pt.x<=ann.cropX2 && pt.y>=ann.cropY1 && pt.y<=ann.cropY2;
    return false;
  }

  // ── Keyboard shortcuts ─────────────────────────────────────────────────────────

  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === 'INPUT') return;
      if ((e.ctrlKey||e.metaKey) && e.key==='z') { e.preventDefault(); undo(); }
      if ((e.ctrlKey||e.metaKey) && (e.key==='y'||(e.shiftKey&&e.key==='z'))) { e.preventDefault(); redo(); }
      if (e.key==='Escape') { setDrawing(null); setSelected(null); commitPaintIfAny(); }
      if (e.key==='Delete'||e.key==='Backspace') deleteSelected();
      // Tool shortcuts
      if (e.key==='b') setTool(TOOLS.BBOX);
      if (e.key==='p') setTool(TOOLS.POLYGON);
      if (e.key==='o') setTool(TOOLS.ELLIPSE);
      if (e.key==='m') setTool(TOOLS.PAINT);
      if (e.key==='e') setTool(TOOLS.ERASE);
      if (e.key==='s') setTool(TOOLS.SELECT);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [histIdx, history, selected, drawing, tool]);

  function commitPaintIfAny() {
    if (paintRef.current?.mask?.some(v=>v)) commitPaint();
  }

  function deleteSelected() {
    if (!selected) return;
    replaceAnnotations(annotations.filter(a => a.id !== selected));
    setSelected(null);
  }

  // ── Save ──────────────────────────────────────────────────────────────────────
  async function handleSave() {
    commitPaintIfAny();
    setSaving(true);
    try {
      const text = serializeLabels(annotations, entry.width, entry.height);
      await fetch(`/api/datasets/${datasetId}/labels/${entry.index}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain' },
        body: text,
      });
      onSaved?.(entry.filename); // ← pass filename
    } catch (err) { console.error('Save failed', err); }
    setSaving(false);
  }

  // ── Cursor ─────────────────────────────────────────────────────────────────────

  const cursor = {
    [TOOLS.SELECT]:  'default',
    [TOOLS.BBOX]:    'crosshair',
    [TOOLS.POLYGON]: 'crosshair',
    [TOOLS.ELLIPSE]: 'crosshair',
    [TOOLS.PAINT]:   'none',
    [TOOLS.ERASE]:   'none',
  }[tool];

  const canUndo = histIdx > 0;
  const canRedo = histIdx < history.length - 1;
  const hasPaint = paintRef.current?.mask?.some(v=>v);

  // ── Tool definitions ──────────────────────────────────────────────────────────

  const allTools = [
    { id: TOOLS.SELECT,  icon: <MousePointer className="w-3.5 h-3.5"/>, label:'Select/Move (S)', show: true },
    { id: TOOLS.BBOX,    icon: <Square className="w-3.5 h-3.5"/>,       label:'Box (B)',          show: true },
    { id: TOOLS.POLYGON, icon: <Pentagon className="w-3.5 h-3.5"/>,     label:'Polygon (P)',      show: isSegmentation },
    { id: TOOLS.ELLIPSE, icon: <Circle className="w-3.5 h-3.5"/>,       label:'Ellipse (O)',      show: isSegmentation },
    { id: TOOLS.PAINT,   icon: <Brush className="w-3.5 h-3.5"/>,        label:'Paint (M)',        show: isSegmentation },
    { id: TOOLS.ERASE,   icon: <span className="text-[10px] font-bold leading-none">ER</span>, label:'Erase (E)', show: isSegmentation },
  ];

  // ── Render ─────────────────────────────────────────────────────────────────────

  return (
    <div className="fixed inset-0 bg-black/85 z-50 flex items-center justify-center p-4">
      <div className="bg-forge-surface border border-forge-border rounded-xl flex flex-col overflow-hidden shadow-2xl"
        style={{ width:'min(97vw,1200px)', maxHeight:'97vh' }}>

        {/* ── Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-forge-border shrink-0">
          <span className="text-sm font-medium font-mono truncate text-forge-text">{entry.filename}</span>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-forge-muted uppercase tracking-wide px-2 py-1 bg-forge-bg rounded border border-forge-border">
              {dsType}
            </span>
            <button onClick={handleSave} disabled={saving}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-forge-accent text-black rounded text-xs font-semibold disabled:opacity-40 hover:bg-forge-accent/90 transition-colors">
              <Save className="w-3 h-3"/>{saving ? 'Saving…' : 'Save Labels'}
            </button>
            <button onClick={onClose} className="text-forge-muted hover:text-forge-text transition-colors ml-1">
              <X className="w-5 h-5"/>
            </button>
          </div>
        </div>

        {/* ── Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-forge-border bg-forge-bg/40 shrink-0 flex-wrap">

          {/* Tools */}
          <div className="flex gap-1 flex-wrap">
            {allTools.filter(t=>t.show).map(t => (
              <button key={t.id}
                onClick={() => { setTool(t.id); setDrawing(null); commitPaintIfAny(); }}
                title={t.label}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs border transition-colors ${
                  tool===t.id
                    ? 'bg-forge-accent/15 border-forge-accent/50 text-forge-accent'
                    : 'border-forge-border text-forge-muted hover:text-forge-text hover:border-forge-accent/30'
                }`}>
                {t.icon}
                <span className="hidden sm:inline">{t.label}</span>
              </button>
            ))}
          </div>

          <div className="h-4 w-px bg-forge-border"/>

          {/* Undo / Redo */}
          <div className="flex gap-1">
            <button onClick={undo} disabled={!canUndo} title="Undo (Ctrl+Z)"
              className="p-1.5 rounded border border-forge-border text-forge-muted hover:text-forge-text disabled:opacity-30 transition-colors">
              <Undo className="w-3.5 h-3.5"/>
            </button>
            <button onClick={redo} disabled={!canRedo} title="Redo (Ctrl+Y)"
              className="p-1.5 rounded border border-forge-border text-forge-muted hover:text-forge-text disabled:opacity-30 transition-colors">
              <Redo className="w-3.5 h-3.5"/>
            </button>
          </div>

          <div className="h-4 w-px bg-forge-border"/>

          {/* Class name */}
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-forge-muted uppercase tracking-wide">Class</span>
            <input value={className} onChange={e=>setClassName(e.target.value)}
              onBlur={e=>{ if(!e.target.value.trim()) setClassName('object'); }}
              className="w-28 bg-forge-bg border border-forge-border rounded px-2 py-1 text-xs font-mono focus:border-forge-accent focus:outline-none"
              placeholder="object"/>
            <div className="w-3.5 h-3.5 rounded-full border border-white/10" style={{background:colorHexForClass(className)}}/>
          </div>

          {/* Brush size (paint/erase only) */}
          {(tool===TOOLS.PAINT||tool===TOOLS.ERASE) && (
            <>
              <div className="h-4 w-px bg-forge-border"/>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-forge-muted uppercase tracking-wide">Brush</span>
                <input type="range" min={4} max={80} value={brushSize}
                  onChange={e=>setBrushSize(Number(e.target.value))}
                  className="w-20 accent-forge-accent"/>
                <span className="text-[10px] text-forge-muted w-5">{brushSize}</span>
              </div>
              <button onClick={commitPaint}
                className="px-2.5 py-1.5 bg-forge-accent/10 border border-forge-accent/30 text-forge-accent rounded text-xs hover:bg-forge-accent/20 transition-colors">
                Commit Mask
              </button>
            </>
          )}

          {/* Polygon hint */}
          {tool===TOOLS.POLYGON && drawing && (
            <span className="text-[10px] text-forge-muted italic">
              {drawing.points.length} pts · click near start or dbl-click to close
            </span>
          )}

          <div className="flex-1"/>

          {/* Delete / Clear */}
          {selected && (
            <button onClick={deleteSelected}
              className="flex items-center gap-1 text-xs text-red-400 hover:text-red-300 px-2 py-1 rounded border border-red-400/20 hover:border-red-400/40 transition-colors">
              <Trash2 className="w-3 h-3"/> Delete
            </button>
          )}
          {selected && annotations.find(a => a.id === selected && a.type === 'mask') && (
            <button
              onClick={() => {
                const ann = annotations.find(a => a.id === selected && a.type === 'mask');
                if (!ann || !imgEl) return;
                let mask = ann._cachedMask;
                if (!mask) mask = rleToMask(ann.rle, ann.cropX1, ann.cropY1, ann.cropX2, ann.cropY2, imgEl.width, imgEl.height);
                paintRef.current = { mask: mask.slice(), className: ann.className };
                setClassName(ann.className);
                replaceAnnotations(annotations.filter(a => a.id !== selected));
                setSelected(null);
                setTool(TOOLS.PAINT);
                redrawPaintOverlay();
              }}
              className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded border border-blue-400/20 hover:border-blue-400/40 transition-colors">
              <Brush className="w-3 h-3" /> Edit Mask
            </button>
          )}
          <button onClick={()=>replaceAnnotations([])}
            className="text-xs text-forge-muted hover:text-forge-text px-2 py-1 transition-colors">
            Clear All
          </button>
          <span className="text-[10px] text-forge-muted">{annotations.length} label{annotations.length!==1?'s':''}</span>
        </div>

        {/* ── Canvas area */}
        <div ref={containerRef}
          className="flex-1 overflow-auto p-4 bg-forge-bg/20 flex items-start justify-center min-h-0">
          {loading ? (
            <div className="flex items-center justify-center py-20 text-forge-muted text-sm">Loading image…</div>
          ) : (
            <div className="relative" style={{display:'inline-block'}}>
              <canvas ref={canvasRef}
                style={{ cursor, display:'block', imageRendering:'pixelated', maxWidth:'100%' }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onDoubleClick={handleDoubleClick}/>
              {/* Paint overlay */}
              <canvas ref={overlayRef}
                style={{
                  position:'absolute', top:0, left:0,
                  pointerEvents:'none',
                  imageRendering:'pixelated',
                  maxWidth:'100%',
                }}/>
              {/* Brush cursor */}
              {(tool===TOOLS.PAINT||tool===TOOLS.ERASE) && (
                <BrushCursor scale={scale} brushSize={brushSize} canvasRef={canvasRef}
                  color={tool===TOOLS.ERASE?'rgba(239,68,68,0.8)':colorForClass(className,0.8)}/>
              )}
            </div>
          )}
        </div>

        {/* ── Label list */}
        {annotations.length > 0 && (
          <div className="border-t border-forge-border px-4 py-2 flex gap-2 flex-wrap shrink-0 bg-forge-bg/40 max-h-20 overflow-y-auto">
            {annotations.map(ann => (
              <button key={ann.id}
                onClick={() => setSelected(selected===ann.id ? null : ann.id)}
                className={`flex items-center gap-1.5 text-[10px] px-2 py-1 rounded border transition-colors ${
                  selected===ann.id
                    ? 'border-forge-accent bg-forge-accent/10 text-forge-accent'
                    : 'border-forge-border text-forge-muted hover:border-forge-accent/40'
                }`}>
                <div className="w-2.5 h-2.5 rounded-sm" style={{background:colorHexForClass(ann.className)}}/>
                {ann.type==='bbox' ? '▭' : '⬟'} {ann.className}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Brush cursor overlay ───────────────────────────────────────────────────────

function BrushCursor({ scale, brushSize, canvasRef, color }) {
  const [pos, setPos] = useState(null);
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onMove = (e) => {
      const rect = el.getBoundingClientRect();
      setPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };
    const onLeave = () => setPos(null);
    el.addEventListener('mousemove', onMove);
    el.addEventListener('mouseleave', onLeave);
    return () => { el.removeEventListener('mousemove', onMove); el.removeEventListener('mouseleave', onLeave); };
  }, [canvasRef]);

  if (!pos) return null;
  const r = brushSize * scale / 2;
  return (
    <div style={{
      position:'absolute', pointerEvents:'none',
      left: pos.x - r, top: pos.y - r,
      width: r*2, height: r*2,
      borderRadius:'50%',
      border:`2px solid ${color}`,
      boxShadow:`0 0 0 1px rgba(0,0,0,0.4)`,
    }}/>
  );
}

