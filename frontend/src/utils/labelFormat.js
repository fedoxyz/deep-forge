// ─── Forge Label Format ───────────────────────────────────────────────────────
// Detection:    D <class_name> <x1> <y1> <x2> <y2>   (absolute pixels, integers)
// Segmentation: S <class_name> <crop_x1> <crop_y1> <crop_x2> <crop_y2> <RLE...>
//   RLE = interleaved (value count) pairs: "0 120 1 34 0 88"
//   → 120 off-pixels, 34 on-pixels, 88 off-pixels, within the crop box (row-major)
// Lines starting with # are comments/metadata
// ─────────────────────────────────────────────────────────────────────────────

export function serializeLabels(annotations, imgW, imgH) {
  return annotations.map(ann => {
    if (ann.type === 'bbox') {
      const x1 = Math.round(Math.min(ann.x1, ann.x2));
      const y1 = Math.round(Math.min(ann.y1, ann.y2));
      const x2 = Math.round(Math.max(ann.x1, ann.x2));
      const y2 = Math.round(Math.max(ann.y1, ann.y2));
      return `D ${ann.className} ${x1} ${y1} ${x2} ${y2}`;
    }
    if (ann.type === 'mask') {
      const { cropX1, cropY1, cropX2, cropY2, rle } = ann;
      return `S ${ann.className} ${cropX1} ${cropY1} ${cropX2} ${cropY2} ${rle.join(' ')}`;
    }
    if (ann.type === 'polygon') {
      // Rasterize polygon → RLE mask on save
      const mask = rasterizePolygon(ann.points, imgW, imgH);
      const rleData = maskToRLE(mask, imgW, imgH);
      const { cropX1, cropY1, cropX2, cropY2, rle } = rleData;
      return `S ${ann.className} ${cropX1} ${cropY1} ${cropX2} ${cropY2} ${rle.join(' ')}`;
    }
    return null;
  }).filter(Boolean).join('\n');
}

export function parseLabels(text) {
  if (!text?.trim()) return [];
  const annotations = [];
  for (const raw of text.trim().split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    const type = parts[0];

    if (type === 'D' && parts.length >= 6) {
      annotations.push({
        id: crypto.randomUUID(),
        type: 'bbox',
        className: parts[1],
        x1: parseInt(parts[2]),
        y1: parseInt(parts[3]),
        x2: parseInt(parts[4]),
        y2: parseInt(parts[5]),
      });
      continue;
    }

    if (type === 'S' && parts.length >= 7) {
      const [, className, cx1, cy1, cx2, cy2, ...rle] = parts;
      annotations.push({
        id: crypto.randomUUID(),
        type: 'mask',
        className,
        cropX1: parseInt(cx1),
        cropY1: parseInt(cy1),
        cropX2: parseInt(cx2),
        cropY2: parseInt(cy2),
        rle: rle.map(Number),
      });
      continue;
    }
  }
  return annotations;
}

// ── RLE helpers ───────────────────────────────────────────────────────────────

/**
 * Convert a flat Uint8Array mask (row-major) to RLE within its tight bounding box.
 * Returns { cropX1, cropY1, cropX2, cropY2, rle }.
 */
export function maskToRLE(flatMask, imgW, imgH) {
  let minX = imgW, minY = imgH, maxX = -1, maxY = -1;
  for (let y = 0; y < imgH; y++) {
    for (let x = 0; x < imgW; x++) {
      if (flatMask[y * imgW + x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX === -1) return { cropX1: 0, cropY1: 0, cropX2: 0, cropY2: 0, rle: [0, 1] };

  const cropX1 = minX, cropY1 = minY;
  const cropX2 = maxX + 1, cropY2 = maxY + 1;
  const W = cropX2 - cropX1;

  const rle = [];
  let current = 0, count = 0;
  for (let y = cropY1; y < cropY2; y++) {
    for (let x = cropX1; x < cropX2; x++) {
      const val = flatMask[y * imgW + x] ? 1 : 0;
      if (val === current) {
        count++;
      } else {
        rle.push(current, count);
        current = val;
        count = 1;
      }
    }
  }
  rle.push(current, count);
  return { cropX1, cropY1, cropX2, cropY2, rle };
}

/**
 * Reconstruct a full flat Uint8Array mask from RLE + crop box.
 */
export function rleToMask(rle, cropX1, cropY1, cropX2, cropY2, imgW, imgH) {
  const flatMask = new Uint8Array(imgW * imgH);
  const W = cropX2 - cropX1;
  let pixel = 0;
  for (let i = 0; i < rle.length - 1; i += 2) {
    const val = rle[i];
    const count = rle[i + 1];
    for (let k = 0; k < count; k++, pixel++) {
      const lx = pixel % W;
      const ly = Math.floor(pixel / W);
      const gx = cropX1 + lx;
      const gy = cropY1 + ly;
      if (gx >= 0 && gx < imgW && gy >= 0 && gy < imgH) {
        flatMask[gy * imgW + gx] = val;
      }
    }
  }
  return flatMask;
}

// ── Rasterizers ───────────────────────────────────────────────────────────────

/**
 * Scanline-fill rasterize an arbitrary polygon to a flat Uint8Array mask.
 * Handles convex, concave, and donut shapes via even-odd rule.
 */
export function rasterizePolygon(points, imgW, imgH) {
  const mask = new Uint8Array(imgW * imgH);
  if (points.length < 3) return mask;
  for (let y = 0; y < imgH; y++) {
    const intersections = [];
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const a = points[i];
      const b = points[(i + 1) % n];
      if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
        const t = (y - a.y) / (b.y - a.y);
        intersections.push(a.x + t * (b.x - a.x));
      }
    }
    intersections.sort((a, b) => a - b);
    for (let i = 0; i < intersections.length - 1; i += 2) {
      const x0 = Math.max(0, Math.ceil(intersections[i]));
      const x1 = Math.min(imgW - 1, Math.floor(intersections[i + 1]));
      for (let x = x0; x <= x1; x++) mask[y * imgW + x] = 1;
    }
  }
  return mask;
}

/**
 * Rasterize an ellipse to a flat Uint8Array mask.
 */
export function rasterizeEllipse(cx, cy, rx, ry, imgW, imgH) {
  const mask = new Uint8Array(imgW * imgH);
  if (rx < 1 || ry < 1) return mask;
  const x0 = Math.max(0, Math.floor(cx - rx));
  const x1 = Math.min(imgW - 1, Math.ceil(cx + rx));
  const y0 = Math.max(0, Math.floor(cy - ry));
  const y1 = Math.min(imgH - 1, Math.ceil(cy + ry));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      if (dx * dx + dy * dy <= 1) mask[y * imgW + x] = 1;
    }
  }
  return mask;
}
