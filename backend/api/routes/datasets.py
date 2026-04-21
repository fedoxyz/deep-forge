"""
Routes for dataset management, caption editing, and concept analysis.

Extends the existing datasets.py routes (catalog/builtin) with:
- Dataset loading/scanning
- Image browsing with thumbnails
- Caption CRUD
- Concept analysis
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.datasets.image_caption import find_closest_bucket, DEFAULT_BUCKETS
from pathlib import Path
from PIL import Image
import os, shutil
from typing import List

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


# ── Existing routes (keep these) ──

@router.get("/catalog")
async def get_dataset_catalog():
    from backend.datasets.builtin_datasets import get_dataset_catalog
    return {"datasets": get_dataset_catalog()}


@router.get("/builtin")
async def list_builtin():
    from backend.datasets.builtin_datasets import BUILTIN_DATASETS
    return {
        "datasets": {
            name: {
                "input_shape": info["input_shape"],
                "num_classes": info["num_classes"],
                "description": info["description"],
            }
            for name, info in BUILTIN_DATASETS.items()
        }
    }


# ── Request models ──

class LoadDatasetRequest(BaseModel):
    directory: str


class UpdateCaptionRequest(BaseModel):
    caption: str


class BatchCaptionUpdate(BaseModel):
    updates: Dict[str, str]  # index -> caption


class ConceptAnalysisParams(BaseModel):
    min_frequency: int = 2
    max_ngram: int = 3
    min_ngram: int = 1
    top_k: int = 200

class CropToBucketRequest(BaseModel):
    filenames: Optional[List[str]] = None   # null = all images
    preset: Optional[str] = "sdxl"          # key from BUCKET_PRESETS
    min_size: Optional[int] = None          # override if preset="custom"
    max_size: Optional[int] = None
    step: Optional[int] = None
    max_aspect: float = 4.0

# ── Dataset management ──

@router.post("/create")
async def create_dataset(name: str = Form(...)):
    """Create a new empty dataset directory."""
    from configs.config import DATASET_BASE_DIR
    dataset_dir = os.path.join(DATASET_BASE_DIR, name)
    if os.path.exists(dataset_dir):
        raise HTTPException(status_code=409, detail=f"Directory already exists: {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)
    # Auto-load it
    from backend.datasets.dataset_manager import scan_dataset
    info = scan_dataset(dataset_dir)
    return {
        "dataset_id": info.dataset_id,
        "directory": info.directory,
        "total_images": 0,
    }


@router.post("/{dataset_id}/upload")
async def upload_files(dataset_id: str, files: List[UploadFile] = File(...)):
    """Upload image and/or caption files to a loaded dataset."""
    from backend.datasets.dataset_manager import get_loaded_dataset, scan_dataset

    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    uploaded = []
    errors = []
    allowed_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.txt'}

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_ext:
            errors.append(f"Skipped {file.filename}: unsupported extension")
            continue

        # Sanitize filename - keep only the basename
        safe_name = os.path.basename(file.filename)
        dest = os.path.join(ds.directory, safe_name)
        try:
            # Read in chunks for large files
            with open(dest, 'wb') as out:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    out.write(chunk)
            uploaded.append(safe_name)
        except Exception as e:
            errors.append(f"Failed {safe_name}: {str(e)}")
        finally:
            await file.close()

    # Re-scan to pick up new files
    info = scan_dataset(ds.directory)

    return {
        "uploaded": uploaded,
        "errors": errors,
        "dataset_id": info.dataset_id,
        "total_images": info.total_images,
        "total_with_captions": info.total_with_captions,
    }


@router.delete("/{dataset_id}/file/{filename:path}")
async def delete_file(dataset_id: str, filename: str):
    """Delete an image (and its caption) from the dataset."""
    from backend.datasets.dataset_manager import get_loaded_dataset, scan_dataset

    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    safe_name = os.path.basename(filename)
    img_path = os.path.join(ds.directory, safe_name)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not os.path.abspath(img_path).startswith(os.path.abspath(ds.directory)):
        raise HTTPException(status_code=403, detail="Access denied")

    os.remove(img_path)
    txt_path = os.path.splitext(img_path)[0] + '.txt'
    if os.path.exists(txt_path):
        os.remove(txt_path)

    # Re-scan
    info = scan_dataset(ds.directory)
    return {"status": "deleted", "filename": safe_name, "total_images": info.total_images}


class BatchDeleteRequest(BaseModel):
    filenames: List[str]

@router.post("/{dataset_id}/delete-batch")
async def delete_files_batch(dataset_id: str, req: BatchDeleteRequest):
    """Delete multiple images (and their captions) from the dataset."""
    from backend.datasets.dataset_manager import get_loaded_dataset, scan_dataset

    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    deleted = []
    errors = []
    for filename in req.filenames:
        safe_name = os.path.basename(filename)
        img_path = os.path.join(ds.directory, safe_name)
        if not os.path.exists(img_path):
            errors.append(f"Not found: {safe_name}")
            continue
        if not os.path.abspath(img_path).startswith(os.path.abspath(ds.directory)):
            errors.append(f"Access denied: {safe_name}")
            continue
        try:
            os.remove(img_path)
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(txt_path):
                os.remove(txt_path)
            deleted.append(safe_name)
        except Exception as e:
            errors.append(f"Failed {safe_name}: {str(e)}")

    info = scan_dataset(ds.directory)
    return {"deleted": deleted, "errors": errors, "total_images": info.total_images}

@router.post("/load")
async def load_dataset(req: LoadDatasetRequest):
    """Scan a directory and load it as a managed dataset."""
    from backend.datasets.dataset_manager import scan_dataset
    try:
        info = scan_dataset(req.directory)
        return {
            "dataset_id": info.dataset_id,
            "directory": info.directory,
            "total_images": info.total_images,
            "total_with_captions": info.total_with_captions,
            "total_without_captions": info.total_without_captions,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loaded")
async def list_loaded():
    """List all currently loaded datasets."""
    from backend.datasets.dataset_manager import get_all_loaded
    return {"datasets": get_all_loaded()}


@router.delete("/loaded/{dataset_id}")
async def unload(dataset_id: str):
    """Unload a dataset from memory."""
    from backend.datasets.dataset_manager import unload_dataset
    if unload_dataset(dataset_id):
        return {"status": "unloaded"}
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.get("/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get full dataset info including all entries."""
    from backend.datasets.dataset_manager import get_loaded_dataset
    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    return ds.to_dict()


@router.get("/{dataset_id}/entries")
async def get_entries(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    filter: Optional[str] = Query(None, description="Filter: 'captioned', 'uncaptioned', or search text"),
):
    """Get paginated dataset entries with optional filtering."""
    from backend.datasets.dataset_manager import get_loaded_dataset
    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    entries = ds.entries
    # Apply filter
    if filter == "captioned":
        entries = [e for e in entries if e.has_caption_file and e.caption]
    elif filter == "uncaptioned":
        entries = [e for e in entries if not e.has_caption_file or not e.caption]
    elif filter:
        lower_filter = filter.lower()
        entries = [e for e in entries if lower_filter in e.caption.lower() or lower_filter in e.filename.lower()]

    total = len(entries)
    page = entries[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": [
            {**e.to_dict(), "index": ds.entries.index(e)}
            for e in page
        ],
    }

@router.get("/{dataset_id}/image/{image_index}")
async def get_full_image(dataset_id: str, image_index: int):
    """Serve the full-resolution image file."""
    from backend.datasets.dataset_manager import get_loaded_dataset
    ds = get_loaded_dataset(dataset_id)
    if not ds or image_index < 0 or image_index >= len(ds.entries):
        raise HTTPException(status_code=404, detail="Image not found")
    path = ds.entries[image_index].image_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(path)

@router.get("/{dataset_id}/image/by-filename/{filename}")
async def get_full_image_by_filename(dataset_id: str, filename: str):
    from backend.datasets.dataset_manager import get_loaded_dataset
    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not loaded")
    entry = next((e for e in ds.entries if e.filename == filename), None)
    if not entry:
        raise HTTPException(404, "Image not found")
    if not os.path.exists(entry.image_path):
        raise HTTPException(404, "File not found on disk")
    return FileResponse(entry.image_path)


# ── Thumbnails ──

@router.get("/{dataset_id}/thumbnail/{image_index}")
async def get_thumbnail(dataset_id: str, image_index: int, size: int = Query(256, ge=64, le=512)):
    """Get a base64 thumbnail for an image."""
    from backend.datasets.dataset_manager import get_loaded_dataset, get_thumbnail_base64
    ds = get_loaded_dataset(dataset_id)
    if not ds or image_index < 0 or image_index >= len(ds.entries):
        raise HTTPException(status_code=404, detail="Image not found")

    thumb = get_thumbnail_base64(ds.entries[image_index].image_path, max_size=size)
    if not thumb:
        raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

    return {"thumbnail": thumb, "filename": ds.entries[image_index].filename}

@router.get("/{dataset_id}/thumbnails/by-filenames")
async def get_thumbnails_by_filenames(
    dataset_id: str,
    filenames: str = Query(..., description="Comma-separated filenames"),
    size: int = Query(192, ge=64, le=512),
):
    from backend.datasets.dataset_manager import get_loaded_dataset, get_thumbnail_base64
    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not loaded")
    name_list = [f.strip() for f in filenames.split(",") if f.strip()]
    entry_map = {e.filename: e for e in ds.entries}
    results = {}
    for fname in name_list:
        entry = entry_map.get(fname)
        if entry:
            results[fname] = {"thumbnail": get_thumbnail_base64(entry.image_path, max_size=size)}
    return {"thumbnails": results}


@router.get("/{dataset_id}/thumbnails")
async def get_thumbnails_batch(
    dataset_id: str,
    indices: str = Query(..., description="Comma-separated image indices"),
    size: int = Query(192, ge=64, le=512),
):
    """Get thumbnails for multiple images at once."""
    from backend.datasets.dataset_manager import get_loaded_dataset, get_thumbnail_base64
    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    idx_list = [int(i.strip()) for i in indices.split(",") if i.strip().isdigit()]
    results = {}
    for idx in idx_list:
        if 0 <= idx < len(ds.entries):
            thumb = get_thumbnail_base64(ds.entries[idx].image_path, max_size=size)
            results[str(idx)] = {
                "thumbnail": thumb,
                "filename": ds.entries[idx].filename,
            }

    return {"thumbnails": results}

@router.post("/{dataset_id}/crop-to-bucket")
async def crop_images_to_bucket(dataset_id: str, req: CropToBucketRequest):
    from backend.datasets.dataset_manager import get_loaded_dataset
    from backend.datasets.image_caption import (
        generate_buckets, find_closest_bucket, BUCKET_PRESETS
    )
    from pathlib import Path
    import shutil

    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")

    # Resolve bucket parameters
    if req.preset and req.preset != "custom" and req.preset in BUCKET_PRESETS:
        min_s, max_s, step, _ = BUCKET_PRESETS[req.preset]
    else:
        min_s  = req.min_size  or 768
        max_s  = req.max_size  or 1344
        step   = req.step      or 64

    buckets = generate_buckets(min_s, max_s, step, req.max_aspect)

    # Which entries to process
    entry_map = {e.filename: e for e in ds.entries}
    if req.filenames is not None:
        entries_to_process = [entry_map[f] for f in req.filenames if f in entry_map]
    else:
        entries_to_process = list(ds.entries)

    backup_dir = Path(ds.directory) / ".originals"
    backup_dir.mkdir(exist_ok=True)

    results = {"cropped": [], "skipped": [], "errors": [], "buckets_used": []}

    for entry in entries_to_process:
        try:
            img_path  = Path(entry.image_path)
            bucket_w, bucket_h = find_closest_bucket(entry.width, entry.height, buckets)

            if entry.width == bucket_w and entry.height == bucket_h:
                results["skipped"].append(entry.filename)
                continue

            backup_path = backup_dir / img_path.name
            if not backup_path.exists():
                shutil.copy2(img_path, backup_path)

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                orig_w, orig_h = img.size
                scale  = max(bucket_w / orig_w, bucket_h / orig_h)
                new_w  = round(orig_w * scale)
                new_h  = round(orig_h * scale)
                img    = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left   = (new_w - bucket_w) // 2
                top    = (new_h - bucket_h) // 2
                img    = img.crop((left, top, left + bucket_w, top + bucket_h))
                img.save(img_path, quality=95)

            entry.width  = bucket_w
            entry.height = bucket_h
            results["cropped"].append({
                "filename": entry.filename,
                "original": f"{orig_w}×{orig_h}",
                "bucket":   f"{bucket_w}×{bucket_h}",
            })
            b = f"{bucket_w}×{bucket_h}"
            if b not in results["buckets_used"]:
                results["buckets_used"].append(b)

        except Exception as e:
            results["errors"].append({"filename": entry.filename, "error": str(e)})

    results["total_buckets_available"] = len(buckets)
    return results

@router.get("/bucket-presets")
async def get_bucket_presets():
    from backend.datasets.image_caption import BUCKET_PRESETS, generate_buckets
    result = {}
    for name, (min_s, max_s, step, desc) in BUCKET_PRESETS.items():
        if name == "custom":
            result[name] = {"description": desc, "bucket_count": None}
        else:
            result[name] = {
                "description": desc,
                "min_size": min_s,
                "max_size": max_s,
                "step": step,
                "bucket_count": len(generate_buckets(min_s, max_s, step)),
            }
    return result

# ── Caption editing ──

@router.put("/{dataset_id}/caption/{image_index}")
async def update_single_caption(dataset_id: str, image_index: int, req: UpdateCaptionRequest):
    """Update caption for a single image."""
    from backend.datasets.dataset_manager import update_caption
    if update_caption(dataset_id, image_index, req.caption):
        return {"status": "updated"}
    raise HTTPException(status_code=404, detail="Image not found")


@router.put("/{dataset_id}/captions")
async def update_batch_captions(dataset_id: str, req: BatchCaptionUpdate):
    """Update multiple captions at once."""
    from backend.datasets.dataset_manager import batch_update_captions
    result = batch_update_captions(dataset_id, {int(k): v for k, v in req.updates.items()})
    return result

@router.put("/{dataset_id}/caption/by-filename/{filename}")
async def update_caption_by_filename(dataset_id: str, filename: str, req: UpdateCaptionRequest):
    from backend.datasets.dataset_manager import get_loaded_dataset, update_caption
    ds = get_loaded_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not loaded")
    entry = next((e for e in ds.entries if e.filename == filename), None)
    if not entry:
        raise HTTPException(404, f"File {filename} not found")
    idx = ds.entries.index(entry)
    update_caption(dataset_id, idx, req.caption)
    return {"ok": True}

# ── Concept Analysis ──

@router.post("/{dataset_id}/analyze")
async def analyze_concepts(dataset_id: str, params: ConceptAnalysisParams = ConceptAnalysisParams()):
    """Run concept analysis on dataset captions."""
    from backend.datasets.dataset_manager import analyze_concepts as _analyze
    try:
        return _analyze(
            dataset_id,
            min_frequency=params.min_frequency,
            max_ngram=params.max_ngram,
            min_ngram=params.min_ngram,
            top_k=params.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/concept-images")
async def concept_images(dataset_id: str, phrase: str = Query(...)):
    """Get all images associated with a concept phrase."""
    from backend.datasets.dataset_manager import get_concept_images
    results = get_concept_images(dataset_id, phrase)
    return {"phrase": phrase, "count": len(results), "images": results}


@router.post("/{dataset_id}/find-similar")
async def find_similar(dataset_id: str, params: ConceptAnalysisParams = ConceptAnalysisParams()):
    """Find groups of similar phrases that might represent the same concept."""
    from backend.datasets.dataset_manager import analyze_concepts as _analyze, find_similar_phrases
    analysis = _analyze(dataset_id, **params.dict())
    groups = find_similar_phrases(analysis["concepts"])
    return {"groups": groups}
