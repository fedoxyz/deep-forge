"""
Routes for model export and download.

Handles all checkpoint download formats:
  - LoRA runs:     native safetensors, ComfyUI LoRA, Kohya/A1111
  - Full finetune: safetensors, diffusers directory (zipped), GGUF (stub)
  - Simple models: safetensors, .pt state dict, ONNX (stub)

GET  /api/export/formats?checkpoint_path=...&run_name=...
     → Returns available formats for this checkpoint based on its metadata.

GET  /api/export/download?checkpoint_path=...&format=...&run_name=...
     → Streams the file as a download.
"""

import glob
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

router = APIRouter(prefix="/api/export", tags=["export"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_meta(checkpoint_path: str) -> dict:
    base = checkpoint_path.rsplit(".", 1)[0]
    candidates = [
        base + "_meta.json",
        base.replace("_trainer", "") + "_meta.json",
    ]
    print(f"[_load_meta] checkpoint_path={checkpoint_path}")
    print(f"[_load_meta] trying candidates={candidates}")
    for meta_path in candidates:
        print(f"[_load_meta] exists({meta_path})={os.path.exists(meta_path)}")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
    raise HTTPException(404, f"Metadata not found. Tried: {candidates}")

def _infer_run_mode(meta: dict) -> str:
    mode = meta.get("mode", "unknown")
    # Normalize custom training modes → simple for export purposes
    if mode not in ("lora", "finetune", "full", "simple"):
        mode = "simple"
    return mode


def _lora_components(meta: dict) -> list[str]:
    return meta.get("lora_components", [])


def _resolve_lora_path(checkpoint_path: str, comp_name: str, n_components: int) -> str:
    """Return the .safetensors path for a LoRA component."""
    suffix = f"_{comp_name}" if n_components > 1 else ""
    return checkpoint_path.replace(".pt", f"{suffix}.safetensors")


def _zip_dir(dir_path: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, os.path.dirname(dir_path))
                zf.write(full, arcname)
    buf.seek(0)
    return buf


# ── Format definitions ────────────────────────────────────────────────────────

FORMAT_DEFS = {
    # LoRA
    "lora_native": {
        "label": "Native LoRA (.safetensors)",
        "desc": "Internal training format. Resume training or inspect weights.",
        "modes": ["lora"],
        "ext": ".safetensors",
        "badge": None,
    },
    "lora_comfyui": {
        "label": "ComfyUI LoRA (.safetensors)",
        "desc": "diffusion_model prefix, lora_down/up keys, alpha scalars. Drop into ComfyUI loras/.",
        "modes": ["lora"],
        "ext": "_comfyui.safetensors",
        "badge": "ComfyUI",
    },
    "lora_kohya": {
        "label": "Kohya / A1111 (.safetensors)",
        "desc": "Compatible with sd-scripts and Automatic1111 LoRA loader.",
        "modes": ["lora"],
        "ext": "_kohya.safetensors",
        "badge": "A1111",
    },
    # Full model
    "model_safetensors": {
        "label": "Safetensors (.safetensors)",
        "desc": "Full model weights. Load directly for inference or continue training.",
        "modes": ["finetune", "full", "unknown"],
        "ext": ".safetensors",
        "badge": None,
    },
    "model_pt": {
        "label": "PyTorch state dict (.pt)",
        "desc": "Classic torch.save format. Compatible with torch.load.",
        "modes": ["finetune", "full", "unknown"],
        "ext": "_state.pt",
        "badge": None,
    },
    "model_diffusers": {
        "label": "Diffusers (zipped directory)",
        "desc": "HuggingFace Diffusers-compatible format, zipped for download.",
        "modes": ["finetune", "full"],
        "ext": "_diffusers.zip",
        "badge": "HF",
    },
    # Simple / generic
    "simple_safetensors": {
        "label": "Safetensors (.safetensors)",
        "desc": "Standard safetensors. Load with safetensors.torch.load_file.",
        "modes": ["simple"],
        "ext": ".safetensors",
        "badge": None,
    },
    "simple_pt": {
        "label": "PyTorch state dict (.pt)",
        "desc": "Classic torch.save format.",
        "modes": ["simple"],
        "ext": "_state.pt",
        "badge": None,
    },
}


def _available_formats(meta: dict) -> list[dict]:
    mode = _infer_run_mode(meta)
    lora_comps = _lora_components(meta)

    results = []
    for fmt_id, fdef in FORMAT_DEFS.items():
        if mode not in fdef["modes"]:
            continue

        # LoRA formats only when there are LoRA components
        if fmt_id.startswith("lora_") and not lora_comps:
            continue

        # Non-LoRA full/simple formats only when NOT a LoRA run
        if not fmt_id.startswith("lora_") and mode == "lora":
            continue

        results.append({
            "id": fmt_id,
            "label": fdef["label"],
            "desc": fdef["desc"],
            "ext": fdef["ext"],
            "badge": fdef["badge"],
        })

    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/formats")
async def get_export_formats(
    checkpoint_path: str = Query(..., description="Path to checkpoint .pt file"),
):
    """
    Return available download formats for a checkpoint.
    Frontend calls this when the user opens the download dropdown.
    """
    if not os.path.exists(checkpoint_path.rsplit(".", 1)[0] + "_meta.json"):
        # Try treating it as a base path (no extension)
        checkpoint_path = checkpoint_path + ".pt"

    meta = _load_meta(checkpoint_path)
    formats = _available_formats(meta)

    return {
        "formats": formats,
        "mode": _infer_run_mode(meta),
        "lora_components": _lora_components(meta),
        "step": meta.get("global_step"),
        "epoch": meta.get("epoch"),
        "tag": meta.get("tag"),
    }


@router.get("/download")
async def download_checkpoint(
    checkpoint_path: str = Query(...),
    format: str = Query(..., description="Format ID from /formats"),
    run_name: Optional[str] = Query(None),
):
    """
    Stream a checkpoint file in the requested format.
    For formats that require conversion (ComfyUI, Kohya), builds on-the-fly.
    """
    meta = _load_meta(checkpoint_path)
    mode = _infer_run_mode(meta)
    lora_comps = _lora_components(meta)
    base_name = Path(checkpoint_path).stem  # e.g. "step_450"
    run_prefix = run_name or "model"
    download_name = f"{run_prefix}_{base_name}"

    # ── LoRA: native ──────────────────────────────────────────────────────────
    if format == "lora_native":
        _require_lora(lora_comps)
        # Single component: no suffix. Multi: first trainable component.
        comp = lora_comps[0]
        path = _resolve_lora_path(checkpoint_path, comp, len(lora_comps))
        if not os.path.exists(path):
            raise HTTPException(404, f"LoRA file not found: {path}")
        return FileResponse(
            path,
            filename=f"{download_name}_lora.safetensors",
            media_type="application/octet-stream",
        )

    # ── LoRA: ComfyUI ─────────────────────────────────────────────────────────
    if format == "lora_comfyui":
        _require_lora(lora_comps)
        comp_name = lora_comps[0]
        src_path = _resolve_lora_path(checkpoint_path, comp_name, len(lora_comps))
        if not os.path.exists(src_path):
            raise HTTPException(404, f"LoRA weights not found: {src_path}")

        out_path = _tmp_path(checkpoint_path, "_comfyui.safetensors")
        _convert_to_comfyui(src_path, out_path, meta)

        return FileResponse(
            out_path,
            filename=f"{download_name}_comfyui.safetensors",
            media_type="application/octet-stream",
            background=_cleanup_after(out_path),
        )

    # ── LoRA: Kohya / A1111 ───────────────────────────────────────────────────
    if format == "lora_kohya":
        _require_lora(lora_comps)
        comp_name = lora_comps[0]
        src_path = _resolve_lora_path(checkpoint_path, comp_name, len(lora_comps))
        if not os.path.exists(src_path):
            raise HTTPException(404, f"LoRA weights not found: {src_path}")

        out_path = _tmp_path(checkpoint_path, "_kohya.safetensors")
        _convert_to_kohya(src_path, out_path, meta)

        return FileResponse(
            out_path,
            filename=f"{download_name}_kohya.safetensors",
            media_type="application/octet-stream",
            background=_cleanup_after(out_path),
        )

    # ── Full model: safetensors ───────────────────────────────────────────────
    if format in ("model_safetensors", "simple_safetensors"):
        # Try direct replacement first, then stem-based lookup
        sf_path = checkpoint_path.replace(".pt", ".safetensors")
        if not os.path.exists(sf_path):
            # e.g. best_model_trainer.pt → best_model.safetensors
            stem = Path(checkpoint_path).stem.replace("_trainer", "")
            sf_path = str(Path(checkpoint_path).parent / f"{stem}.safetensors")
        if not os.path.exists(sf_path):
            raise HTTPException(404, f"Safetensors file not found: {sf_path}")
        return FileResponse(
            sf_path,
            filename=f"{download_name}.safetensors",
            media_type="application/octet-stream",
        )
    
    # ── Full model / simple: .pt state dict ───────────────────────────────────
    if format in ("model_pt", "simple_pt"):
        stem = Path(checkpoint_path).stem.replace("_trainer", "")
        # Prefer _trainer.pt (full optimizer state), else the base .pt
        trainer_pt = Path(checkpoint_path).parent / f"{stem}_trainer.pt"
        model_pt   = Path(checkpoint_path).parent / f"{stem}.pt"
        src = str(trainer_pt) if trainer_pt.exists() else str(model_pt)
        if not Path(src).exists():
            raise HTTPException(404, f"PyTorch file not found: {src}")
        return FileResponse(
            src,
            filename=f"{download_name}.pt",
            media_type="application/octet-stream",
        )

    # ── Full model: Diffusers zip ─────────────────────────────────────────────
    if format == "model_diffusers":
        # Expect a diffusers dir saved alongside the checkpoint
        run_dir = os.path.dirname(os.path.dirname(checkpoint_path))  # up from checkpoints/
        diffusers_dir = os.path.join(run_dir, "diffusers")
        if not os.path.isdir(diffusers_dir):
            raise HTTPException(
                404,
                "No diffusers directory found for this run. "
                "Re-save with pipeline.save_pretrained(run_dir/diffusers).",
            )
        buf = _zip_dir(diffusers_dir)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{download_name}_diffusers.zip"'
            },
        )

    raise HTTPException(400, f"Unknown format: {format!r}")


# ── Conversion helpers ────────────────────────────────────────────────────────

def _require_lora(lora_comps: list):
    if not lora_comps:
        raise HTTPException(400, "This checkpoint has no LoRA components.")


def _tmp_path(checkpoint_path: str, suffix: str) -> str:
    base = checkpoint_path.rsplit(".", 1)[0]
    return base + "_export" + suffix


def _cleanup_after(path: str):
    """BackgroundTask: delete temp file after response is sent."""
    from starlette.background import BackgroundTask

    def _rm():
        try:
            os.remove(path)
        except Exception:
            pass

    return BackgroundTask(_rm)


def _convert_to_comfyui(src_path: str, out_path: str, meta: dict):
    """
    Convert native LoRA safetensors → ComfyUI format.
    Mirrors LoRAInjector.save_weights_comfyui but works from a saved file
    so the trainer doesn't need to be in memory.
    """
    from safetensors.torch import load_file, save_file

    sd = load_file(src_path)
    out = {}

    # Group by layer name (everything before .lora_A / .lora_B)
    layers: dict[str, dict] = {}
    for key, tensor in sd.items():
        # Expected key pattern: "<layer_name>.lora_A.weight" or ".lora_B.weight"
        if ".lora_A." in key:
            layer = key.split(".lora_A.")[0]
            param = "lora_A." + key.split(".lora_A.")[1]
        elif ".lora_B." in key:
            layer = key.split(".lora_B.")[0]
            param = "lora_B." + key.split(".lora_B.")[1]
        else:
            # Pass through unknowns (alpha scalars, etc.)
            out[key] = tensor.contiguous().cpu().to(torch.float16)
            continue
        layers.setdefault(layer, {})[param] = tensor

    alpha_val = float(meta.get("lora_info", {}).get(
        list(meta.get("lora_info", {}).keys())[0], {}
    ).get("alpha", 16)) if meta.get("lora_info") else 16.0

    for layer_name, params in layers.items():
        # Normalize prefix
        if layer_name.startswith("model."):
            comfy_layer = "diffusion_model." + layer_name[len("model."):]
        elif layer_name.startswith("diffusion_model."):
            comfy_layer = layer_name
        else:
            comfy_layer = "diffusion_model." + layer_name

        for pname, tensor in params.items():
            comfy_pname = (
                pname.replace("lora_A.", "lora_down.").replace("lora_B.", "lora_up.")
            )
            out[f"{comfy_layer}.{comfy_pname}"] = (
                tensor.contiguous().cpu().to(torch.float16)
            )

        out[f"{comfy_layer}.alpha"] = torch.tensor(alpha_val, dtype=torch.float32)

    save_file(
        out,
        out_path,
        metadata={
            "format": "comfyui",
            "base_model": meta.get("mode", "unknown"),
        },
    )


def _convert_to_kohya(src_path: str, out_path: str, meta: dict):
    """
    Convert native LoRA safetensors → Kohya / A1111 format.
    Key renames: lora_A→lora_down, lora_B→lora_up (same as ComfyUI),
    but uses unet prefix rather than diffusion_model.
    Also adds te_ prefix for text encoder layers if present.
    """
    from safetensors.torch import load_file, save_file

    sd = load_file(src_path)
    out = {}

    alpha_val = float(meta.get("lora_info", {}).get(
        list(meta.get("lora_info", {}).keys())[0] if meta.get("lora_info") else "",
        {},
    ).get("alpha", 16)) if meta.get("lora_info") else 16.0

    layers: dict[str, dict] = {}
    for key, tensor in sd.items():
        if ".lora_A." in key:
            layer = key.split(".lora_A.")[0]
            param = "lora_A." + key.split(".lora_A.")[1]
        elif ".lora_B." in key:
            layer = key.split(".lora_B.")[0]
            param = "lora_B." + key.split(".lora_B.")[1]
        else:
            out[key] = tensor.contiguous().cpu().to(torch.float16)
            continue
        layers.setdefault(layer, {})[param] = tensor

    for layer_name, params in layers.items():
        # Kohya uses lora_unet_ prefix for denoiser layers
        clean = layer_name.replace(".", "_")
        if layer_name.startswith("model."):
            clean = "lora_unet_" + layer_name[len("model."):].replace(".", "_")
        elif layer_name.startswith("text_encoder"):
            clean = "lora_te_" + layer_name.replace(".", "_")
        else:
            clean = "lora_unet_" + clean

        for pname, tensor in params.items():
            kohya_pname = (
                pname.replace("lora_A.", "lora_down.").replace("lora_B.", "lora_up.")
            )
            out[f"{clean}.{kohya_pname}"] = (
                tensor.contiguous().cpu().to(torch.float16)
            )

        out[f"{clean}.alpha"] = torch.tensor(alpha_val, dtype=torch.float32)

    save_file(
        out,
        out_path,
        metadata={
            "format": "kohya",
            "ss_base_model_version": meta.get("mode", "unknown"),
        },
    )
