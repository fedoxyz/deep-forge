"""
Auto-download model files if they don't exist on disk.
Called during component loading before any model is instantiated.
"""

import os
from typing import Optional

_ZIMAGE_URLS = {
    "z_image_turbo.safetensors":      "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    "z_image_turbo_bf16.safetensors": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    "qwen_3_4b.safetensors":          "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
    "ae.safetensors":                 "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
}

def ensure_model_file(path: str, url: Optional[str] = None) -> str:
    """
    Check if model file exists; download it if not.
    
    Args:
        path: Absolute path where the model should be
        url:  Direct download URL. If None, infers from filename.
    
    Returns:
        The path (same as input) once the file is confirmed present.
    """
    if os.path.exists(path):
        return path

    filename = os.path.basename(path)

    # Infer URL from filename if not provided
    if url is None:
        url = _ZIMAGE_URLS.get(filename)

    if url is None:
        raise FileNotFoundError(
            f"Model file not found and no download URL known:\n  {path}\n"
            f"Either place the file there manually or add its URL to _ZIMAGE_URLS."
        )

    print(f"[Downloader] '{filename}' not found at {path}")
    print(f"[Downloader] Downloading from:\n  {url}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    _download(url, path)
    return path


def _download(url: str, dest: str):
    """Stream download with progress bar."""
    import urllib.request

    tmp = dest + ".tmp"
    try:
        downloaded = 0
        last_pct = -1

        def _progress(block_num, block_size, total_size):
            nonlocal downloaded, last_pct
            downloaded += block_size
            if total_size > 0:
                pct = int(downloaded * 100 / total_size)
                if pct != last_pct and pct % 5 == 0:
                    gb = downloaded / 1e9
                    total_gb = total_size / 1e9
                    print(f"[Downloader]   {pct}%  ({gb:.2f}/{total_gb:.2f} GB)", flush=True)
                    last_pct = pct

        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        os.rename(tmp, dest)
        size_gb = os.path.getsize(dest) / 1e9
        print(f"[Downloader] Done — {size_gb:.2f} GB saved to {dest}")

    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"[Downloader] Failed to download {url}: {e}")
