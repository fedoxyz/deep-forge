"""
Classification dataset loader.
Expects directory structure:
    root/
        class_a/  (or "0", "1", etc.)
            img1.jpg
            img2.png
        class_b/
            ...

Returns (train_dataloader, val_dataloader) and stores class_to_idx mapping.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

# Stored after first load so training code / routes can inspect it
_last_class_to_idx: dict = {}
_last_idx_to_class: dict = {}


def get_class_mapping():
    """Return the class↔index mappings from the last loaded dataset."""
    return _last_class_to_idx.copy(), _last_idx_to_class.copy()


def create_classification_dataloader(
    root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    global _last_class_to_idx, _last_idx_to_class

    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    from PIL import Image as _PIL
    _sample_mode = _PIL.open(next(
        p for p, _ in datasets.ImageFolder(root=root, is_valid_file=_is_valid_image).samples
    )).mode

    train_tf = transforms.Compose([
        *([ transforms.Grayscale(1) ] if _sample_mode != 'RGB' else []),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        *([ transforms.Grayscale(1) ] if _sample_mode != 'RGB' else []),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=root, transform=train_tf,
                                        is_valid_file=_is_valid_image)

    if len(full_dataset) == 0:
        raise ValueError(
            f"No images found in {root}. "
            f"Expected subfolders per class containing image files."
        )

    # Store mappings globally for inspection
    _last_class_to_idx = full_dataset.class_to_idx          # {"cat": 0, "dog": 1}
    _last_idx_to_class = {v: k for k, v in _last_class_to_idx.items()}  # {0: "cat", 1: "dog"}

    print(f"[Dataset] Found {len(full_dataset)} samples in {root}")
    print(f"[Dataset] Classes ({len(_last_class_to_idx)}): {_last_class_to_idx}")

    # Bucket distribution log (mirrors existing style)
    from collections import Counter
    label_counts = Counter(label for _, label in full_dataset.samples)
    print("[Dataset] Class distribution:")
    for idx, count in sorted(label_counts.items()):
        print(f"  [{idx}] {_last_idx_to_class[idx]}: {count} images")

    if val_split > 0:
        val_size = max(1, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        # Apply val transform to val split
        val_ds.dataset = datasets.ImageFolder(root=root, transform=val_tf,
                                               is_valid_file=_is_valid_image)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        return train_dl, val_dl
    else:
        train_dl = DataLoader(full_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
        return train_dl, None


def _is_valid_image(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
