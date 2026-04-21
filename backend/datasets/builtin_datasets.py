"""
Built-in dataset loaders.

Provides ready-to-use datasets for testing and experimentation:
- MNIST, FashionMNIST, CIFAR-10, CIFAR-100
- Custom image folder (for classification)
- Image+caption pairs (existing, for diffusion LoRA)

All datasets return dicts with 'input'/'target' keys for unified trainer compatibility.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# ── Standard datasets ──

BUILTIN_DATASETS = {
    'mnist': {
        'class': datasets.MNIST,
        'input_shape': [1, 28, 28],
        'num_classes': 10,
        'description': 'Handwritten digits 0-9, 28x28 grayscale',
    },
    'fashion_mnist': {
        'class': datasets.FashionMNIST,
        'input_shape': [1, 28, 28],
        'num_classes': 10,
        'description': 'Fashion items, 28x28 grayscale',
    },
    'cifar10': {
        'class': datasets.CIFAR10,
        'input_shape': [3, 32, 32],
        'num_classes': 10,
        'description': 'Natural images in 10 classes, 32x32 RGB',
    },
    'cifar100': {
        'class': datasets.CIFAR100,
        'input_shape': [3, 32, 32],
        'num_classes': 100,
        'description': 'Natural images in 100 classes, 32x32 RGB',
    },
}


class WrappedDataset(Dataset):
    """Wraps a torchvision dataset to return dicts."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {'input': img, 'target': label}


def get_builtin_dataset(
    name: str,
    data_dir: str = './data',
    train: bool = True,
    augment: bool = True,
    val_split: float = 0.1,
) -> Tuple[Dataset, Optional[Dataset], Dict[str, Any]]:
    """
    Load a built-in dataset.

    Returns: (train_dataset, val_dataset, info_dict)
    """
    if name not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(BUILTIN_DATASETS.keys())}")

    info = BUILTIN_DATASETS[name]
    ds_class = info['class']

    # Transforms
    if name in ('mnist', 'fashion_mnist'):
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if augment and train:
            transform_list = [transforms.RandomRotation(10), transforms.RandomAffine(0, translate=(0.1, 0.1))] + transform_list
    else:
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if augment and train:
            transform_list = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)] + transform_list

    transform = transforms.Compose(transform_list)

    full_dataset = ds_class(root=data_dir, train=train, download=True, transform=transform)

    # Split into train/val
    val_dataset = None
    if val_split > 0 and train:
        n = len(full_dataset)
        n_val = int(n * val_split)
        n_train = n - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_dataset = WrappedDataset(train_dataset)
        val_dataset = WrappedDataset(val_dataset)
    else:
        train_dataset = WrappedDataset(full_dataset)

    # Test set
    if not train:
        train_dataset = WrappedDataset(full_dataset)

    return train_dataset, val_dataset, {
        'input_shape': info['input_shape'],
        'num_classes': info['num_classes'],
        'description': info['description'],
        'train_size': len(train_dataset),
        'val_size': len(val_dataset) if val_dataset else 0,
    }


def create_builtin_dataloaders(
    name: str,
    batch_size: int = 64,
    num_workers: int = 2,
    data_dir: str = './data',
    val_split: float = 0.1,
    augment: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Dict]:
    """Create DataLoaders for a built-in dataset."""
    train_ds, val_ds, info = get_builtin_dataset(
        name, data_dir=data_dir, train=True, augment=augment, val_split=val_split,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

    val_dl = None
    if val_ds:
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, info


class ImageFolderDataset(Dataset):
    """
    Generic image folder dataset for classification.
    Structure:
        root/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
    """

    def __init__(self, root: str, image_size: int = 224, augment: bool = True):
        transform_list = []
        if augment:
            transform_list.extend([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            transform_list.extend([
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dataset = datasets.ImageFolder(root, transform=transforms.Compose(transform_list))
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {'input': img, 'target': label}


def get_dataset_catalog() -> Dict[str, Any]:
    """Return catalog of available datasets for the frontend."""
    catalog = {}
    for name, info in BUILTIN_DATASETS.items():
        catalog[name] = {
            'input_shape': info['input_shape'],
            'num_classes': info['num_classes'],
            'description': info['description'],
            'type': 'builtin',
        }
    catalog['image_folder'] = {
        'description': 'Custom image folder (class subfolders)',
        'type': 'custom',
    }
    catalog['image_caption'] = {
        'description': 'Image + text caption pairs (for diffusion LoRA)',
        'type': 'custom',
    }
    return catalog
