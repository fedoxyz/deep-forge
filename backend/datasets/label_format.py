"""
Forge label format parser/serializer.
D <class_name> <x1> <y1> <x2> <y2>
S <class_name> <crop_x1> <crop_y1> <crop_x2> <crop_y2> <rle values...>
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union
import numpy as np


@dataclass
class DetectionLabel:
    class_name: str
    x1: int; y1: int; x2: int; y2: int


@dataclass
class SegmentationLabel:
    class_name: str
    crop_x1: int; crop_y1: int; crop_x2: int; crop_y2: int
    rle: List[int]

    def to_mask(self, img_w: int, img_h: int) -> np.ndarray:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        W = self.crop_x2 - self.crop_x1
        pixel = 0
        for i in range(0, len(self.rle), 2):
            val, count = self.rle[i], self.rle[i+1]
            for _ in range(count):
                lx, ly = pixel % W, pixel // W
                gx, gy = self.crop_x1 + lx, self.crop_y1 + ly
                if 0 <= gx < img_w and 0 <= gy < img_h:
                    mask[gy, gx] = val
                pixel += 1
        return mask


AnyLabel = Union[DetectionLabel, SegmentationLabel]


def parse_labels(text: str) -> List[AnyLabel]:
    labels: List[AnyLabel] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        t = parts[0]
        if t == 'D' and len(parts) >= 6:
            labels.append(DetectionLabel(
                class_name=parts[1],
                x1=int(parts[2]), y1=int(parts[3]),
                x2=int(parts[4]), y2=int(parts[5]),
            ))
        elif t == 'S' and len(parts) >= 7:
            labels.append(SegmentationLabel(
                class_name=parts[1],
                crop_x1=int(parts[2]), crop_y1=int(parts[3]),
                crop_x2=int(parts[4]), crop_y2=int(parts[5]),
                rle=list(map(int, parts[6:])),
            ))
    return labels


def serialize_labels(labels: List[AnyLabel]) -> str:
    lines = []
    for lb in labels:
        if isinstance(lb, DetectionLabel):
            lines.append(f"D {lb.class_name} {lb.x1} {lb.y1} {lb.x2} {lb.y2}")
        elif isinstance(lb, SegmentationLabel):
            rle_str = ' '.join(map(str, lb.rle))
            lines.append(
                f"S {lb.class_name} {lb.crop_x1} {lb.crop_y1} "
                f"{lb.crop_x2} {lb.crop_y2} {rle_str}"
            )
    return '\n'.join(lines)
