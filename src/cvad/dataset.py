from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_mask(path: Optional[Path]) -> Optional[Image.Image]:
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("L")


@dataclass
class AnomalySample:
    image: torch.Tensor
    mask: Optional[torch.Tensor]
    is_anomaly: int
    path: Path


class MVTecDataset(Dataset):
    def __init__(self, root: Path, split: str, category: Optional[str], transform: Callable, mask_transform: Callable | None = None):
        self.root = Path(root)
        self.split = split
        self.category = category
        self.transform = transform
        self.mask_transform = mask_transform
        self.items: List[Tuple[Path, Optional[Path], int]] = []
        self._collect()

    def _collect(self) -> None:
        categories = [self.category] if self.category else [d.name for d in self.root.iterdir() if d.is_dir()]
        for cat in categories:
            base = self.root / cat / self.split
            mask_root = self.root / cat / "ground_truth"
            for defect_dir in sorted(base.iterdir()):
                if not defect_dir.is_dir():
                    continue
                is_anomaly = 0 if defect_dir.name == "good" else 1
                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if is_anomaly:
                        relative = img_path.relative_to(base)
                        mask_path = mask_root / relative.with_suffix("") / f"{relative.stem}_mask.png"
                    self.items.append((img_path, mask_path, is_anomaly))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> AnomalySample:
        img_path, mask_path, is_anomaly = self.items[idx]
        image = self.transform(_load_image(img_path))
        mask_img = _load_mask(mask_path)
        mask_tensor = self.mask_transform(mask_img) if (mask_img and self.mask_transform) else None
        return AnomalySample(image=image, mask=mask_tensor, is_anomaly=is_anomaly, path=img_path)


def build_transforms(image_size: int) -> tuple[Callable, Callable]:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
            transforms.Lambda(lambda t: (t > 0).float()),
        ]
    )
    return transform, mask_transform


def dataset_hash(root: Path) -> str:
    hasher = hashlib.sha256()
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            path = Path(dirpath) / fname
            hasher.update(str(path.relative_to(root)).encode())
            hasher.update(str(path.stat().st_size).encode())
    return hasher.hexdigest()
