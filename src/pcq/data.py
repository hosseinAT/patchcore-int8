"""Minimal MVTec AD reader (folder layout as distributed by MVTec / Kaggle).

<root>/<category>/train/good/*.png
<root>/<category>/test/<defect or good>/*.png
<root>/<category>/ground_truth/<defect>/<name>_mask.png

Kept independent of anomalib's datamodule so that the benchmark controls
pre-processing exactly and runs the same on Colab, Kaggle and locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def find_root(candidates: list[str | Path]) -> Path:
    """Return the first directory that contains an MVTec category folder (e.g. 'bottle/train')."""
    for c in candidates:
        c = Path(c)
        if not c.exists():
            continue
        for p in [c, *c.glob("*"), *c.glob("*/*"), *c.glob("*/*/*")]:
            if p.is_dir() and any((p / cat / "train").is_dir() for cat in CATEGORIES):
                return p
    raise FileNotFoundError(f"No MVTec AD folder found in {candidates}")


def load_image(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def load_mask(path: Path | None, size: int) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(1, size, size)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return torch.from_numpy((np.asarray(m) > 127).astype(np.float32)).unsqueeze(0)


def load_category(root: str | Path, category: str, size: int = 256, max_train: int | None = None, max_test: int | None = None):
    """-> dict(train [N,3,S,S], test [M,3,S,S], masks [M,1,S,S], labels [M], defect_types list[str])."""
    root = Path(root) / category
    train_files = sorted((root / "train" / "good").glob("*.png"))[:max_train]
    if not train_files:
        raise FileNotFoundError(f"no training images in {root / 'train' / 'good'}")
    test_items = []
    for d in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.png")):
            mask = None if d.name == "good" else root / "ground_truth" / d.name / f"{f.stem}_mask.png"
            test_items.append((f, mask, int(d.name != "good"), d.name))
    if max_test is not None:  # keep class balance roughly by taking every k-th item
        step = max(1, len(test_items) // max_test)
        test_items = test_items[::step][:max_test]
    return {
        "train": torch.stack([load_image(f, size) for f in train_files]),
        "test": torch.stack([load_image(f, size) for f, *_ in test_items]),
        "masks": torch.stack([load_mask(m, size) for _, m, *_ in test_items]),
        "labels": torch.tensor([l for *_, l, _ in test_items]),
        "defect_types": [t for *_, t in test_items],
    }
