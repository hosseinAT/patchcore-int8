"""Tiny fake MVTec folder tree for smoke tests and CI (no real data, no download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_fake_mvtec(root: Path, categories, n_train: int = 6, n_good: int = 3, n_bad: int = 3, size: int = 64) -> Path:
    rng = np.random.default_rng(0)
    for cat in categories:
        base = root / cat
        for d in ("train/good", "test/good", "test/scratch", "ground_truth/scratch"):
            (base / d).mkdir(parents=True, exist_ok=True)

        def img(defect: bool):
            a = rng.normal(128, 10, (size, size, 3))
            if defect:
                a[20:30, 10:50] = 255
            return Image.fromarray(a.clip(0, 255).astype(np.uint8))

        for i in range(n_train):
            img(False).save(base / "train/good" / f"{i:03d}.png")
        for i in range(n_good):
            img(False).save(base / "test/good" / f"{i:03d}.png")
        for i in range(n_bad):
            img(True).save(base / "test/scratch" / f"{i:03d}.png")
            m = np.zeros((size, size), np.uint8)
            m[20:30, 10:50] = 255
            Image.fromarray(m).save(base / "ground_truth/scratch" / f"{i:03d}_mask.png")
    return root
