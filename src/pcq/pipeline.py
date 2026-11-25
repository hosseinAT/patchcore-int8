"""PatchCore with interchangeable backbones and memory banks.

The anomalib `PatchcoreModel` performs feature extraction and scoring in one
`forward`. To compare FP32 and INT8 backbones on exactly the same data, this
module splits it into the same steps anomalib uses internally:

    backbone -> 3x3 avg-pool -> generate_embedding -> reshape_embedding
    -> (fit)   k-center-greedy coreset            = memory bank
    -> (score) nearest_neighbors -> compute_anomaly_score -> anomaly_map_generator

All steps after the backbone are anomalib's own methods; only the backbone and
the memory-bank storage are swapped.
"""

from __future__ import annotations

import time
import warnings

import torch


def build_patchcore(pretrained: bool = True, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3")):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PatchcoreModel(layers=list(layers), backbone=backbone, pre_trained=pretrained, num_neighbors=9)
    model.eval()
    return model


@torch.no_grad()
def embed(model, backbone, images: torch.Tensor, backbone_device: str, compute_device: str, batch: int = 8):
    """Yield (embedding [B*h*w, C], (B, h, w)) batch by batch."""
    for i in range(0, len(images), batch):
        x = images[i : i + batch].to(backbone_device)
        feats = backbone(x)
        if not isinstance(feats, dict):
            feats = dict(zip(model.feature_extractor.layers, feats, strict=True))
        feats = {k: model.feature_pooler(v.float().to(compute_device)) for k, v in feats.items()}
        emb = model.generate_embedding(feats)
        b, _, h, w = emb.shape
        yield model.reshape_embedding(emb), (b, h, w)


@torch.no_grad()
def fit_memory_bank(model, backbone, train_images, backbone_device, compute_device, coreset_ratio: float = 0.1) -> torch.Tensor:
    embs = [e for e, _ in embed(model, backbone, train_images, backbone_device, compute_device)]
    model.embedding_store = [torch.cat(embs)]
    model.memory_bank = torch.empty(0, device=compute_device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.subsample_embedding(coreset_ratio)
    return model.memory_bank.detach().clone()


@torch.no_grad()
def score(model, memory_bank: torch.Tensor, embedding: torch.Tensor, shape, image_size):
    """anomalib's inference tail, on a given memory bank. Returns (image scores [B], maps [B,1,H,W])."""
    b, h, w = shape
    model.memory_bank = memory_bank
    patch_scores, locations = model.nearest_neighbors(embedding=embedding, n_neighbors=1)
    patch_scores = patch_scores.reshape((b, -1))
    locations = locations.reshape((b, -1))
    pred = model.compute_anomaly_score(patch_scores, locations, embedding)
    maps = model.anomaly_map_generator(patch_scores.reshape((b, 1, h, w)), image_size)
    return pred.cpu(), maps.cpu()


@torch.no_grad()
def time_module(fn, x: torch.Tensor, warmup: int = 3, runs: int = 15) -> float:
    """Median wall-clock time of fn(x) in milliseconds."""
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    return t[len(t) // 2]
