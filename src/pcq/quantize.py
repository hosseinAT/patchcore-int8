"""Post-training quantization of the two expensive parts of PatchCore.

1. Backbone (WideResNet-50 feature extractor, ~68 M parameters)
   Static INT8 quantization with PyTorch FX graph mode (x86 / fbgemm backend):
   per-channel symmetric INT8 weights, per-tensor affine UINT8 activations,
   Conv+BN+ReLU fused, activation ranges calibrated on normal training images.

2. Memory bank (coreset of patch features, typically 10^4 - 10^5 x 1536 floats)
   * fp16 : plain half-precision storage
   * int8 : per-dimension symmetric INT8 with one fp32 scale per feature dimension.
            Nearest-neighbour search runs on de-quantized values, so only the
            storage changes, not the search algorithm.
"""

from __future__ import annotations

import copy
import io
import warnings

import torch
from torch import nn


def state_dict_bytes(module: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


def quantize_backbone(backbone: nn.Module, calib_images: torch.Tensor, batch: int = 8) -> nn.Module:
    """Return an INT8 copy of a timm `features_only` backbone. Inference runs on CPU."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "x86"
    fp = copy.deepcopy(backbone).cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(fp, get_default_qconfig_mapping("x86"), example_inputs=(calib_images[:1],))
        with torch.no_grad():
            for i in range(0, len(calib_images), batch):
                prepared(calib_images[i : i + batch])
        return convert_fx(prepared)


class QuantizedMemoryBank:
    """Symmetric per-dimension INT8 storage of a (N, D) float matrix."""

    def __init__(self, bank: torch.Tensor):
        amax = bank.abs().amax(dim=0).clamp_min(1e-8)
        self.scale = (amax / 127.0).float()
        self.q = torch.round(bank / self.scale).clamp(-127, 127).to(torch.int8)

    def dequantize(self) -> torch.Tensor:
        return self.q.float() * self.scale

    def nbytes(self) -> int:
        return self.q.numel() + self.scale.numel() * 4


def compress_memory_bank(model, mode: str) -> int:
    """Replace model.memory_bank in place by its fp16 / int8 round-trip; return the storage size in bytes."""
    bank = model.memory_bank.detach().float()
    if mode == "fp32":
        return bank.numel() * 4
    if mode == "fp16":
        model.memory_bank = bank.half().float()
        return bank.numel() * 2
    if mode == "int8":
        qb = QuantizedMemoryBank(bank)
        model.memory_bank = qb.dequantize()
        return qb.nbytes()
    raise ValueError(mode)


def relative_feature_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / a.norm().clamp_min(1e-12))
