"""QAT of MobileNetV2's *classifier head only*, with a matched FP32 fine-tune.

The pretrained MobileNetV2 backbone stays FP32 and frozen. This explicitly is
not end-to-end QAT or a fully compiled/int8-accelerated Qualcomm model.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

from .data import load_manifest, preprocess
from .utils import sha256


class QuantAwareHead(nn.Module):
    def __init__(self, original: nn.Linear, *, quant_aware: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())
        self.quant_aware = quant_aware
        if quant_aware:
            self.input_fakequant = FakeQuantize(
                observer=MovingAverageMinMaxObserver,
                quant_min=0, quant_max=255, dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            )
            self.weight_fakequant = FakeQuantize(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric, ch_axis=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_aware:
            x = self.input_fakequant(x)
            w = self.weight_fakequant(self.weight)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


def extract_features(backbone: nn.Module, manifest: Path, data_root: Path) -> tuple:
    examples = load_manifest(manifest, data_root)
    if any(e.synthetic or e.label is None for e in examples):
        raise ValueError("QAT outcome requires real labeled images")
    features, labels = [], []
    backbone.eval()
    with torch.inference_mode():
        for example in examples:
            x = torch.from_numpy(preprocess(example.path))
            f = backbone(x)
            pooled = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
            features.append(pooled.cpu())
            labels.append(int(example.label))
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


def train_head(
    original_head: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    quant_aware: bool,
    seed: int,
    epochs: int = 3,
    learning_rate: float = 0.001,
    batch_size: int = 16,
) -> tuple[QuantAwareHead, list[float]]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    if len(features) != len(labels) or not len(features):
        raise ValueError("Need aligned nonempty features and labels")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    head = QuantAwareHead(copy.deepcopy(original_head), quant_aware=quant_aware)
    optimizer = torch.optim.SGD(head.parameters(), lr=learning_rate, momentum=0.9)
    histories = []
    head.train()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed * 100 + epoch)
        permutation = torch.randperm(len(labels), generator=generator)
        losses = []
        for positions in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = head(features[positions])
            loss = F.cross_entropy(logits, labels[positions])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        histories.append(float(np.mean(losses)))
    head.eval()
    if quant_aware:
        head.input_fakequant.disable_observer()
        head.weight_fakequant.disable_observer()
    return head, histories


@torch.inference_mode()
def predict_head(head: nn.Module, features: torch.Tensor) -> list[int]:
    head.eval()
    return head(features).argmax(dim=1).cpu().tolist()


def feature_provenance(manifest: Path, length: int) -> dict:
    return {"manifest_sha256": sha256(manifest), "image_count": length}
