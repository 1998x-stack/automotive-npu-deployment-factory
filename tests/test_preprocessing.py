"""Regression guard: pretrained MobileNetV2 normalization must match torchvision."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from npu_factory.data import preprocess


def test_mobilenet_v2_transforms_match(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.models import MobileNet_V2_Weights

    rng = np.random.default_rng(2026)
    for width, height in ((321, 245), (240, 330), (256, 256)):
        path = tmp_path / f"{width}-{height}.png"
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(path)
        with Image.open(path) as img:
            reference = MobileNet_V2_Weights.DEFAULT.transforms()(img.convert("RGB"))
        actual = preprocess(path)[0]
        np.testing.assert_allclose(actual, reference.numpy(), atol=1e-5, rtol=0)
