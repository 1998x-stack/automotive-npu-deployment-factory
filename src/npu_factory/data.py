"""Disjoint, immutable calibration and test image manifests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import sha256


@dataclass(frozen=True)
class Example:
    path: Path
    label: int | None
    synthetic: bool


def load_manifest(manifest: Path, root: Path) -> list[Example]:
    manifest = Path(manifest).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    output: list[Example] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        path_value = row.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Invalid path on line {line_number}")
        path = (root / path_value).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Path escapes dataset root on line {line_number}")
        if path in seen or not path.is_file():
            raise ValueError(f"Duplicate or invalid image path: {path}")
        seen.add(path)
        label = row.get("label")
        if label is not None and (type(label) is not int or not 0 <= label < 1000):
            raise ValueError(f"Invalid ImageNet label on line {line_number}")
        output.append(Example(path, label, row.get("source") == "synthetic"))
    if not output:
        raise ValueError("Manifest contains no images")
    return output


def validate_disjoint(calibration: list[Example], test: list[Example]) -> None:
    """Detect leakage by resolved paths AND image bytes, including renamed copies."""
    a_paths = {e.path for e in calibration}
    b_paths = {e.path for e in test}
    if a_paths & b_paths:
        raise ValueError("Calibration/test manifests overlap by path")
    if {sha256(e.path) for e in calibration} & {sha256(e.path) for e in test}:
        raise ValueError("Calibration/test manifests overlap by file content")


def preprocess(path: Path) -> np.ndarray:
    """ImageNet resize-short-side=256, center-crop=224, NCHW float32 normalization."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        target = 232
        resized = (int(w * target / min(w, h)), int(h * target / min(w, h)))
        image = image.resize(resized, resample=Image.Resampling.BILINEAR)
        w, h = image.size
        x, y = round((w - 224) / 2), round((h - 224) / 2)
        image = image.crop((x, y, x + 224, y + 224))
        array = np.asarray(image, dtype=np.float32) / np.float32(255)
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def make_smoke_dataset(root: Path, count: int = 6, seed: int = 7) -> dict[str, str]:
    """Only pipeline fixtures. Synthetic labels MUST NOT be reported as model accuracy."""
    if count < 4:
        raise ValueError("count must be >= 4 to create two nonempty splits")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    records = {"calibration": [], "test": []}
    for index in range(count):
        split = "calibration" if index < count // 2 else "test"
        relative = Path("images") / f"synthetic_{index:04d}.png"
        path = root / relative
        path.parent.mkdir(exist_ok=True, parents=True)
        pixels = generator.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        Image.fromarray(pixels, "RGB").save(path)
        records[split].append(json.dumps({
            "path": relative.as_posix(), "label": 0, "source": "synthetic"
        }))
    paths = {}
    for split, lines in records.items():
        path = root / f"{split}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[split] = str(path)
    return paths
