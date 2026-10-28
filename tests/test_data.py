import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from npu_factory.data import load_manifest, make_smoke_dataset, preprocess, validate_disjoint


def test_smoke_splits_disjoint_and_preprocess(tmp_path: Path):
    paths = make_smoke_dataset(tmp_path / "data", count=6, seed=123)
    cal = load_manifest(Path(paths["calibration"]), tmp_path / "data")
    test = load_manifest(Path(paths["test"]), tmp_path / "data")
    validate_disjoint(cal, test)
    assert len(cal) == len(test) == 3
    assert all(e.synthetic for e in cal + test)
    value = preprocess(cal[0].path)
    assert value.shape == (1, 3, 224, 224)
    assert value.dtype == np.float32
    assert np.isfinite(value).all()


def test_reject_path_escape(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "image.png"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(outside)
    manifest = root / "evil.jsonl"
    manifest.write_text(json.dumps({"path": "../image.png"}) + "\n")
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(manifest, root)


def test_duplicate_image_bytes_in_different_paths_rejected(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    image = Image.fromarray(np.full((8, 8, 3), 80, dtype=np.uint8))
    image.save(root / "a.png")
    image.save(root / "b.png")
    (root / "cal.jsonl").write_text('{"path":"a.png","label":1}\n')
    (root / "test.jsonl").write_text('{"path":"b.png","label":1}\n')
    cal = load_manifest(root / "cal.jsonl", root)
    test = load_manifest(root / "test.jsonl", root)
    with pytest.raises(ValueError, match="content"):
        validate_disjoint(cal, test)


def test_invalid_label_rejected(tmp_path: Path):
    paths = make_smoke_dataset(tmp_path)
    manifest = Path(paths["test"])
    record = json.loads(manifest.read_text().splitlines()[0])
    record["label"] = -1
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="label"):
        load_manifest(manifest, tmp_path)
