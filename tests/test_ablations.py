import json
from pathlib import Path

import pytest

from npu_factory.ablations import stratified_calibration_subset
from npu_factory.data import load_manifest, make_smoke_dataset


def test_stratified_subset_deterministic(tmp_path: Path):
    paths = make_smoke_dataset(tmp_path / "data", count=12, seed=4)
    src = Path(paths["calibration"])
    root = tmp_path / "data"
    a = stratified_calibration_subset(src, root, 4, 77, tmp_path / "a.jsonl")
    b = stratified_calibration_subset(src, root, 4, 77, tmp_path / "b.jsonl")
    assert a.read_bytes() == b.read_bytes()
    assert len(load_manifest(a, root)) == 4
    with pytest.raises(ValueError, match="outside"):
        stratified_calibration_subset(src, root, 999, 77, tmp_path / "c.jsonl")


def test_stratified_preserves_synthetic_tag(tmp_path: Path):
    root = tmp_path / "d"
    paths = make_smoke_dataset(root, count=10)
    output = stratified_calibration_subset(
        Path(paths["calibration"]), root, 3, 2026, tmp_path / "subset.jsonl"
    )
    examples = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(e["source"] == "synthetic" for e in examples)
