"""Opt-in, class-balanced Imagenette subset with ImageNet-1K label alignment.

Data remains outside Git. Source: fast.ai Imagenette2-160 (derived from ImageNet).
Users must comply with the dataset's applicable license/terms.
"""
from __future__ import annotations

import json
import random
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from .data import load_manifest, validate_disjoint
from .utils import sha256, write_json

# Official Imagenette synsets mapped to torchvision ImageNet-1K zero-based indices.
# Fail closed on pretrained classifier vocab changes.
LABELS: dict[str, tuple[int, str]] = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}
DEFAULT_SOURCE = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def verify_imagenet_classes(categories: list[str]) -> None:
    if len(categories) != 1000:
        raise ValueError("Expected a 1000-class ImageNet classifier")
    for _, (index, name) in LABELS.items():
        if categories[index].casefold() != name.casefold():
            raise ValueError(
                f"ImageNet class mapping mismatch at {index}: "
                f"{categories[index]!r} != {name!r}"
            )


def _safe_archive_member(member: tarfile.TarInfo) -> tuple[str, str] | None:
    """Select only files in two known dataset splits and ten allowed synsets."""
    parts = Path(member.name).parts
    if (not member.isfile() or len(parts) != 4 or parts[0] != "imagenette2-160"
            or parts[1] not in ("train", "val") or parts[2] not in LABELS
            or Path(parts[3]).suffix.lower() not in (".jpg", ".jpeg")):
        return None
    return parts[1], parts[2]


def _download(archive: Path, url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "s3.amazonaws.com":
        raise ValueError("Dataset URL must be the approved S3 Imagenette origin")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        return
    temporary = archive.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=90) as source, temporary.open("wb") as dest:
            size = 0
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > 400 * 1024 * 1024:
                    raise ValueError("Dataset archive exceeds size limit")
                dest.write(chunk)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_imagenette(
    root: Path,
    *,
    calibration_per_class: int = 24,
    validation_per_class: int = 10,
    test_per_class: int = 10,
    seed: int = 2026,
    url: str = DEFAULT_SOURCE,
) -> dict:
    """Download and extract only selected samples. Never overlap dataset splits."""
    if min(calibration_per_class, validation_per_class, test_per_class) < 1:
        raise ValueError("Every split must have at least one sample per class")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenette2-160.tgz"
    _download(archive, url)
    grouping: dict[tuple[str, str], list[tarfile.TarInfo]] = defaultdict(list)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            kind = _safe_archive_member(member)
            if kind:
                grouping[kind].append(member)

        selected: dict[str, list[tuple[tarfile.TarInfo, str]]] = {
            "calibration": [], "validation": [], "test": []
        }
        for synset in LABELS:
            for src_split in ("train", "val"):
                members = sorted(grouping[(src_split, synset)], key=lambda x: x.name)
                rng = random.Random(f"{seed}:{src_split}:{synset}")
                rng.shuffle(members)
                required = calibration_per_class if src_split == "train" else (
                    validation_per_class + test_per_class
                )
                if len(members) < required:
                    raise RuntimeError(f"Not enough {src_split}/{synset} images")
                if src_split == "train":
                    selected["calibration"].extend((m, synset) for m in members[:required])
                else:
                    selected["validation"].extend(
                        (m, synset) for m in members[:validation_per_class]
                    )
                    selected["test"].extend(
                        (m, synset) for m in members[
                            validation_per_class:validation_per_class + test_per_class
                        ]
                    )

        manifest_paths = {}
        for split, examples in selected.items():
            records = []
            for member, synset in sorted(examples, key=lambda pair: pair[0].name):
                relative = Path("images") / split / synset / Path(member.name).name
                absolute = root / relative
                absolute.parent.mkdir(exist_ok=True, parents=True)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Archive image unreadable: {member.name}")
                if member.size > 10 * 1024 * 1024:
                    raise ValueError(f"Oversized image: {member.name}")
                absolute.write_bytes(source.read())
                records.append(json.dumps({
                    "path": relative.as_posix(), "label": LABELS[synset][0],
                    "source": "imagenette_real"
                }))
            manifest = root / f"{split}.jsonl"
            manifest.write_text("\n".join(records) + "\n", encoding="utf-8")
            manifest_paths[split] = str(manifest.resolve())

    calibration = load_manifest(Path(manifest_paths["calibration"]), root)
    validation = load_manifest(Path(manifest_paths["validation"]), root)
    test = load_manifest(Path(manifest_paths["test"]), root)
    validate_disjoint(calibration, validation)
    validate_disjoint(calibration, test)
    validate_disjoint(validation, test)
    result = {
        "dataset": "imagenette2-160", "archive_source": url,
        "archive_sha256": sha256(archive), "seed": seed,
        "class_map": {key: index for key, (index, _) in LABELS.items()},
        "splits": {key: len(items) for key, items in (
            ("calibration", calibration), ("validation", validation), ("test", test)
        )},
        "manifest_paths": manifest_paths,
        "manifest_sha256": {
            split: sha256(Path(path)) for split, path in manifest_paths.items()
        },
        "evaluation_scope": "imagenette_subset_not_full_imagenet_1000",
        "terms": "Imagenette uses ImageNet-derived images; comply with ImageNet terms.",
    }
    write_json(root / "dataset_provenance.json", result)
    return result
