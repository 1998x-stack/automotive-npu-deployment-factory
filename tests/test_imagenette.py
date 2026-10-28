import tarfile

import pytest

from npu_factory.imagenette import LABELS, _safe_archive_member, verify_imagenet_classes


def test_imagenette_imagenet_mapping_validated():
    categories = ["unused"] * 1000
    for label_id, label in LABELS.values():
        categories[label_id] = label
    verify_imagenet_classes(categories)
    categories[217] = "wrong class"
    with pytest.raises(ValueError, match="mismatch"):
        verify_imagenet_classes(categories)


def test_archive_rejects_unsafe_and_unknown_paths():
    good = tarfile.TarInfo("imagenette2-160/train/n01440764/tench.JPEG")
    good.type = tarfile.REGTYPE
    assert _safe_archive_member(good) == ("train", "n01440764")
    for bad in (
        "../train/n01440764/tench.JPEG",
        "imagenette2-160/train/../../escape.jpg",
        "imagenette2-160/train/not_a_class/x.JPEG",
        "imagenette2-160/train/n01440764/malware.sh",
    ):
        member = tarfile.TarInfo(bad)
        member.type = tarfile.REGTYPE
        assert _safe_archive_member(member) is None
