"""ONNX FP16 conversion. Host CPU evaluation is numerical, not an FP16 speed claim."""
from __future__ import annotations

from pathlib import Path

import onnx
from onnxconverter_common import float16

from .utils import sha256


def export_fp16_onnx(fp32_path: Path, output_path: Path) -> dict:
    original = onnx.load(str(fp32_path))
    onnx.checker.check_model(original)
    converted = float16.convert_float_to_float16(
        original, keep_io_types=True, disable_shape_infer=False,
    )
    onnx.checker.check_model(converted)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(output_path))
    return {
        "status": "converted_not_verified_on_htp",
        "dtype": "float16_internal_float32_io",
        "original_sha256": sha256(fp32_path),
        "fp16_sha256": sha256(output_path),
        "original_size_mb": fp32_path.stat().st_size / 1e6,
        "fp16_size_mb": output_path.stat().st_size / 1e6,
        "important": "ONNX FP16 checker success does not prove HTP support or acceleration",
    }
