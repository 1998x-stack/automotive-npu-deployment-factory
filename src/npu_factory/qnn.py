"""QNN offline command planning; no implicit execution or simulated NPU measurements."""
from __future__ import annotations

import shlex
from pathlib import Path

from .utils import required_file


def create_qnn_plan(
    model: Path, input_list: Path, sdk_root: Path, output_dir: Path
) -> dict:
    model = required_file(model)
    input_list = required_file(input_list)
    if not input_list.read_text(encoding="utf-8").strip():
        raise ValueError("QNN calibration input list is empty")
    sdk_root = Path(sdk_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    bin_dir = sdk_root / "bin" / "x86_64-linux-clang"
    converter = bin_dir / "qnn-onnx-converter"
    generator = bin_dir / "qnn-model-lib-generator"
    cpp = output_dir / "model.cpp"
    weights = output_dir / "model.bin"
    libraries = output_dir / "model_libs"
    cmds = [
        [str(converter), "--input_network", str(model), "--input_list",
         str(input_list), "--output_path", str(cpp)],
        [str(generator), "-c", str(cpp), "-b", str(weights), "-o", str(libraries)]
    ]
    return {
        "status": "planned_not_executed",
        "backend": "qnn_offline_converter",
        "precision_intent": "PTQ via calibration input list; inspect converter report",
        "sdk_root": str(sdk_root),
        "commands": cmds,
        "shell_preview": [shlex.join(cmd) for cmd in cmds],
        "prerequisites": [
            "QAIRT licensed/installed and compatible with target BSP",
            "SDK command flags checked against installed version",
            "QNN input name, dtype and tensor layout confirmed",
            "real representative calibration data, never test images",
            "output model.bin existence checked before model-lib-generator",
            "HTP context compilation performed separately for each target SoC",
        ],
    }
