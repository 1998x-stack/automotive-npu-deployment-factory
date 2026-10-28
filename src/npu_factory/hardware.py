"""Hardware evidence contract: records intended experiments without pretending to run them."""
from __future__ import annotations

from pathlib import Path

from .utils import write_json

REQUIRED_PHYSICAL_FIELDS = (
    "target_soc", "device_serial_redacted", "runtime_provider",
    "qairt_version", "bsp_version", "dsp_firmware_version",
    "htp_architecture", "model_sha256", "raw_device_log_sha256",
)


def hardware_preflight(
    out: Path, *, soc: str, sdk_root: Path | None = None,
    physical_device_attached: bool = False,
) -> dict:
    if soc.lower() not in ("sa8255p", "sa8295p"):
        raise ValueError("Only requested SA8255P/SA8295P supported by this worksheet")
    checks = {
        "authorized_qairt_sdk": bool(sdk_root and Path(sdk_root).is_dir()),
        "physical_target_confirmed_by_operator": physical_device_attached,
        "compatible_host_stub_and_dsp_skeleton": False,
        "runtime_bsp_compatibility_confirmed": False,
        "signed_device_log_and_profiler_results": False,
        "qnn_context_compiled_for_exact_soc": False,
    }
    result = {
        "stage": "S04-S05",
        "soc": soc.upper(),
        "status": "blocked_unverified",
        "preconditions": checks,
        "required_provenance": list(REQUIRED_PHYSICAL_FIELDS),
        "warnings": [
            "Host ONNX CPU and quantization results are not NPU timing",
            "Context Binary and Stub/Skeleton must match exact SoC, SDK and BSP",
            "AI Hub Proxy is not a measured physical SA8255P/SA8295P result",
            "A physical-device claim additionally needs independent inspection of raw logs",
            "Do not upload proprietary SDKs, OEM firmware, BSP blobs, or credentials"
        ],
    }
    write_json(out, result)
    return result
