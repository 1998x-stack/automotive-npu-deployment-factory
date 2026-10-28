"""Experiment-01 executable baseline: PyTorch -> ONNX -> PTQ -> evaluation."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data import load_manifest, preprocess, validate_disjoint
from .metrics import classification, latency_stats, numerical_difference
from .utils import environment, required_file, sha256, write_json


def export_mobilenet(output: Path, *, pretrained: bool = False, seed: int = 2026) -> dict:
    """No pretrained download unless the caller explicitly requests it."""
    import onnx
    import onnxruntime as ort
    import torch
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    torch.manual_seed(seed)
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights).eval()
    example = torch.randn(1, 3, 224, 224)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        expected = model(example).cpu().numpy()
        torch.onnx.export(
            model, (example,), str(output), opset_version=17, dynamo=False,
            input_names=["image"], output_names=["logits"], do_constant_folding=True
        )

    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": example.cpu().numpy()})[0]
    diff = numerical_difference(expected, actual)
    # Absolute tolerance deliberately specified; match randomized and trained weights.
    if not np.allclose(expected, actual, atol=1e-4, rtol=1e-3):
        raise RuntimeError(f"Export parity gate failed: {diff}")

    report = {
        "stage": "S01", "status": "verified_locally",
        "model": "torchvision_mobilenet_v2",
        "weights": "ImageNet1K_V2" if pretrained else "random_for_smoke_only",
        "seed": seed, "opset": 17, "input": [1, 3, 224, 224],
        "numerical_parity": diff,
        "model_sha256": sha256(output), "environment": environment(),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }
    write_json(output.with_suffix(".export.json"), report)
    return report


def _session(path: Path, *, provider: str = "CPUExecutionProvider", backend: str = ""):
    import onnxruntime as ort

    if provider not in ("CPUExecutionProvider", "QNNExecutionProvider"):
        raise ValueError(f"Unsupported provider: {provider}")
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"{provider} unavailable in this ONNX Runtime build: {available}")
    opts = ort.SessionOptions()
    providers: list = [provider]
    if provider == "QNNExecutionProvider":
        # Fail closed if any operator would execute on the CPU.
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if backend not in ("htp", "cpu", "gpu"):
            raise ValueError("QNN requires explicit backend: htp, cpu or gpu")
        providers = [(provider, {"backend_type": backend})]
    session = ort.InferenceSession(str(required_file(path)), sess_options=opts, providers=providers)
    if provider not in session.get_providers():
        raise RuntimeError(f"Requested provider was not registered: {session.get_providers()}")
    return session


def quantize_onnx(
    model: Path, calibration: Path, test: Path, data_root: Path, output: Path,
    *, method: str = "minmax", per_channel: bool = True
) -> dict:
    """Static QDQ PTQ. Disjoint split mandatory; no claim of hardware compatibility."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    calibration_images = load_manifest(calibration, data_root)
    test_images = load_manifest(test, data_root)
    validate_disjoint(calibration_images, test_images)
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if method not in methods:
        raise ValueError(f"Unknown calibration method: {method}")
    model = required_file(model)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    session = _session(model)
    names = [i.name for i in session.get_inputs()]
    if len(names) != 1:
        raise ValueError("Experiment 01 supports exactly one image input")
    input_name = names[0]

    class ImageReader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(calibration_images)

        def get_next(self):
            example = next(self._iter, None)
            return None if example is None else {input_name: preprocess(example.path)}

        def rewind(self):
            self._iter = iter(calibration_images)

    quantize_static(
        model_input=str(model), model_output=str(output),
        calibration_data_reader=ImageReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=methods[method], per_channel=per_channel,
    )
    import onnx
    onnx.checker.check_model(str(output))
    report = {
        "stage": "S02", "status": "compiled_and_onnx_validated",
        "precision": "QDQ U8 activations / S8 weights",
        "method": method, "per_channel": per_channel,
        "model_sha256": sha256(model),
        "output_sha256": sha256(output),
        "calibration_manifest_sha256": sha256(calibration),
        "test_manifest_sha256": sha256(test),
        "calibration_count": len(calibration_images),
        "test_count": len(test_images),
        "synthetic_calibration": any(x.synthetic for x in calibration_images),
        "hardware_qnn_compatibility": "NOT_VERIFIED",
        "environment": environment(),
    }
    write_json(output.with_suffix(".quantization.json"), report)
    return report


def evaluate_pair(
    float_model: Path, quant_model: Path, test: Path, data_root: Path, report_path: Path
) -> dict:
    """All samples evaluated; synthetic labels never treated as scientific accuracy."""
    images = load_manifest(test, data_root)
    session_a = _session(float_model)
    session_b = _session(quant_model)
    name_a = session_a.get_inputs()[0].name
    name_b = session_b.get_inputs()[0].name
    scores_a, scores_b, diffs = [], [], []
    for image in images:
        x = preprocess(image.path)
        a = session_a.run(None, {name_a: x})[0]
        b = session_b.run(None, {name_b: x})[0]
        scores_a.append(a.reshape(-1))
        scores_b.append(b.reshape(-1))
        diffs.append(numerical_difference(a, b))

    synthetic = any(x.synthetic for x in images)
    labeled = all(x.label is not None for x in images)
    scientifically_valid = not synthetic and labeled
    result = {
        "stage": "S02", "status": "evaluated",
        "dataset_kind": "synthetic_smoke" if synthetic else "provided",
        "real_accuracy_eligible": scientifically_valid,
        "sample_count": len(images),
        "agreement_top1": sum(d["top1_equal"] for d in diffs) / len(diffs),
        "max_absolute_error": max(d["max_absolute_error"] for d in diffs),
        "mean_absolute_error": float(np.mean([d["mean_absolute_error"] for d in diffs])),
        "float_model_sha256": sha256(float_model),
        "quant_model_sha256": sha256(quant_model),
        "test_manifest_sha256": sha256(test),
        "environment": environment(),
    }
    if scientifically_valid:
        labels = [int(x.label) for x in images if x.label is not None]
        result["float_accuracy"] = classification(scores_a, labels)
        result["quant_accuracy"] = classification(scores_b, labels)
        result["top1_drop_percentage_points"] = (
            result["float_accuracy"]["top1"] - result["quant_accuracy"]["top1"]
        ) * 100
    else:
        result["accuracy_note"] = (
            "No accuracy claim: synthetic samples or missing ground-truth labels"
        )
    write_json(report_path, result)
    return result


def benchmark(
    model: Path, test: Path, data_root: Path, report_path: Path,
    *, provider: str = "CPUExecutionProvider", backend: str = "",
    warmup: int = 10, repeats: int = 100
) -> dict:
    """Measures session.run only, excluding image preprocessing and initialization."""
    if warmup < 0 or repeats < 10:
        raise ValueError("warmup >= 0 and repeats >= 10 required")
    session = _session(model, provider=provider, backend=backend)
    name = session.get_inputs()[0].name
    x = preprocess(load_manifest(test, data_root)[0].path)
    for _ in range(warmup):
        session.run(None, {name: x})
    timings = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        session.run(None, {name: x})
        timings.append((time.perf_counter_ns() - started) / 1e6)
    result = {
        "stage": "S05",
        "status": "measured",
        "precision": "as_encoded_in_model",
        "provider": provider,
        "backend": backend or "host_cpu",
        "device": "local; fill in platform identity from target logs",
        "scope": "model_only_session_run_excludes_preprocess_and_session_init",
        "warmup": warmup,
        "latency": latency_stats(timings),
        "model_sha256": sha256(model),
        "environment": environment(),
    }
    write_json(report_path, result)
    return result
