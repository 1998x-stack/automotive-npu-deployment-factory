"""Full 3 x 3 x 2 Experiment-01 quantization ablation matrix and holdout protocol."""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from .data import load_manifest, validate_disjoint
from .metrics import classification
from .pipeline import _session, benchmark, evaluate_pair, export_mobilenet, quantize_onnx
from .utils import environment, sha256, write_json
from .visualize import generate_report


def stratified_calibration_subset(
    input_manifest: Path, dataset_root: Path, count: int, seed: int, output: Path
) -> Path:
    examples = load_manifest(input_manifest, dataset_root)
    if count < 1 or count > len(examples):
        raise ValueError(f"Calibration size {count} outside 1..{len(examples)}")
    groups: dict[int, list] = defaultdict(list)
    for row in examples:
        if row.label is None:
            raise ValueError("Stratified calibration requires labels")
        groups[row.label].append(row)
    rng = random.Random(seed)
    for items in groups.values():
        items.sort(key=lambda e: str(e.path))
        rng.shuffle(items)
    labels = sorted(groups)
    chosen = []
    while len(chosen) < count:
        progressed = False
        for label in labels:
            if groups[label] and len(chosen) < count:
                chosen.append(groups[label].pop())
                progressed = True
        if not progressed:
            break
    if len(chosen) != count:
        raise RuntimeError("Stratified selection could not meet requested count")
    root = Path(dataset_root).resolve()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in sorted(chosen, key=lambda e: str(e.path)):
            stream.write(json.dumps({
                "path": str(row.path.relative_to(root)),
                "label": row.label, "source": "synthetic" if row.synthetic else "real"
            }) + "\n")
    return output


def evaluate_single(model_path: Path, manifest: Path, data_root: Path) -> dict:
    from .data import preprocess

    images = load_manifest(manifest, data_root)
    if any(e.synthetic or e.label is None for e in images):
        raise ValueError("Real labeled data required for scientific baseline accuracy")
    session = _session(model_path)
    name = session.get_inputs()[0].name
    output_scores = [
        session.run(None, {name: preprocess(image.path)})[0].reshape(-1)
        for image in images
    ]
    return classification(output_scores, [int(e.label) for e in images if e.label is not None])


def run_ablation_matrix(
    data_root: Path,
    calibration: Path,
    validation: Path,
    test: Path,
    output_root: Path,
    *,
    calibration_sizes: tuple[int, ...] = (32, 64, 128),
    methods: tuple[str, ...] = ("minmax", "entropy", "percentile"),
    per_channel_options: tuple[bool, ...] = (False, True),
    seed: int = 2026,
    warmup: int = 2,
    repeats: int = 20,
    pretrained: bool = True,
    continue_on_error: bool = True,
) -> dict:
    """Select on validation only; evaluate sealed test split once for the winner."""
    data_root = Path(data_root)
    calibration_rows = load_manifest(calibration, data_root)
    validation_rows = load_manifest(validation, data_root)
    test_rows = load_manifest(test, data_root)
    validate_disjoint(calibration_rows, validation_rows)
    validate_disjoint(calibration_rows, test_rows)
    validate_disjoint(validation_rows, test_rows)
    if not pretrained:
        raise ValueError("Ablation quality comparisons require a pretrained classifier")
    if any(e.synthetic or e.label is None for e in calibration_rows + validation_rows + test_rows):
        raise ValueError("Real labeled images only for scientific ablation results")
    if not calibration_sizes or max(calibration_sizes) > len(calibration_rows):
        raise ValueError("Calibration matrix exceeds available samples")
    if min(calibration_sizes) <= 0:
        raise ValueError("Calibration sizes must be positive")
    if not methods or any(m not in ("minmax", "entropy", "percentile") for m in methods):
        raise ValueError("Invalid calibration method matrix")
    if not per_channel_options:
        raise ValueError("No per-channel matrix options")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = output_root / "models" / "mobilenet_v2.fp32.onnx"
    export_info = export_mobilenet(model, pretrained=True, seed=seed)
    from torchvision.models import MobileNet_V2_Weights

    from .imagenette import verify_imagenet_classes
    verify_imagenet_classes(MobileNet_V2_Weights.DEFAULT.meta["categories"])
    baseline_validation = evaluate_single(model, validation, data_root)
    baseline_bench = benchmark(
        model, validation, data_root, output_root / "baseline_benchmark.json",
        warmup=warmup, repeats=repeats
    )

    summary = {
        "experiment_id": "experiment-01-imagenette-ptq-ablation",
        "status": "running", "seed": seed,
        "dataset_scope": "Imagenette-160 real held-out images (10 of ImageNet-1K classes)",
        "hardware": "GitHub Actions or local x86 CPU; NOT Qualcomm HTP",
        "methodology": "validation model selection; test evaluated once after selection",
        "calibration_sha256": sha256(calibration),
        "validation_sha256": sha256(validation),
        "test_sha256": sha256(test),
        "model_sha256": export_info["model_sha256"],
        "environment": environment(),
        "baseline": {
            "validation_top1": baseline_validation["top1"],
            "validation_top5": baseline_validation["top5"],
            "cpu_p50_ms": baseline_bench["latency"]["p50_ms"],
            "cpu_p95_ms": baseline_bench["latency"]["p95_ms"],
        },
        "experiments": [], "selected_id": None, "final_test": None
    }
    save_path = output_root / "summary.json"
    write_json(save_path, summary)

    for count in calibration_sizes:
        manifest = stratified_calibration_subset(
            calibration, data_root, count, seed, output_root / "manifests" / f"cal_{count}.jsonl"
        )
        for method in methods:
            for per_channel in per_channel_options:
                suffix = "channel" if per_channel else "tensor"
                run_id = f"cal{count}-{method}-{suffix}"
                output = output_root / "models" / f"{run_id}.qdq.onnx"
                row = {
                    "id": run_id, "calibration_size": count, "method": method,
                    "per_channel": per_channel, "status": "running",
                    "calibration_manifest_sha256": sha256(manifest),
                }
                summary["experiments"].append(row)
                write_json(save_path, summary)
                try:
                    started = time.perf_counter()
                    quant_report = quantize_onnx(
                        model, manifest, test, data_root, output,
                        method=method, per_channel=per_channel
                    )
                    quant_elapsed = time.perf_counter() - started
                    evaluation = evaluate_pair(
                        model, output, validation, data_root,
                        output_root / "metrics" / f"{run_id}.validation.json"
                    )
                    bench = benchmark(
                        output, validation, data_root,
                        output_root / "metrics" / f"{run_id}.cpu.json",
                        warmup=warmup, repeats=repeats
                    )
                    quant_accuracy = evaluation.get("quant_accuracy")
                    if quant_accuracy is None:
                        raise RuntimeError("Missing ground-truth accuracy; refusing selection")
                    row.update({
                        "status": "completed",
                        "validation_top1": quant_accuracy["top1"],
                        "validation_top5": quant_accuracy["top5"],
                        "top1_drop_pp": evaluation["top1_drop_percentage_points"],
                        "agreement_top1": evaluation["agreement_top1"],
                        "max_absolute_error": evaluation["max_absolute_error"],
                        "model_size_mb": output.stat().st_size / 1e6,
                        "cpu_p50_ms": bench["latency"]["p50_ms"],
                        "cpu_p95_ms": bench["latency"]["p95_ms"],
                        "quantization_seconds": quant_elapsed,
                        "model_sha256": quant_report["output_sha256"],
                    })
                except Exception as exc:
                    row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                    if not continue_on_error:
                        write_json(save_path, summary)
                        raise
                write_json(save_path, summary)

    completed = [r for r in summary["experiments"] if r["status"] == "completed"]
    if completed:
        # Select on VAL accuracy only, then smaller validation error, then model size.
        best = sorted(completed, key=lambda r: (
            -r["validation_top1"], r["max_absolute_error"], r["model_size_mb"]
        ))[0]
        summary["selected_id"] = best["id"]
        chosen = output_root / "models" / f"{best['id']}.qdq.onnx"
        # FINAL test is never touched during per-variant comparison.
        selected_test = evaluate_pair(
            model, chosen, test, data_root, output_root / "metrics" / "selected.test.json"
        )
        float_test = selected_test["float_accuracy"]
        quant_test = selected_test["quant_accuracy"]
        summary["final_test"] = {
            "float_top1": float_test["top1"], "float_top5": float_test["top5"],
            "quant_top1": quant_test["top1"], "quant_top5": quant_test["top5"],
            "top1_drop_pp": selected_test["top1_drop_percentage_points"],
            "sample_count": selected_test["sample_count"], "selection_not_based_on_test": True
        }
    summary["status"] = "complete" if len(completed) == len(summary["experiments"]) else (
        "partial" if completed else "failed"
    )
    write_json(save_path, summary)
    if completed:
        generate_report(summary, output_root / "report")
    return summary
