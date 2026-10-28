"""Experiment 02: FP16 portability, matched classifier-head QAT, paired statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .ablations import stratified_calibration_subset
from .data import load_manifest, preprocess, validate_disjoint
from .fp16 import export_fp16_onnx
from .imagenette import verify_imagenet_classes
from .pipeline import _session, export_mobilenet, quantize_onnx
from .qat import extract_features, predict_head, train_head
from .statistics import paired_accuracy_evidence
from .utils import environment, sha256, write_json


def correct_from_predictions(predicted: list[int], labels: list[int]) -> list[bool]:
    if len(predicted) != len(labels):
        raise ValueError("Predictions and labels must have equal length")
    return [a == b for a, b in zip(predicted, labels)]


def prediction_onnx(model_path: Path, manifest: Path, root: Path) -> tuple[list[int], list[int]]:
    images = load_manifest(manifest, root)
    if any(x.label is None or x.synthetic for x in images):
        raise ValueError("Only labeled real images are eligible for evaluation")
    session = _session(model_path)
    input_name = session.get_inputs()[0].name
    predictions, labels = [], []
    for item in images:
        out = session.run(None, {input_name: preprocess(item.path)})[0]
        predictions.append(int(np.argmax(out)))
        labels.append(int(item.label))
    return predictions, labels


def _comparison(a: list[int], b: list[int], labels: list[int], seed: int) -> dict:
    return paired_accuracy_evidence(
        correct_from_predictions(a, labels), correct_from_predictions(b, labels),
        seed=seed, bootstrap_replicates=5000,
    )


def _render(summary: dict, root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = summary["qat_seeds"]
    names = [str(item["seed"]) for item in seed_rows]
    xs = np.arange(len(seed_rows))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(xs - 0.18, [x["matched_fp32_validation_accuracy"] * 100 for x in seed_rows],
           width=0.36, label="FP32 fine-tuned head")
    ax.bar(xs + 0.18, [x["qat_validation_accuracy"] * 100 for x in seed_rows],
           width=0.36, label="Fake-quant QAT head")
    ax.set_xticks(xs, labels=names)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("1000-way Top-1 on Imagenette validation (%)")
    ax.set_title("Matched head-only training: FP32 vs QAT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "qat_seed_validation.png", dpi=160)
    plt.close(fig)

    comparisons = summary["paired_comparisons"]
    labels = list(comparisons.keys())
    values = [comparisons[k]["candidate_minus_reference_pp"] for k in labels]
    lower = [
        values[i] - comparisons[k]["paired_bootstrap_low_pp"]
        for i, k in enumerate(labels)
    ]
    upper = [
        comparisons[k]["paired_bootstrap_high_pp"] - values[i]
        for i, k in enumerate(labels)
    ]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.errorbar(values, range(len(values)), xerr=[lower, upper], fmt="o", capsize=4)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Candidate minus reference: Top-1 percentage points (paired 95% CI)")
    ax.set_title("Independent holdout: paired bootstrap comparisons")
    fig.tight_layout()
    fig.savefig(fig_dir / "holdout_paired_confidence.png", dpi=160)
    plt.close(fig)

    rows = []
    for key, comp in comparisons.items():
        rows.append(
            "<tr><td>" + key + "</td><td>" +
            f"{comp['candidate_minus_reference_pp']:.2f}" + "</td><td>" +
            f"[{comp['paired_bootstrap_low_pp']:.2f}, "
            f"{comp['paired_bootstrap_high_pp']:.2f}]" + "</td><td>" +
            f"{comp['mcnemar_exact_two_sided_p']:.4f}" + "</td></tr>"
        )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Experiment 02 — FP16/QAT and robustness</title>"
        '<style>body{font:15px system-ui;max-width:1100px;margin:30px auto}'
        'img{width:95%;max-width:850px}td,th{border:1px solid #ccc;padding:8px}'
        'table{border-collapse:collapse}</style>'
        "<h1>Experiment 02: FP16, head-only QAT, paired robustness</h1>"
        "<p>Only Imagenette real data. Frozen MobileNetV2 FP32 backbone in QAT;"
        " not full-model INT8 or Qualcomm NPU performance.</p>"
        '<img src="figures/qat_seed_validation.png">'
        '<img src="figures/holdout_paired_confidence.png">'
        "<table><tr><th>Comparison</th><th>Delta pp</th>"
        "<th>Paired bootstrap 95% CI</th><th>McNemar exact p</th></tr>"
        + "".join(rows) + "</table>"
        "<p>For full data and provenance inspect summary.json.</p></html>"
    )
    (root / "report.html").write_text(html, encoding="utf-8")


def run_next_stage(
    data_root: Path,
    calibration: Path,
    validation: Path,
    test: Path,
    out: Path,
    *,
    training_seeds: tuple[int, ...] = (7, 19, 31),
    qat_images: int = 80,
    ptq_images: int = 64,
    epochs: int = 3,
    lr: float = 0.001,
    bootstrap_seed: int = 2026,
) -> dict:
    """Complete measured CPU study. No NPU/FP16 acceleration claim."""
    data_root = Path(data_root).resolve()
    cal_rows = load_manifest(calibration, data_root)
    val_rows = load_manifest(validation, data_root)
    test_rows = load_manifest(test, data_root)
    validate_disjoint(cal_rows, val_rows)
    validate_disjoint(cal_rows, test_rows)
    validate_disjoint(val_rows, test_rows)
    if any(x.synthetic or x.label is None for x in cal_rows + val_rows + test_rows):
        raise ValueError("Only independent real labeled data is allowed")
    if len(training_seeds) < 3 or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("At least three distinct training seeds are required")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    verify_imagenet_classes(MobileNet_V2_Weights.DEFAULT.meta["categories"])
    fp32 = out / "models" / "mobilenet_v2.fp32.onnx"
    export_meta = export_mobilenet(fp32, pretrained=True)

    fp16 = out / "models" / "mobilenet_v2.fp16.onnx"
    fp16_meta = export_fp16_onnx(fp32, fp16)
    # Fixed S01-selected PTQ recipe, not retuned on final test.
    ptq_manifest = stratified_calibration_subset(
        calibration, data_root, ptq_images, 2026, out / "manifests" / "ptq64.jsonl"
    )
    ptq = out / "models" / "mobilenet_v2.ptq.qdq.onnx"
    ptq_meta = quantize_onnx(fp32, ptq_manifest, test, data_root, ptq,
                             method="minmax", per_channel=True)

    pretrained = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    for parameter in pretrained.parameters():
        parameter.requires_grad_(False)
    head = pretrained.classifier[1]
    # For each seed, train only on calibration images; test features are not read
    # until the best QAT seed is selected on validation.
    features_val, labels_val = extract_features(pretrained.features, validation, data_root)
    results, candidates = [], {}
    for seed in training_seeds:
        manifest = stratified_calibration_subset(
            calibration, data_root, qat_images, seed,
            out / "manifests" / f"qat-{seed}.jsonl"
        )
        features_train, labels_train = extract_features(
            pretrained.features, manifest, data_root
        )
        fp32_head, fp32_loss = train_head(
            head, features_train, labels_train, quant_aware=False, seed=seed,
            epochs=epochs, learning_rate=lr
        )
        qat_head, qat_loss = train_head(
            head, features_train, labels_train, quant_aware=True, seed=seed,
            epochs=epochs, learning_rate=lr
        )
        fp32_val = predict_head(fp32_head, features_val)
        qat_val = predict_head(qat_head, features_val)
        row = {
            "seed": seed,
            "calibration_manifest_sha256": sha256(manifest),
            "matched_fp32_train_loss": fp32_loss,
            "qat_train_loss": qat_loss,
            "matched_fp32_validation_accuracy": float(np.mean(
                correct_from_predictions(fp32_val, labels_val.tolist())
            )),
            "qat_validation_accuracy": float(np.mean(
                correct_from_predictions(qat_val, labels_val.tolist())
            )),
            "qat_vs_fp32_validation_delta_pp": 100 * float(np.mean(
                correct_from_predictions(qat_val, labels_val.tolist())
            ) - np.mean(correct_from_predictions(fp32_val, labels_val.tolist()))),
        }
        candidates[seed] = (fp32_head, qat_head)
        results.append(row)

    selected = sorted(
        results, key=lambda r: (-r["qat_validation_accuracy"], r["seed"])
    )[0]["seed"]
    # Only now use the sealed final test set for all fixed models and selected QAT.
    fp32_pred, labels = prediction_onnx(fp32, test, data_root)
    fp16_pred, fp16_labels = prediction_onnx(fp16, test, data_root)
    int8_pred, ptq_labels = prediction_onnx(ptq, test, data_root)
    if labels != fp16_labels or labels != ptq_labels:
        raise RuntimeError("Unaligned final test predictions")
    test_features, test_labels = extract_features(pretrained.features, test, data_root)
    if labels != test_labels.tolist():
        raise RuntimeError("Torch and ONNX test labels differ")
    fp32_head, qat_head = candidates[selected]
    matched_fp32_pred = predict_head(fp32_head, test_features)
    qat_pred = predict_head(qat_head, test_features)

    comparisons = {
        "FP16 vs pretrained FP32": _comparison(fp32_pred, fp16_pred, labels, bootstrap_seed),
        "PTQ INT8 vs pretrained FP32": _comparison(
            fp32_pred, int8_pred, labels, bootstrap_seed + 1
        ),
        "Head QAT vs matched FP32 tune": _comparison(
            matched_fp32_pred, qat_pred, labels, bootstrap_seed + 2
        ),
    }
    accuracies = {
        "pretrained_fp32": float(np.mean(correct_from_predictions(fp32_pred, labels))),
        "onnx_fp16": float(np.mean(correct_from_predictions(fp16_pred, labels))),
        "ptq_int8": float(np.mean(correct_from_predictions(int8_pred, labels))),
        "head_fp32_finetuned": float(np.mean(
            correct_from_predictions(matched_fp32_pred, labels)
        )),
        "head_fakequant_qat": float(np.mean(correct_from_predictions(qat_pred, labels))),
    }
    summary = {
        "stage": "experiment-02",
        "status": "complete",
        "hardware": "x86 host CPU only; no HTP execution",
        "qat_scope": "last 1000-way linear classifier head; frozen FP32 backbone",
        "fp16_scope": "ONNX internal FP16, FP32 inputs/outputs; host CPU compatibility only",
        "data": {
            "calibration_manifest_sha256": sha256(calibration),
            "validation_manifest_sha256": sha256(validation),
            "test_manifest_sha256": sha256(test),
            "calibration_count": len(cal_rows),
            "validation_count": len(val_rows),
            "test_count": len(test_rows),
        },
        "models": {"fp32": export_meta, "fp16": fp16_meta, "ptq": ptq_meta},
        "qat_seeds": results,
        "selected_qat_seed_on_validation": selected,
        "final_test_accuracies": accuracies,
        "paired_comparisons": comparisons,
        "environment": {**environment(), "torch": torch.__version__},
        "limitations": [
            "QAT affects final linear layer only: no full-model fake-quant deployment claim",
            "Imagenette ten ImageNet classes only, not real automotive camera data",
            "No inference speedups inferred from FP16/INT8 file sizes or CPU timing",
            "Repeated training seeds do not replace independent image sampling"
        ],
    }
    write_json(out / "summary.json", summary)
    _render(summary, out)
    return summary
