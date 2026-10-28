# Experiment 01 — CNN Quantization and QNN Deployment Factory

**Scope:** reproducible MobileNetV2 export, independent PTQ ablations, QNN conversion, genuine-device verification, and runtime validation. This is a teaching and engineering lab, not a vehicle-production release.

## Stage map and evidence gates

| Stage | Work package | Concrete deliverables | Objective acceptance |
|---|---|---|---|
| S00 | Python env, dataset contracts | pyproject, smoke fixtures, JSONL manifests, SHA | pytest and smoke CLI pass; calibration/test paths **and bytes** disjoint |
| S01 | PyTorch -> ONNX | model.onnx, export JSON, parity metrics | ONNX checker pass, deterministic seed, allclose at fixed tolerance; random weights **not** used to claim accuracy |
| S02 | Static PTQ ablation | QDQ ONNX, dataset records, paired eval | independent labeled test set, Top1/Top5 and top1 drop, quantization method and per-channel flag recorded |
| S03 | Native QNN converter | raw calibration input list, auditable QAIRT command plan | manually verified SDK version/flags, successful converter and model-lib logs, supported operator report |
| S04 | SA8295P/SA8255P target profiling | provider/device identity, target artifact, performance evidence | real hardware or explicitly tagged Proxy; no invented HTP result; unsupported target marked blocked |
| S05 | Performance optimization | timings, memory/power/thermal context, ablation comparison | repeatable warmup, P50/P95/P99 and concurrency context; control device and model revisions |
| S06 | Android/Linux service | versioned model registry + service contract | bounded timeouts, ACL, model-hash verification, crash recovery, end-to-end integration test on approved target |
| S07 | Advanced model extension | optional ASR/VLM/LLM research memo | model-specific objective metrics (WER/mAP/TTFT/TPS); **outside Experiment 01 CNN implementation** |
| S08 | Engineering quality | CI, provenance JSON, acceptance report | unit/lint pipeline passes, model/data/SDK/BSP provenance, no tokens or proprietary blobs in git |

## Status truth table

- S00: code and unit contracts implemented; CI result must be checked on PR.
- S01: ONNX export and parity **implemented**, needs environment with torch/torchvision/onnx/onnxruntime and optional pretrained weights.
- S02: QDQ PTQ and paired evaluation **implemented**, quality results require a labeled independent real dataset.
- S03: raw-image writer and safe QNN command **plan implemented**; converter execution pending a compatible licensed QAIRT SDK.
- S04: design-only, gated on compatible real device and access.
- S05: CPU model-only timing implemented; genuine HTP/thermal/power measurements pending hardware.
- S06: interface and acceptance criteria documented; native Android/board implementation deferred.
- S07: future, separate advanced-model experiments.
- S08: unit test/CI and source-control evidence implemented; build/release certification pending.

Never mark a stage complete because its design has been written. The signed-off evidence must be inspected.

## Experiment matrix

Freeze model weights, label vocabulary, preprocessing, and test set. Vary one primary factor at a time:

1. Calibration sample count: 32, 128, 512, all sampled **only** from calibration split.
2. Calibration method: MinMax, Entropy, Percentile.
3. Per-channel vs per-tensor weight quantization.
4. FP32 ONNX vs U8/S8 QDQ on the same holdout.
5. ORT CPU vs QNN HTP only with fully qualified target metadata.
6. Cold vs warm load, isolated vs concurrent inference, stable vs thermal-throttled runs.

Use at least three deterministic seeds for stochastic data selection when valid real data is available. Aggregate mean, spread and individual observations; no fabricated significance.

## Fail-closed rules

- Device- or SDK-specific commands must not be automatically executed by the model or CI.
- Generated **qnn-plan** is documentation, not a successful compilation report.
- **smoke-data** creates meaningless synthetic labels; it is only for smoke and correctness wiring.
- No model accuracy based on synthetic random images.
- Do not call a CPU benchmark HTP without a real HTP provider and captured proof.
- HTP library, DSP skeleton, binaries and private BSP data stay outside the Git repository.
- A context binary must be compiled/validated for the exact SDK/firmware/SoC combination.

## Acceptance matrix by role

| Actor | Inputs | Exit gate |
|---|---|---|
| Dataset owner | licensed images, ImageNet mapping | labels audited; calibration and test disjoint |
| ML engineer | model, preprocessing, seed | ONNX parity + accuracy report |
| Quantization engineer | PTQ config, representative images | quantization table with independent holdout |
| Runtime engineer | QAIRT/ORT, compilation logs | operator support and correct backend |
| Board engineer | SoC, firmware, DSP libs | signed-off actual device benchmark |
| Verification engineer | benchmark config, fault scenarios | evidence matrix and reproducibility check |
