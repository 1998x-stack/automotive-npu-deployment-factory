# Experiment 01 — Executed Imagenette PTQ ablations (2026-10-28)

**Status:** 18/18 complete; GitHub Actions [run 40666390664](https://github.com/1998x-stack/automotive-npu-deployment-factory/actions/runs/40666390664) completed successfully. Original detailed evidence is in the run's `exp01-imagenette-ablation-reports` artifact, containing `summary.json`, `ablation_results.csv`, 5 PNG charts, HTML report and per-variant JSON. Keep artifact ZIP for permanent archival beyond GitHub's 30-day retention.

## Dataset and environment

- MobileNetV2 `IMAGENET1K_V2` pretrained weights, ONNX FP32 baseline, static U8/S8 QDQ PTQ.
- Imagenette2-160: 240 calibration, 100 validation, 100 independent final test images; ten ImageNet-1K categories.
- Source Imagenette archive SHA256 begins `64d0c4859f35`; FP32 ONNX SHA256 begins `35988b419ac5`.
- Ubuntu 22.04 GitHub Actions x86_64 CPU, Python 3.11.17; **not** Qualcomm NPU.
- Explicit model/input split provenance, separated validation-only model selection and sealed final holdout.

## Complete validation results

Accuracy below is **Imagenette validation set Top-1** (100 real images per evaluation). P95 latency is single-run ONNX Runtime CPU model-only timing, not a robust device performance guarantee.

| Calibration | Method | Weight quantization | Top-1 | QDQ model MB | CPU P95 ms |
|---:|---|---|---:|---:|---:|
| 32 | MinMax | Per-tensor | 51% | 3.67 | 2.17 |
| 32 | MinMax | Per-channel | 60% | 3.90 | 2.25 |
| 32 | Entropy | Per-tensor | 51% | 3.67 | 2.33 |
| 32 | Entropy | Per-channel | 60% | 3.90 | 2.25 |
| 32 | Percentile | Per-tensor | 51% | 3.67 | 2.21 |
| 32 | Percentile | Per-channel | 60% | 3.90 | 2.09 |
| 64 | MinMax | Per-tensor | 51% | 3.67 | 1.93 |
| **64** | **MinMax** | **Per-channel** | **63%** | **3.90** | **2.21** |
| 64 | Entropy | Per-tensor | 51% | 3.67 | 2.17 |
| 64 | Entropy | Per-channel | 63% | 3.90 | 2.27 |
| 64 | Percentile | Per-tensor | 51% | 3.67 | 2.17 |
| 64 | Percentile | Per-channel | 61% | 3.90 | 2.28 |
| 128 | MinMax | Per-tensor | 51% | 3.67 | 2.15 |
| 128 | MinMax | Per-channel | 63% | 3.90 | 2.02 |
| 128 | Entropy | Per-tensor | 51% | 3.67 | 2.13 |
| 128 | Entropy | Per-channel | 63% | 3.90 | 2.98 |
| 128 | Percentile | Per-tensor | 51% | 3.67 | 1.92 |
| 128 | Percentile | Per-channel | 63% | 3.90 | 2.50 |

## Selection and final holdout

- FP32 **validation**: Top-1 **63%**, Top-5 **85%**; CPU P50 **3.52ms**, CPU P95 **4.00ms**.
- Selected based **only on validation**: `cal64-minmax-channel`.
- Selected INT8 validation Top-1: **63%**, matching baseline on 100 validation images.
- Independent **final test** (100 images): FP32 **61% Top-1 / 88% Top-5** vs selected INT8 **60% Top-1 / 86% Top-5**, differences **-1/-2 percentage points**.
- Selected QDQ model size ~**3.90MB**; CPU P50 ~**1.97ms**. These are host CPU measurements only.

## Main observations and limitations

1. On this sample, all nine per-tensor variants reached 51% validation Top-1; per-channel variants reached 60–63%. This is evidence that channel-wise weight quantization matters **for this model and sampled images**, not a universal result.
2. At 64 calibration examples the best per-channel accuracy had already reached 63%. Larger calibration sets did not increase observed Top-1 in this single run.
3. Entropy/Percentile were slower to calibrate and did not outperform the highest MinMax validation Top-1 in this matrix. The study does **not** prove they are worse in other domains.
4. The final Top-1 difference corresponds to **one of 100 examples**; uncertainties are substantial and no formal significance claim is made.
5. No SA8255P/SA8295P NPU execution, QNN native compile, automotive camera data, thermal, memory or power result is included.
6. Re-run with additional seeds, larger holdouts, authentic automotive images and target device for stronger conclusions. The ORT QDQ model is not yet a compiled QNN HTP artifact.

## Reproduction

See [full protocol](../stages/exp01-ablation.md) and [full ablation workflow](../../.github/workflows/full-ablation.yml). Model weights, image data and proprietary SDK binaries are intentionally not committed.
