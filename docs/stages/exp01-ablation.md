# Experiment 01 — full ablation protocol

## Research scope

For pretrained MobileNetV2 (ImageNet-1K weights), compare **3 calibration sizes × 3 methods × 2 per-channel options** on an independent Imagenette subset. This evaluates classification and host CPU performance, **not** actual Qualcomm NPU performance.

## Dataset and provenance

- Source: fast.ai Imagenette2-160 (derived from ImageNet), https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
- Synsets mapped to zero-based ImageNet class IDs: 0, 217, 482, 491, 497, 566, 569, 571, 574, 701
- Label mapping checked against torchvision MobileNet_V2_Weights.DEFAULT metadata
- Calibration images selected from source train; validation and final test are nonoverlapping slices of source val
- Seeded class-balanced selection, dataset archive SHA256 and all manifest SHA256 stored
- Images and models are not committed. Follow applicable ImageNet/Imagenette dataset licensing terms.

## Full matrix

| Dimension | Levels |
|---|---|
| Calibration samples | 32, 64, 128 |
| Calibrator | MinMax, Entropy, Percentile |
| Weight quantization | Per-tensor; per-channel |
| Fixed | Model hash, preprocessing, image splits, seed |
| Selection | Best validation Top-1; tie by smallest max absolute error, then model size |
| Final test | Opened only for selected variant (not used for tuning) |

Output per run: validation Top-1/Top-5, loss vs FP32 in percentage points, Top-1 agreement, maximum numerical output error, QDQ model size, local ORT CPU P50/P95 and quantization seconds.

## Run on Ubuntu

~~~bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,model,viz]"

npu-factory prepare-imagenette \
  --root data/imagenette --calibration-per-class 24 \
  --validation-per-class 10 --test-per-class 10

npu-factory ablate \
  --data-root data/imagenette \
  --calibration data/imagenette/calibration.jsonl \
  --validation data/imagenette/validation.jsonl \
  --test data/imagenette/test.jsonl \
  --out runs/exp01-imagenette \
  --sizes 32,64,128 \
  --methods minmax,entropy,percentile \
  --warmup 2 --repeats 20

npu-factory visualize \
  --summary runs/exp01-imagenette/summary.json \
  --out runs/exp01-imagenette/report
~~~

The generated report contains **five PNG charts, an HTML report and CSV table** plus per-variant JSON results. Failed configurations are marked failed, not silently omitted. A completed study must have 18/18 completed and a final holdout result.

## GitHub Actions evidence

The dedicated [full ablation workflow](../../.github/workflows/full-ablation.yml) installs CPU-only PyTorch, executes real-data evaluation, and uploads report artifacts. It does not upload the underlying dataset or model binaries. Use its recorded run status and artifacts rather than assuming configured experiments completed.

## Interpretation limits

Imagenette selects only ten of the 1000 ImageNet classes, so the resulting accuracy is on a restricted but real image distribution; it is not full ImageNet Top-1. These are x86 CPU timings excluding preprocessing/session creation, and are not NPU, power, thermals or camera-pipeline benchmarks. Hardware deployment gates remain open pending real SA8255P/SA8295P access and licensed QAIRT.
