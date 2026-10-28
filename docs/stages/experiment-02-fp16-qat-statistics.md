# Experiment 02: FP16 / QAT / statistical robustness

This follow-up branch builds on Experiment 01. It does not touch main and does not claim NPU performance.

## Design

- FP16: convert pretrained MobileNetV2 ONNX internals to FP16 (FP32 inputs and outputs), run on CPU and compare image-level correctness.
- PTQ: repeat the selected 64-image MinMax/per-channel INT8 QDQ recipe without tuning on final test.
- QAT: fine-tune **only the last 1000-class linear head**, keeping the pretrained FP32 backbone frozen. Compare against a matched head-only FP32 fine-tune with identical training split, epochs and optimizer.
- Multi-seed: seeds 7, 19, 31; choose QAT seed on validation, evaluate the chosen seed on the sealed final test set.
- Statistical evaluation: 350 independent Imagenette test images, paired bootstrap 95% CIs and exact McNemar p-values.
- QNN/HTP: use the preflight command to disclose missing SDK and physical target requirements. No hardware claims.

## Reproduce

~~~bash
python -m pip install -e ".[dev,model,viz,fp16]"
ruff check .
pytest -q
npu-factory prepare-imagenette --root data/imagenette --calibration-per-class 24 --validation-per-class 10 --test-per-class 35
npu-factory stage-02 --data-root data/imagenette --calibration data/imagenette/calibration.jsonl --validation data/imagenette/validation.jsonl --test data/imagenette/test.jsonl --out runs/experiment-02 --seeds 7,19,31 --qat-images 80 --ptq-images 64 --epochs 3
npu-factory htp-preflight --soc sa8295p --out runs/htp-preflight.json
~~~

## Evidence gating

- Require valid real labels, no overlapping images across calibration, validation and final test.
- Require actual FP16 execution, not just a converted model file.
- Only validation determines chosen QAT training seed; holdout opened once for selected comparison.
- The FP32 and QAT heads use matched training, while backbone remains completely unquantized.
- The 350-image test set estimates uncertain accuracy differences; intervals are conditional on this dataset.
- Report test accuracy, effect-size in percentage points, paired 95% bootstrap interval, exact McNemar and images/seed.
- No inference-speed claims: CPU FP16 execution and QAT head simulation do not establish HTP support.

## Artifacts

Experiment output contains summary.json, report.html, figures/qat_seed_validation.png and figures/holdout_paired_confidence.png, in addition to source/archive SHA256 and Python environment freeze. CI retains reports for 30 days.