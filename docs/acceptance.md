# Experiment 01 — objective verification and report schema

## Evidence tiers

| Level | Claim allowed | Evidence required |
|---|---|---|
| DESIGN | API, architecture and acceptance plans written | versioned docs |
| SMOKE | CLI and data plumbing works | reproducible local logs |
| NUMERICAL | Torch/ONNX outputs agree | output tensors, tolerance and model hash |
| QUANTIZED | PTQ model runs and accuracy measured | independent real labels + QDQ model hash |
| COMPILED_QNN | QNN conversion/compilation succeeds | licensed SDK logs and version |
| MEASURED_HTP | target HTP execution and timing | device/BSP/DSP/runtime identity, raw logs |
| INTEGRATED | Android/Linux service runs under target workload | system test reports |
| QUALIFIED | OEM acceptance completed | OEM release/verification process |

Never transform a lower evidence tier into a higher tier by narrative.

## Holdout policy

- Disjoint manifests validated by normalized paths and file contents.
- Model calibration must use calibration split only.
- Test labels are authentic ImageNet indices and independent of quantization choices.
- Ablation decisions selected using validation data; keep a final untouched test set if tuning parameters.
- Synthetic smoke labels may not be used to report Accuracy, mAP or WER.

## Benchmark report extension for real devices

```json
{
  "experiment_id": "exp01-demo",
  "evidence_tier": "MEASURED_HTP",
  "device": {"soc": "SA8295P", "physical_or_proxy": "physical", "bsp": "REQUIRED"},
  "runtime": {"provider": "QNN", "sdk": "REQUIRED", "dsp_firmware": "REQUIRED"},
  "artifacts": {"onnx_sha256": "REQUIRED", "input_sha256": "REQUIRED"},
  "sample_count": 100,
  "latency_ms": {"p50": null, "p95": null, "p99": null},
  "raw_log_uri": "REQUIRED",
  "verification": "PENDING"
}
```

The example has no measured values: **do not publish it as a real benchmark**.

## Required error cases

- invalid or traversal image path is rejected
- duplicated calibration/test input is rejected by bytes
- missing model input / no samples fails
- numerical parity gate fails on wrong model output
- QNN unsupported provider must fail without CPU fallback
- QNN plan must not execute anything by default
- service restart or device fault must not silently mark success
- missing SDK, target or firmware identity means NPU evidence is inadmissible
