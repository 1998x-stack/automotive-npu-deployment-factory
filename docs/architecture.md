# Architecture — Experiment 01

## Data and model flow

```text
versioned calibration manifest ---+
                                  v
torchvision MobileNetV2 -> ONNX FP32 -> ORT QDQ PTQ -> ONNX INT8
                           |                        |
                           +-- parity report         +-- paired holdout accuracy
                           |                        +-- ORT CPU benchmark
                           +-- QNN converter plan -> licensed QAIRT toolchain
                                                   -> HTP target binary
                                                   -> SA8255P / SA8295P evidence
```

## Trust boundaries

1. PyTorch weights are explicit: randomly initialized for pipeline smoke, optionally ImageNet pretrained for genuine evaluation.
2. Calibration/test manifests are separate; path and byte-level overlap are rejected.
3. ImageNet labels must use zero-based integer indices in the same vocabulary/order as model logits.
4. Synthetic/unknown labels never yield an accuracy claim.
5. Graph conversion and quantization artifacts are not proofs of HTP execution.
6. The external QAIRT SDK, device credentials, DSP libraries and OEM BSP are **not** committed.
7. QNN plan is a list of commands, not a shell interpreter; no hidden execution.
8. Only a device-measured result with exact hardware/firmware/model provenance may be called an NPU benchmark.

## Logical packages

- **data.py:** dataset contract, split integrity, preprocess, deterministic synthetic fixtures.
- **metrics.py:** numerical difference, classification, model-only latency.
- **pipeline.py:** lazy-import optional ML components, export, PTQ, paired accuracy, CPU/QNN EP benchmark.
- **qnn.py:** safe plan generation; command execution requires an external authorized environment.
- **cli.py:** repeatable command interface and auditable JSON output.
- **tests/:** no network, QNN or GPU requirements for smoke contract tests.

## Key missing hardware components

The experiments do not implement Android service, AIDL, DSP FastRPC, HTP Op Package or OEM BSP. Those require target access, review and device-specific support. S06 design requires model registry, narrow inference RPC, ACL, memory ownership, deadlines, restart policy, and integration tests under combined IVI load.
