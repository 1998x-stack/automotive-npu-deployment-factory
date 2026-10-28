# Automotive NPU Deployment Factory

**Experiment 01: CNN Model Export → INT8 PTQ → QNN/HTP Porting**

An engineering-focused, reproducible learning laboratory for the Qualcomm SA8255P/SA8295P deployment toolchain. Hardware is optional for the first stages. All outputs distinguish **local measurements**, **plans**, and **hardware-unverified work**.

> This repository is not a production automotive software component and does not include proprietary Qualcomm SDKs, OEM BSP images or automotive firmware.

## Start here

- [Full stage plan](docs/stages/exp01-roadmap.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Hardware and QAIRT preparation](docs/hardware-setup.md)
- [Experiment evidence and acceptance template](docs/acceptance.md)
- [Engineering agent rules](AGENTS.md)

## Stage implementation overview

| Stage | Objective | Current implementation |
|---|---|---|
| S00 | environment, manifests, dataset isolation | Source + unit tests |
| S01 | TorchVision MobileNetV2 → ONNX, numerical parity | Executable CLI; needs optional ML packages |
| S02 | PTQ calibration, holdout evaluation | Executable CLI; genuine accuracy needs real labeled images |
| S03 | QNN conversion preparation | Raw tensor writer + safe command plan; licensed QAIRT required to compile |
| S04 | SA8255P/SA8295P profiling | Acceptance protocol; real target required |
| S05 | latency, memory, thermal optimization | ORT session.run latency implemented; actual hardware requires measurement |
| S06 | Android/Linux native inference service | Design contract; not claimed implemented |
| S07 | ASR / VLM / LLM extension | Research boundary; planned outside this CNN experiment |
| S08 | CI, provenance and reviews | Python unit tests, GitHub Actions, JSON SHA256 evidence |

## Install (Ubuntu 22.04 / 24.04 recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,model]"
ruff check .
pytest -q
```

No QAIRT, network access or high-end GPU required for the basic tests.

## A. Offline smoke run (does NOT establish model accuracy)

```bash
npu-factory smoke-data --root data/generated --count 6
npu-factory check-splits \
  --calibration data/generated/calibration.jsonl \
  --test data/generated/test.jsonl --data-root data/generated

# By default weights are random and no model weights are downloaded.
npu-factory export --output artifacts/mobilenet_v2.onnx

npu-factory quantize --model artifacts/mobilenet_v2.onnx \
  --calibration data/generated/calibration.jsonl \
  --test data/generated/test.jsonl \
  --data-root data/generated \
  --output artifacts/mobilenet_v2.int8.qdq.onnx

npu-factory evaluate --float-model artifacts/mobilenet_v2.onnx \
  --quant-model artifacts/mobilenet_v2.int8.qdq.onnx \
  --test data/generated/test.jsonl --data-root data/generated \
  --report runs/exp01-smoke-evaluation.json

npu-factory benchmark --model artifacts/mobilenet_v2.onnx \
  --test data/generated/test.jsonl --data-root data/generated \
  --report runs/exp01-cpu-fp32.json
```

The smoke evaluation reports numerical agreement but intentionally excludes meaningful model accuracy; synthetic labels are not ground truth.

## B. Scientific evaluation with real images

1. Acquire a properly licensed ImageNet-compatible holdout dataset and calibration dataset.
2. Prepare JSONL manifests whose paths are relative to a local dataset root.
3. Set each record's label to the actual **zero-based ImageNet class index (0..999)**. All labels must use the same class vocabulary as the pretrained model.
4. Make calibration and test images disjoint (the CLI checks paths and file contents); if training or tuning new models, keep that split separate too.
5. Export using `npu-factory export --pretrained` (explicit pretrained weight retrieval).
6. Run PTQ, paired evaluation and benchmark on unchanged images/preprocessing.
7. Vary calibration size, method and per-channel/per-tensor precision under the Experiment-01 matrix.

Example manifest line:

```json
{"path": "images/val_00001.jpg", "label": 237, "source": "real"}
```

A fabricated label or model trained with random weights is not scientific evidence.

## C. QNN hand-off

The QNN converter consumes FP32 ONNX and real calibration inputs; the ORT QDQ file is a **separate** experiment route, not automatically the correct native QNN input.

```bash
npu-factory qnn-input-list \
  --calibration /DATA/calibration.jsonl \
  --test /DATA/test.jsonl \
  --data-root /DATA \
  --out artifacts/qnn_calibration

npu-factory qnn-plan \
  --model artifacts/mobilenet_v2.onnx \
  --input-list artifacts/qnn_calibration/input_list.txt \
  --sdk-root /opt/qairt \
  --out artifacts/qnn
```

The output is a human-reviewable command plan. It does not run QNN, claim hardware support, or compile a context binary. Follow target-specific SDK documents and record every converter and HTP log.

## D. Model-only QNN EP benchmark (only on a capable target)

```bash
npu-factory benchmark --model artifacts/mobilenet_v2.int8.qdq.onnx \
  --test /DATA/test.jsonl --data-root /DATA \
  --provider QNNExecutionProvider --backend htp \
  --report runs/htp-target.json
```

It fails if the QNN provider is unavailable; CPU fallback is explicitly disabled. **Do not interpret this command as an NPU measurement unless it actually ran on the target with verifiable logs.**

## Upstream reference material

- [ONNX Runtime static quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models)
- [ExecuTorch Qualcomm backend](https://github.com/pytorch/executorch)

## Provenance policy

Record input model SHA256, quantized model SHA256, calibration and test manifest hashes, Python and tool versions, backend, target device, BSP and QAIRT releases, warmup, sample count, output report and raw logs. Keep images, model blobs, native SDK libraries, OEM configuration and access tokens outside Git.
