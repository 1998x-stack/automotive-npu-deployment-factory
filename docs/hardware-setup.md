# Hardware and toolchain setup

## Development tiers

| Tier | Hardware | Can confirm | Cannot confirm |
|---|---|---|---|
| L0 | x86 Ubuntu + optional RTX 4060 | ONNX graph, CPU baseline, ORT PTQ, QNN plan | HTP speed/power |
| L1 | compatible Snapdragon Android device or QCS development board | selected QNN HTP execution, real device logs | automatic SA8255P/SA8295P portability |
| L2 | authorized SA8255P/SA8295P BSP + board | actual automotive target latency and integration | vehicle production qualification without OEM procedures |

**No purchase is required** to implement S00-S02.

## Toolchain prerequisites

- Python 3.10+ and recent Ubuntu x86 host for Python/ONNX.
- Licensed, compatible Qualcomm AI Runtime (QAIRT) SDK for native QNN steps.
- Device-specific HTP Host Stub / DSP Skeleton, firmware and permissions.
- Exact SoC/BSP/SDK/firmware compatibility matrix provided by OEM or vendor.

Never redistribute proprietary components inside the public repo.

## Host and device verification worksheet

```text
host_os:
python:
torch:
torchvision:
onnx:
onnxruntime:
qairt:
device_name:
device_is_proxy: unknown
soc_model:
htp_arch:
bsp_revision:
dsp_firmware:
model_sha256:
dataset_sha256:
runtime_backend:
raw_profile_log:
thermal_conditions:
```

## Hardware gating

1. Query the currently accessible AI Hub device list; the historic catalog does not guarantee device access.
2. Distinguish physical and proxy devices explicitly.
3. Compile only against a target supported by the exact QAIRT release.
4. Before NPU claim, retain unmodified output/log, device identity and output parity.
5. If hardware is absent, mark test **blocked** or **pending**, not passed.
