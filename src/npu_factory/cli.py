"""CLI for Experiment 01; intentionally avoids implicit downloads and device writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_manifest, make_smoke_dataset, preprocess, validate_disjoint
from .qnn import create_qnn_plan
from .utils import sha256, write_json


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-factory", description="Experiment 01: reproducible CNN/QNN deployment"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke-data", help="Generate synthetic pipeline fixtures (NO accuracy)")
    smoke.add_argument("--root", type=_path, default=Path("data/generated"))
    smoke.add_argument("--count", type=int, default=6)

    check = sub.add_parser("check-splits", help="Validate disjoint calibration/test inputs")
    for cmd in (check,):
        cmd.add_argument("--calibration", type=_path, required=True)
        cmd.add_argument("--test", type=_path, required=True)
        cmd.add_argument("--data-root", type=_path, required=True)

    export = sub.add_parser("export", help="TorchVision MobileNetV2 -> ONNX + numerical parity")
    export.add_argument("--output", type=_path, default=Path("artifacts/mobilenet_v2.onnx"))
    export.add_argument("--pretrained", action="store_true", help="Explicitly fetch ImageNet weights")
    export.add_argument("--seed", type=int, default=2026)

    quant = sub.add_parser("quantize", help="Static QDQ PTQ, calibration/test isolation required")
    quant.add_argument("--model", type=_path, required=True)
    quant.add_argument("--calibration", type=_path, required=True)
    quant.add_argument("--test", type=_path, required=True)
    quant.add_argument("--data-root", type=_path, required=True)
    quant.add_argument("--output", type=_path, required=True)
    quant.add_argument("--method", choices=["minmax", "entropy", "percentile"],
                       default="minmax")
    quant.add_argument("--per-tensor", action="store_true")

    evaluate = sub.add_parser("evaluate", help="FP32 vs QDQ paired evaluation")
    evaluate.add_argument("--float-model", type=_path, required=True)
    evaluate.add_argument("--quant-model", type=_path, required=True)
    evaluate.add_argument("--test", type=_path, required=True)
    evaluate.add_argument("--data-root", type=_path, required=True)
    evaluate.add_argument("--report", type=_path, default=Path("runs/eval.json"))

    bench = sub.add_parser("benchmark", help="Model-only P50/P95/P99 inference timing")
    bench.add_argument("--model", type=_path, required=True)
    bench.add_argument("--test", type=_path, required=True)
    bench.add_argument("--data-root", type=_path, required=True)
    bench.add_argument("--report", type=_path, default=Path("runs/benchmark.json"))
    bench.add_argument("--provider", choices=["CPUExecutionProvider", "QNNExecutionProvider"],
                       default="CPUExecutionProvider")
    bench.add_argument("--backend", choices=["", "htp", "cpu", "gpu"], default="")
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--repeats", type=int, default=100)

    prepare = sub.add_parser("qnn-input-list", help="Convert calibration JPEGs to NCHW float32 raw")
    prepare.add_argument("--calibration", type=_path, required=True)
    prepare.add_argument("--test", type=_path, required=True)
    prepare.add_argument("--data-root", type=_path, required=True)
    prepare.add_argument("--out", type=_path, default=Path("artifacts/qnn_calibration"))

    plan = sub.add_parser("qnn-plan", help="Generate auditable, NOT EXECUTED, QAIRT commands")
    plan.add_argument("--model", type=_path, required=True)
    plan.add_argument("--input-list", type=_path, required=True)
    plan.add_argument("--sdk-root", type=_path, required=True)
    plan.add_argument("--out", type=_path, default=Path("artifacts/qnn"))
    dataset = sub.add_parser(
        "prepare-imagenette",
        help="Opt-in download of real 10-class Imagenette holdout dataset",
    )
    dataset.add_argument("--root", type=_path, default=Path("data/imagenette"))
    dataset.add_argument("--calibration-per-class", type=int, default=24)
    dataset.add_argument("--validation-per-class", type=int, default=10)
    dataset.add_argument("--test-per-class", type=int, default=10)
    dataset.add_argument("--seed", type=int, default=2026)

    ablate = sub.add_parser(
        "ablate", help="Run complete PTQ matrix; select by validation, test once",
    )
    ablate.add_argument("--data-root", type=_path, required=True)
    ablate.add_argument("--calibration", type=_path, required=True)
    ablate.add_argument("--validation", type=_path, required=True)
    ablate.add_argument("--test", type=_path, required=True)
    ablate.add_argument("--out", type=_path, default=Path("runs/exp01"))
    ablate.add_argument("--sizes", default="32,64,128")
    ablate.add_argument("--methods", default="minmax,entropy,percentile")
    ablate.add_argument("--seed", type=int, default=2026)
    ablate.add_argument("--warmup", type=int, default=2)
    ablate.add_argument("--repeats", type=int, default=20)
    ablate.add_argument("--fail-fast", action="store_true")

    visualize = sub.add_parser("visualize", help="Recreate plots from real JSON evidence")
    visualize.add_argument("--summary", type=_path, required=True)
    visualize.add_argument("--out", type=_path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke-data":
        result = make_smoke_dataset(args.root, args.count)
    elif args.command == "check-splits":
        calibration = load_manifest(args.calibration, args.data_root)
        test = load_manifest(args.test, args.data_root)
        validate_disjoint(calibration, test)
        result = {"status": "disjoint", "calibration": len(calibration), "test": len(test)}
    elif args.command == "prepare-imagenette":
        from .imagenette import prepare_imagenette
        result = prepare_imagenette(
            args.root, calibration_per_class=args.calibration_per_class,
            validation_per_class=args.validation_per_class,
            test_per_class=args.test_per_class, seed=args.seed,
        )
    elif args.command == "ablate":
        from .ablations import run_ablation_matrix
        result = run_ablation_matrix(
            args.data_root, args.calibration, args.validation, args.test, args.out,
            calibration_sizes=tuple(int(x) for x in args.sizes.split(",")),
            methods=tuple(x.strip() for x in args.methods.split(",")),
            seed=args.seed, warmup=args.warmup, repeats=args.repeats,
            continue_on_error=not args.fail_fast,
        )
    elif args.command == "visualize":
        from .visualize import from_summary_file
        result = from_summary_file(args.summary, args.out)
    elif args.command == "export":
        from .pipeline import export_mobilenet
        result = export_mobilenet(args.output, pretrained=args.pretrained, seed=args.seed)
    elif args.command == "quantize":
        from .pipeline import quantize_onnx
        result = quantize_onnx(
            args.model, args.calibration, args.test, args.data_root, args.output,
            method=args.method, per_channel=not args.per_tensor
        )
    elif args.command == "evaluate":
        from .pipeline import evaluate_pair
        result = evaluate_pair(
            args.float_model, args.quant_model, args.test, args.data_root, args.report
        )
    elif args.command == "benchmark":
        from .pipeline import benchmark
        result = benchmark(
            args.model, args.test, args.data_root, args.report,
            provider=args.provider, backend=args.backend,
            warmup=args.warmup, repeats=args.repeats
        )
    elif args.command == "qnn-input-list":
        calibration = load_manifest(args.calibration, args.data_root)
        test = load_manifest(args.test, args.data_root)
        validate_disjoint(calibration, test)
        args.out.mkdir(parents=True, exist_ok=True)
        lines = []
        for index, item in enumerate(calibration):
            path = (args.out / f"calibration_{index:05d}.raw").resolve()
            preprocess(item.path).astype("<f4").tofile(path)
            lines.append(f"image:={path}")
        manifest = args.out / "input_list.txt"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = {
            "status": "prepared_not_compiled", "layout": "NCHW", "dtype": "float32",
            "input_name": "image", "calibration_count": len(lines),
            "input_list": str(manifest.resolve()),
            "input_list_sha256": sha256(manifest),
            "caveat": "Confirm QNN converter input tensor layout before deploying"
        }
        write_json(args.out / "preparation.json", result)
    elif args.command == "qnn-plan":
        result = create_qnn_plan(args.model, args.input_list, args.sdk_root, args.out)
        write_json(args.out / "plan.json", result)
    else:
        raise AssertionError("unhandled command")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
