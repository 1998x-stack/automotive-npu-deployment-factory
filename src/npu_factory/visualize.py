"""Produce figures and a static report only from recorded measurement data."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from .utils import write_json


def generate_report(summary: dict, output_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summary.get("experiments", [])
    completed = [r for r in rows if r.get("status") == "completed"]
    if not completed:
        raise ValueError("No completed experiments: refusing empty charts")

    keys = [
        "id", "calibration_size", "method", "per_channel", "status",
        "validation_top1", "validation_top5", "top1_drop_pp",
        "agreement_top1", "max_absolute_error", "model_size_mb",
        "cpu_p50_ms", "cpu_p95_ms", "quantization_seconds",
    ]
    with (output_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    labels = [r["id"] for r in completed]
    xs = np.arange(len(completed))
    width = max(10, len(completed) * 0.6)

    def plot_bar(metric: str, title: str, filename: str, ylabel: str) -> None:
        vals = [r.get(metric) for r in completed]
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not valid:
            return
        fig, ax = plt.subplots(figsize=(width, 5))
        ax.bar([i for i, _ in valid], [v for _, v in valid])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=145)
        plt.close(fig)

    plot_bar("validation_top1", "Imagenette validation: Top-1 (1000-way)", "top1.png", "accuracy")
    plot_bar("top1_drop_pp", "PTQ vs FP32: Top-1 change", "accuracy_drop.png", "percentage points")
    plot_bar("model_size_mb", "QDQ ONNX model size", "model_size.png", "MB")
    plot_bar("cpu_p50_ms", "ONNX Runtime CPU: P50 model-only latency", "cpu_latency.png", "ms")
    plot_bar("quantization_seconds", "Calibration and quantization elapsed", "quant_time.png", "seconds")

    baseline = summary.get("baseline", {})
    def fmt(x, digits=4):
        return "N/A" if x is None else f"{x:.{digits}f}"

    table_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(k, 'N/A')))}</td>" for k in keys
        )
        table_rows.append(f"<tr>{cells}</tr>")
    columns = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    figures = []
    for file in ("top1.png", "accuracy_drop.png", "model_size.png",
                 "cpu_latency.png", "quant_time.png"):
        if (output_dir / file).exists():
            figures.append(f'<figure><img src="{file}" alt="{file}"/></figure>')

    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Experiment 01 - Ablations</title>
<style>body{{font:15px system-ui;max-width:1250px;margin:35px auto;padding:0 20px}}
figure img{{max-width:100%}} table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #bbb;padding:7px}}th{{background:#eee}}
aside{{border-left:3px solid #777;padding:10px;background:#f5f5f5}}</style>
<h1>Experiment 01: reproducible INT8 PTQ ablations</h1>
<aside>Data: {html.escape(str(summary.get("dataset_scope","unknown")))}.
Host CPU measurements are <strong>not</strong> Qualcomm HTP measurements.
No SA8255P/SA8295P benchmark is claimed.</aside>
<p>Runs: {len(completed)} completed / {len(rows)} total;
FP32 validation Top-1: {fmt(baseline.get("validation_top1"))};
chosen method: {html.escape(str(summary.get("selected_id","none")))}</p>
{''.join(figures)}
<h2>Complete measurements</h2><table><thead><tr>{columns}</tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p>Source: <code>summary.json</code>; report contains no simulated NPU values.</p></html>"""
    (output_dir / "report.html").write_text(content, encoding="utf-8")
    manifest = {
        "report": str(output_dir / "report.html"),
        "csv": str(output_dir / "ablation_results.csv"),
        "figures": [
            x for x in ("top1.png", "accuracy_drop.png", "model_size.png",
                        "cpu_latency.png", "quant_time.png") if (output_dir / x).exists()
        ],
        "completed": len(completed),
        "attempted": len(rows),
    }
    write_json(output_dir / "report_manifest.json", manifest)
    return manifest


def from_summary_file(path: Path, output_dir: Path) -> dict:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    return generate_report(summary, output_dir)
