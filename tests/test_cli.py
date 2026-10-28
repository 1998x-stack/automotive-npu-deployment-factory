import json
from pathlib import Path

from npu_factory.cli import main


def test_smoke_cli_end_to_end(tmp_path: Path, capsys):
    root = tmp_path / "fixtures"
    assert main(["smoke-data", "--root", str(root), "--count", "6"]) == 0
    response = json.loads(capsys.readouterr().out)
    assert set(response) == {"calibration", "test"}
    assert main([
        "check-splits", "--calibration", str(root / "calibration.jsonl"),
        "--test", str(root / "test.jsonl"), "--data-root", str(root)
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "disjoint"


def test_qnn_plan_cli(tmp_path: Path, capsys):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"placeholder")
    inputs = tmp_path / "inputs.txt"
    inputs.write_text("image:=/tmp/a.raw\n")
    assert main([
        "qnn-plan", "--model", str(model), "--input-list", str(inputs),
        "--sdk-root", str(tmp_path / "sdk"), "--out", str(tmp_path / "qnn")
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "planned_not_executed"
    assert (tmp_path / "qnn" / "plan.json").exists()
