from pathlib import Path

from npu_factory.qnn import create_qnn_plan


def test_qnn_plan_does_not_execute_or_claim_hardware(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx-placeholder")
    inputs = tmp_path / "inputs.txt"
    inputs.write_text("image:=/path/to/example.raw\n")
    plan = create_qnn_plan(model, inputs, tmp_path / "qairt", tmp_path / "output")
    assert plan["status"] == "planned_not_executed"
    assert len(plan["commands"]) == 2
    assert plan["commands"][0][0].endswith("qnn-onnx-converter")
    assert not (tmp_path / "output").exists()
