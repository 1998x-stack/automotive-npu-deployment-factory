import pytest

from npu_factory.visualize import generate_report


def test_generate_report_refuses_unmeasured_claims(tmp_path):
    with pytest.raises(ValueError, match="No completed"):
        generate_report({"experiments": [{"id": "x", "status": "failed"}]}, tmp_path)


def test_visualize_verified_rows(tmp_path):
    pytest.importorskip("matplotlib")
    result = generate_report(
        {
            "dataset_scope": "unit-test-only synthetic metadata",
            "baseline": {"validation_top1": 0.5},
            "selected_id": "x",
            "experiments": [
                {
                    "id": "x", "status": "completed", "validation_top1": 0.4,
                    "top1_drop_pp": 10, "model_size_mb": 5.0,
                    "cpu_p50_ms": 12.0, "quantization_seconds": 3.0
                }
            ],
        },
        tmp_path,
    )
    assert result["completed"] == 1
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "ablation_results.csv").exists()
    assert (tmp_path / "top1.png").exists()
