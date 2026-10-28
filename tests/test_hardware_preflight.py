import pytest

from npu_factory.hardware import hardware_preflight


def test_unverified_soC_is_not_reported_as_hardware_pass(tmp_path):
    path = tmp_path / "preflight.json"
    record = hardware_preflight(path, soc="sa8295p")
    assert record["status"] == "blocked_unverified"
    assert not record["preconditions"]["physical_target_confirmed_by_operator"]
    assert path.exists()


def test_preflight_rejects_unrequested_soc(tmp_path):
    with pytest.raises(ValueError):
        hardware_preflight(tmp_path / "bad.json", soc="sm8550")
