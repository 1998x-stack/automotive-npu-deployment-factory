"""Model-level integration tests are opt-in: heavy dependencies not required in CI smoke."""
import pytest

from npu_factory.pipeline import benchmark


def test_benchmark_rejects_short_sampling():
    with pytest.raises(ValueError, match="repeats"):
        benchmark("unused.onnx", "unused.jsonl", ".", "unused.json", repeats=1)
