import numpy as np
import pytest

from npu_factory.metrics import classification, latency_stats, numerical_difference


def test_numerical_equal():
    value = np.array([[1.0, 3.0, 2.0]])
    result = numerical_difference(value, value)
    assert result["max_absolute_error"] == 0
    assert result["top1_equal"]


def test_numerical_mismatch_shape():
    with pytest.raises(ValueError, match="Shape mismatch"):
        numerical_difference(np.zeros((2,)), np.zeros((3,)))


def test_classification_top1_top5():
    values = [np.array([0.2, 0.9, 0.3, 0.0, 0.1, 0.4])]
    result = classification(values, [1])
    assert result == {"count": 1, "top1": 1.0, "top5": 1.0}


def test_latency_distribution_and_reject_nans():
    result = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert result["p50_ms"] == 2.5
    assert result["samples"] == 4
    with pytest.raises(ValueError, match="finite"):
        latency_stats([1.0, float("nan")])
