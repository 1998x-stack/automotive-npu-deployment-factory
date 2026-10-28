import pytest

from npu_factory.statistics import paired_accuracy_evidence


def test_identical_predictions_zero_delta():
    stats = paired_accuracy_evidence(
        [True, False, True, False, True],
        [True, False, True, False, True],
        bootstrap_replicates=100,
    )
    assert stats["candidate_minus_reference_pp"] == 0
    assert stats["paired_bootstrap_low_pp"] == 0
    assert stats["paired_bootstrap_high_pp"] == 0
    assert stats["mcnemar_exact_two_sided_p"] == 1.0


def test_all_candidate_wins_exact_mcnemar():
    stats = paired_accuracy_evidence(
        [False] * 8, [True] * 8, bootstrap_replicates=200,
    )
    assert stats["candidate_minus_reference_pp"] == 100
    assert stats["paired_bootstrap_low_pp"] == 100
    assert stats["mcnemar_exact_two_sided_p"] == pytest.approx(2 / 256)


def test_bootstrap_determinism():
    a = [True, True, False, True, False] * 12
    b = [False, True, True, True, True] * 12
    stats1 = paired_accuracy_evidence(a, b, seed=23, bootstrap_replicates=500)
    stats2 = paired_accuracy_evidence(a, b, seed=23, bootstrap_replicates=500)
    assert stats1 == stats2


def test_reject_misalignment():
    with pytest.raises(ValueError, match="paired"):
        paired_accuracy_evidence([True], [False, True])
