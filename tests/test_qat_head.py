import pytest

torch = pytest.importorskip("torch")
from npu_factory.qat import QuantAwareHead, predict_head, train_head


def test_qat_head_trains_with_fake_quant_and_updates_weights():
    torch.manual_seed(7)
    original = torch.nn.Linear(8, 11)
    x = torch.randn(32, 8)
    labels = torch.randint(0, 11, (32,))
    qat, losses = train_head(
        original, x, labels, quant_aware=True, seed=5, epochs=2, batch_size=8,
    )
    assert len(losses) == 2
    assert losses[0] > 0
    assert not torch.equal(qat.weight.detach(), original.weight.detach())
    assert len(predict_head(qat, x)) == 32
    assert qat.quant_aware


def test_fp32_matched_head_training():
    original = torch.nn.Linear(8, 11)
    x = torch.randn(12, 8)
    labels = torch.randint(0, 11, (12,))
    trained, _ = train_head(
        original, x, labels, quant_aware=False, seed=5, epochs=1,
    )
    assert not trained.quant_aware


def test_head_qat_has_only_expected_quantized_component():
    original = torch.nn.Linear(8, 11)
    qat = QuantAwareHead(original, quant_aware=True)
    assert hasattr(qat, "input_fakequant")
    assert hasattr(qat, "weight_fakequant")
