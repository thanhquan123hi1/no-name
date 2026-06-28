import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Unit-test the new module in isolation so importing the existing detector package
# does not import CLIP. The real package registry is still used for registration.
from metrics.registry import DETECTOR  # noqa: E402

detectors_package = types.ModuleType("detectors")
detectors_package.__path__ = [str(TRAINING_DIR / "detectors")]
detectors_package.DETECTOR = DETECTOR
sys.modules["detectors"] = detectors_package

from detectors.BiasRealAnchor import (  # noqa: E402
    BiasRealAnchorDetector,
    RealAnchorSupConLoss,
)


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=4):
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)
        self.scale = nn.Parameter(torch.ones(feature_dim))

    def forward(self, image):
        return SimpleNamespace(pooler_output=self.proj(image) * self.scale)


def test_detector_is_registered_with_expected_name():
    assert DETECTOR["bias_real_anchor"] is BiasRealAnchorDetector


@pytest.fixture
def detector():
    torch.manual_seed(7)
    config = {
        "feature_dim": 4,
        "weight_real": 1.0,
        "weight_fake": 1.3,
        "label_smoothing": 0.1,
        "alpha_real_max": 0.05,
        "alpha_real_start_epoch": 0,
        "alpha_real_warmup_epochs": 5,
        "temperature": 0.2,
        "base_temperature": 0.2,
        "normalize_eps": 1e-6,
        "use_lora": False,
        "strict_trainable_check": True,
    }
    return BiasRealAnchorDetector(config, backbone=TinyBackbone(feature_dim=4))


def test_only_two_real_samples_are_anchors():
    temperature = 0.5
    loss_fn = RealAnchorSupConLoss(
        temperature=temperature,
        base_temperature=temperature,
    )
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    actual = loss_fn(features, labels)

    normalized = F.normalize(features, dim=1)
    logits = normalized @ normalized.T / temperature
    manual_anchor_losses = []
    for anchor, positive in ((0, 1), (1, 0)):
        denominator_indices = [index for index in range(4) if index != anchor]
        log_denominator = torch.logsumexp(logits[anchor, denominator_indices], dim=0)
        manual_anchor_losses.append(-(logits[anchor, positive] - log_denominator))
    expected = torch.stack(manual_anchor_losses).mean()

    assert torch.allclose(actual, expected, atol=1e-6)


def test_fake_samples_are_neither_anchors_nor_fake_fake_positives():
    loss_fn = RealAnchorSupConLoss(temperature=0.5, base_temperature=0.5)
    labels = torch.tensor([0, 0, 1, 1])
    shared_real = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    aligned_fakes = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    opposed_fakes = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

    aligned_loss = loss_fn(torch.cat([shared_real, aligned_fakes]), labels)
    opposed_loss = loss_fn(torch.cat([shared_real, opposed_fakes]), labels)

    assert F.cosine_similarity(aligned_fakes[0:1], aligned_fakes[1:2]).item() == 1.0
    assert F.cosine_similarity(opposed_fakes[0:1], opposed_fakes[1:2]).item() == -1.0
    assert torch.allclose(aligned_loss, opposed_loss, atol=1e-6)


def test_fake_samples_receive_gradient_as_real_anchor_negatives():
    loss_fn = RealAnchorSupConLoss(temperature=0.3, base_temperature=0.3)
    features = torch.tensor(
        [
            [1.0, 0.2, 0.1],
            [0.8, 0.3, -0.2],
            [0.2, 1.0, 0.4],
            [-0.3, 0.7, 1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])

    loss_fn(features, labels).backward()

    assert features.grad is not None
    assert torch.all(features.grad[2:].norm(dim=1) > 0)


def test_single_real_sample_returns_finite_differentiable_zero():
    loss_fn = RealAnchorSupConLoss()
    features = torch.randn(3, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 1])

    loss = loss_fn(features, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert features.grad is not None
    assert torch.count_nonzero(features.grad).item() == 0


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.0), (1, 0.01), (5, 0.05), (6, 0.05)],
)
def test_progressive_alpha(detector, epoch, expected):
    detector.epoch = epoch
    assert detector._current_alpha_real_value() == pytest.approx(expected)


def test_only_backbone_bias_and_classifier_are_trainable(detector):
    for name, parameter in detector.backbone.named_parameters():
        assert parameter.requires_grad == name.endswith(".bias")

    assert all(parameter.requires_grad for parameter in detector.head.parameters())
    assert not any(
        "lora_A" in name or "lora_B" in name
        for name, _ in detector.named_parameters()
    )
    assert not hasattr(detector, "projection_head")


def test_forward_and_losses_have_required_keys_and_shapes(detector):
    data = {
        "image": torch.randn(4, 3),
        "label": torch.tensor([0, 0, 1, 1]),
    }
    detector.epoch = 1

    predictions = detector(data)
    losses = detector.get_losses(data, predictions)

    assert set(predictions) == {"cls", "prob", "feat", "feat_norm"}
    assert predictions["cls"].shape == (4, 2)
    assert predictions["prob"].shape == (4,)
    assert predictions["feat"].shape == (4, 4)
    assert predictions["feat_norm"].shape == (4, 4)
    assert torch.allclose(
        predictions["feat_norm"].norm(dim=1),
        torch.ones(4),
        atol=1e-6,
    )
    assert "feat_proj" not in predictions

    required_loss_keys = {
        "overall",
        "loss_ce",
        "loss_real_anchor_supcon",
        "alpha_real",
        "weighted_real_anchor_supcon",
        "real_loss",
        "fake_loss",
        "mean_real_real_cosine",
        "mean_real_fake_cosine",
        "mean_fake_fake_cosine",
        "num_real",
        "num_fake",
    }
    assert required_loss_keys.issubset(losses)
    assert all(losses[key].ndim == 0 for key in required_loss_keys)
    assert losses["num_real"].item() == 2
    assert losses["num_fake"].item() == 2
    assert not losses["mean_real_real_cosine"].requires_grad
    assert not losses["mean_real_fake_cosine"].requires_grad
    assert not losses["mean_fake_fake_cosine"].requires_grad


def test_overall_loss_matches_declared_formula(detector):
    data = {
        "image": torch.randn(4, 3),
        "label": torch.tensor([0, 0, 1, 1]),
    }
    detector.epoch = 3
    predictions = detector(data)
    losses = detector.get_losses(data, predictions)

    expected = losses["loss_ce"] + (
        losses["alpha_real"] * losses["loss_real_anchor_supcon"]
    )
    assert torch.allclose(losses["overall"], expected)
    assert torch.allclose(
        losses["weighted_real_anchor_supcon"],
        losses["alpha_real"] * losses["loss_real_anchor_supcon"],
    )
