import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from metrics.registry import DETECTOR  # noqa: E402

detectors_package = types.ModuleType("detectors")
detectors_package.__path__ = [str(TRAINING_DIR / "detectors")]
detectors_package.DETECTOR = DETECTOR
sys.modules["detectors"] = detectors_package

from detectors.BiasArtifactConsistency import (  # noqa: E402
    BiasArtifactConsistencyDetector,
    CorrectnessAwareKLLoss,
)


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=4):
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.scale = nn.Parameter(torch.ones(feature_dim))

    def forward(self, image):
        return SimpleNamespace(
            pooler_output=self.norm(self.proj(image) * self.scale)
        )


def detector_config():
    return {
        "feature_dim": 4,
        "weight_real": 1.0,
        "weight_fake": 1.0,
        "label_smoothing": 0.05,
        "weak_ce_weight": 0.75,
        "strong_ce_weight": 0.25,
        "use_lora": False,
        "train_backbone_bias": True,
        "use_consistency": True,
        "use_consistency_views": True,
        "consistency_temperature": 1.0,
        "consistency_confidence_threshold": 0.0,
        "consistency_require_teacher_correct": True,
        "consistency_class_balanced": True,
        "alpha_consistency_max": 0.2,
        "alpha_consistency_start_epoch": 2,
        "alpha_consistency_warmup_epochs": 3,
        "normalize_eps": 1e-6,
        "strict_trainable_check": True,
        "strict_clip_architecture": False,
    }


def make_batch():
    return {
        "image": torch.tensor(
            [
                [1.0, 0.2, -0.1],
                [0.7, -0.3, 0.4],
                [-0.2, 0.8, 0.5],
                [0.1, 0.6, -0.7],
            ]
        ),
        "image_strong": torch.tensor(
            [
                [0.8, 0.3, -0.2],
                [0.6, -0.1, 0.5],
                [-0.3, 0.7, 0.4],
                [0.2, 0.4, -0.8],
            ]
        ),
        "label": torch.tensor([0, 0, 1, 1]),
    }


def test_detector_is_registered():
    assert (
        DETECTOR["bias_artifact_consistency"]
        is BiasArtifactConsistencyDetector
    )


def test_consistency_selects_only_confident_correct_teachers():
    loss_fn = CorrectnessAwareKLLoss(
        confidence_threshold=0.8,
        require_teacher_correct=True,
        class_balanced=True,
    )
    weak_logits = torch.tensor(
        [[5.0, 0.0], [0.0, 5.0], [0.0, 5.0], [0.1, 0.0]],
        requires_grad=True,
    )
    strong_logits = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 0, 0])

    loss, diagnostics = loss_fn(weak_logits, strong_logits, labels)
    loss.backward()

    assert diagnostics["consistency_confident_fraction"].item() == pytest.approx(0.75)
    assert diagnostics["consistency_selected_fraction"].item() == pytest.approx(0.5)
    assert diagnostics["teacher_accuracy"].item() == pytest.approx(0.75)
    assert diagnostics["consistency_selected_real_fraction"].item() == pytest.approx(1 / 3)
    assert diagnostics["consistency_selected_fake_fraction"].item() == pytest.approx(1.0)
    assert weak_logits.grad is None
    assert strong_logits.grad is not None


def test_detector_uses_artifact_preserving_loss_formula():
    torch.manual_seed(11)
    detector = BiasArtifactConsistencyDetector(
        detector_config(),
        backbone=TinyBackbone(feature_dim=4),
    )
    detector.train()
    detector.epoch = 5
    batch = make_batch()

    predictions = detector(batch)
    losses = detector.get_losses(batch, predictions)

    assert torch.allclose(
        predictions["cls"],
        0.75 * predictions["cls_weak"]
        + 0.25 * predictions["cls_strong"],
    )
    assert torch.allclose(
        losses["loss_ce"],
        0.75 * losses["loss_ce_weak"]
        + 0.25 * losses["loss_ce_strong"],
    )
    assert losses["alpha_consistency"].item() == pytest.approx(0.2)
    assert torch.allclose(
        losses["overall"],
        losses["loss_ce"]
        + losses["alpha_consistency"] * losses["loss_consistency"],
    )

    losses["overall"].backward()
    for name, parameter in detector.backbone.named_parameters():
        assert parameter.requires_grad == name.endswith(".bias")
    assert detector.backbone.proj.weight.grad is None
    assert detector.backbone.proj.bias.grad is not None
    assert detector.head.weight.grad is not None


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.0), (2, 0.0), (3, 0.2 / 3.0), (5, 0.2)],
)
def test_consistency_starts_late_and_warms_up(epoch, expected):
    detector = BiasArtifactConsistencyDetector(
        detector_config(),
        backbone=TinyBackbone(feature_dim=4),
    )
    detector.train()
    detector.epoch = epoch

    assert detector._current_alpha_consistency_value() == pytest.approx(expected)


def test_eval_uses_only_clean_weak_view():
    detector = BiasArtifactConsistencyDetector(
        detector_config(),
        backbone=TinyBackbone(feature_dim=4),
    )
    detector.eval()
    batch = make_batch()
    batch.pop("image_strong")

    with torch.no_grad():
        predictions = detector(batch, inference=True)
        losses = detector.get_losses(batch, predictions)

    assert predictions["cls_strong"] is None
    assert losses["loss_consistency"].item() == 0.0
    assert losses["alpha_consistency"].item() == 0.0

