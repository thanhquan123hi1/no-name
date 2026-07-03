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

from detectors.BiasArtifactConsistency import CorrectnessAwareKLLoss  # noqa: E402
from detectors.BiasEMATeacherConsistency import (  # noqa: E402
    BiasEMATeacherConsistencyDetector,
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
        "consistency_require_teacher_correct": False,
        "alpha_consistency_max": 0.2,
        "alpha_consistency_start_epoch": 2,
        "alpha_consistency_warmup_epochs": 3,
        "ema_decay": 0.5,
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


def make_detector():
    torch.manual_seed(11)
    return BiasEMATeacherConsistencyDetector(
        detector_config(),
        backbone=TinyBackbone(feature_dim=4),
    )


def test_detector_is_registered():
    assert (
        DETECTOR["bias_ema_teacher_consistency"]
        is BiasEMATeacherConsistencyDetector
    )


def test_ema_teacher_is_frozen_and_initialized_from_student():
    detector = make_detector()

    assert not any(parameter.requires_grad for parameter in detector.ema_backbone.parameters())
    assert not any(parameter.requires_grad for parameter in detector.ema_head.parameters())
    for ema_parameter, student_parameter in zip(
        detector.ema_backbone.parameters(),
        detector.backbone.parameters(),
    ):
        assert torch.allclose(ema_parameter, student_parameter)
    for ema_parameter, student_parameter in zip(
        detector.ema_head.parameters(),
        detector.head.parameters(),
    ):
        assert torch.allclose(ema_parameter, student_parameter)


def test_ema_update_moves_teacher_toward_student():
    detector = make_detector()
    old_teacher_bias = detector.ema_head.bias.detach().clone()

    with torch.no_grad():
        detector.head.bias.add_(2.0)
        expected = 0.5 * old_teacher_bias + 0.5 * detector.head.bias

    detector.update_ema_teacher()

    assert torch.allclose(detector.ema_head.bias, expected)
    assert detector.ema_update_count.item() == 1


def test_train_forward_and_loss_use_ema_teacher_logits():
    detector = make_detector()
    detector.train()
    detector.epoch = 5

    with torch.no_grad():
        detector.ema_head.weight.zero_()
        detector.ema_head.bias.copy_(torch.tensor([3.0, -3.0]))

    batch = make_batch()
    predictions = detector(batch)
    losses = detector.get_losses(batch, predictions)

    expected_loss, _ = CorrectnessAwareKLLoss(
        temperature=detector.loss_consistency.temperature,
        confidence_threshold=detector.loss_consistency.confidence_threshold,
        require_teacher_correct=detector.loss_consistency.require_teacher_correct,
    )(
        predictions["cls_teacher"],
        predictions["cls_strong"],
        batch["label"],
    )

    assert predictions["cls_teacher"] is not None
    assert not predictions["cls_teacher"].requires_grad
    assert not torch.allclose(predictions["cls_teacher"], predictions["cls_weak"])
    assert torch.allclose(losses["loss_consistency"], expected_loss)
    assert torch.allclose(
        losses["overall"],
        losses["loss_ce"] + losses["alpha_consistency"] * losses["loss_consistency"],
    )

    losses["overall"].backward()
    assert detector.ema_head.weight.grad is None
    assert detector.head.weight.grad is not None


def test_eval_uses_only_student_weak_view():
    detector = make_detector()
    detector.eval()
    batch = make_batch()
    batch.pop("image_strong")

    with torch.no_grad():
        predictions = detector(batch, inference=True)
        losses = detector.get_losses(batch, predictions)

    assert predictions["cls_teacher"] is None
    assert predictions["cls_strong"] is None
    assert losses["loss_consistency"].item() == 0.0
    assert losses["alpha_consistency"].item() == 0.0


def test_training_requires_strong_view():
    detector = make_detector()
    detector.train()
    batch = make_batch()
    batch.pop("image_strong")

    with pytest.raises(RuntimeError, match="image_strong"):
        detector(batch)
