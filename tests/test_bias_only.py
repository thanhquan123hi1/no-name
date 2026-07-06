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

from detectors.BiasOnly import BiasOnlyDetector  # noqa: E402


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


@pytest.fixture
def detector():
    torch.manual_seed(17)
    config = {
        "feature_dim": 4,
        "weight_real": 1.0,
        "weight_fake": 1.0,
        "label_smoothing": 0.1,
        "use_lora": False,
        "train_backbone_bias": True,
        "normalize_eps": 1e-6,
        "strict_trainable_check": True,
        "strict_clip_architecture": False,
    }
    return BiasOnlyDetector(
        config,
        backbone=TinyBackbone(feature_dim=4),
    )


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
        "label": torch.tensor([0, 0, 1, 1]),
    }


def test_detector_is_registered_with_expected_name():
    assert DETECTOR["bias_only"] is BiasOnlyDetector


def test_only_backbone_bias_and_classifier_are_trainable(detector):
    for name, parameter in detector.backbone.named_parameters():
        assert parameter.requires_grad == name.endswith(".bias")
    assert all(parameter.requires_grad for parameter in detector.head.parameters())
    assert not any(
        "lora_A" in name or "lora_B" in name
        for name, _ in detector.named_parameters()
    )


def test_train_forward_and_loss_are_ce_only(detector):
    detector.train()
    batch = make_batch()

    predictions = detector(batch)
    losses = detector.get_losses(batch, predictions)

    assert predictions["cls"].shape == (4, 2)
    assert predictions["prob"].shape == (4,)
    assert predictions["feat"].shape == (4, 4)
    assert predictions["feat_norm"].shape == (4, 4)
    assert set(losses) == {"overall", "loss_ce", "real_loss", "fake_loss"}
    assert torch.allclose(losses["overall"], losses["loss_ce"])

    losses["overall"].backward()
    assert detector.backbone.proj.weight.grad is None
    assert detector.backbone.scale.grad is None
    assert detector.backbone.proj.bias.grad is not None
    assert detector.backbone.norm.bias.grad is not None
    assert detector.head.weight.grad is not None


def test_classifier_matches_forward_logits(detector):
    batch = make_batch()
    predictions = detector(batch)

    logits_from_classifier = detector.classifier(predictions["feat"])

    assert torch.allclose(predictions["cls"], logits_from_classifier)


def test_strict_architecture_guard_checks_bias_count():
    backbone = TinyBackbone(feature_dim=4)
    backbone.config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    config = {
        "feature_dim": 4,
        "use_lora": False,
        "train_backbone_bias": True,
        "strict_clip_architecture": True,
        "expected_clip_hidden_size": 4,
        "expected_clip_intermediate_size": 8,
        "expected_clip_num_hidden_layers": 2,
        "expected_clip_num_attention_heads": 2,
        "expected_clip_image_size": 8,
        "expected_clip_patch_size": 2,
        # proj.bias (4) + norm.bias (4)
        "expected_backbone_bias_params": 8,
    }

    checked = BiasOnlyDetector(config, backbone=backbone)
    assert checked.get_trainable_summary()["trainable"] == 18

    config["expected_backbone_bias_params"] = 7
    bad_backbone = TinyBackbone(feature_dim=4)
    bad_backbone.config = backbone.config
    with pytest.raises(RuntimeError, match="bias parameters"):
        BiasOnlyDetector(
            config,
            backbone=bad_backbone,
        )
