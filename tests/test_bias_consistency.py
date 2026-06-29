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

from detectors.BiasConsistency import (  # noqa: E402
    BiasConsistencyDetector,
    WeakStrongKLLoss,
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


@pytest.fixture
def detector():
    torch.manual_seed(11)
    config = {
        "feature_dim": 4,
        "weight_real": 1.0,
        "weight_fake": 1.0,
        "label_smoothing": 0.1,
        "use_lora": False,
        "train_backbone_bias": True,
        "use_consistency": True,
        "consistency_temperature": 1.0,
        "consistency_confidence_threshold": 0.0,
        "alpha_consistency_max": 0.6,
        "alpha_consistency_start_epoch": 0,
        "alpha_consistency_warmup_epochs": 3,
        "normalize_eps": 1e-6,
        "strict_trainable_check": True,
        "strict_clip_architecture": False,
    }
    return BiasConsistencyDetector(
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


def test_detector_is_registered_with_expected_name():
    assert DETECTOR["bias_consistency"] is BiasConsistencyDetector


def test_consistency_detaches_weak_teacher():
    loss_fn = WeakStrongKLLoss(temperature=1.0)
    weak_logits = torch.tensor(
        [[2.0, -1.0], [0.2, 0.7]],
        requires_grad=True,
    )
    strong_logits = torch.tensor(
        [[0.4, -0.1], [-0.5, 1.2]],
        requires_grad=True,
    )

    loss, selected_fraction, _ = loss_fn(weak_logits, strong_logits)
    loss.backward()

    assert weak_logits.grad is None
    assert strong_logits.grad is not None
    assert strong_logits.grad.norm().item() > 0
    assert selected_fraction.item() == 1.0


def test_identical_predictions_have_zero_consistency():
    loss_fn = WeakStrongKLLoss(temperature=1.0)
    logits = torch.tensor([[1.0, -1.0], [-0.4, 0.9]])

    loss, _, _ = loss_fn(logits, logits.clone())

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_confidence_mask_can_return_differentiable_zero():
    loss_fn = WeakStrongKLLoss(
        temperature=1.0,
        confidence_threshold=0.9,
    )
    weak_logits = torch.zeros(2, 2, requires_grad=True)
    strong_logits = torch.randn(2, 2, requires_grad=True)

    loss, selected_fraction, _ = loss_fn(weak_logits, strong_logits)
    loss.backward()

    assert loss.item() == 0.0
    assert selected_fraction.item() == 0.0
    assert weak_logits.grad is None
    assert strong_logits.grad is not None
    assert torch.count_nonzero(strong_logits.grad).item() == 0


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.0), (1, 0.2), (3, 0.6), (4, 0.6)],
)
def test_progressive_consistency_weight(detector, epoch, expected):
    detector.train()
    detector.epoch = epoch
    assert detector._current_alpha_consistency_value() == pytest.approx(expected)


def test_only_backbone_bias_and_classifier_are_trainable(detector):
    for name, parameter in detector.backbone.named_parameters():
        assert parameter.requires_grad == name.endswith(".bias")
    assert all(parameter.requires_grad for parameter in detector.head.parameters())
    assert not any(
        "lora_A" in name or "lora_B" in name
        for name, _ in detector.named_parameters()
    )


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
        "use_consistency": False,
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

    checked = BiasConsistencyDetector(config, backbone=backbone)
    assert checked.get_trainable_summary()["trainable"] == 18

    config["expected_backbone_bias_params"] = 7
    bad_backbone = TinyBackbone(feature_dim=4)
    bad_backbone.config = backbone.config
    with pytest.raises(RuntimeError, match="bias parameters"):
        BiasConsistencyDetector(
            config,
            backbone=bad_backbone,
        )


def test_train_forward_and_loss_follow_declared_formula(detector):
    detector.train()
    detector.epoch = 2
    batch = make_batch()

    predictions = detector(batch)
    losses = detector.get_losses(batch, predictions)

    assert predictions["cls"].shape == (4, 2)
    assert predictions["cls_weak"].shape == (4, 2)
    assert predictions["cls_strong"].shape == (4, 2)
    assert torch.allclose(
        predictions["cls"],
        0.5 * (predictions["cls_weak"] + predictions["cls_strong"]),
    )
    assert torch.allclose(
        losses["loss_ce"],
        0.5 * (losses["loss_ce_weak"] + losses["loss_ce_strong"]),
    )
    assert torch.allclose(
        losses["overall"],
        losses["loss_ce"]
        + losses["alpha_consistency"] * losses["loss_consistency"],
    )
    assert torch.allclose(
        losses["weighted_consistency"],
        losses["alpha_consistency"] * losses["loss_consistency"],
    )
    assert losses["consistency_selected_fraction"].item() == 1.0
    assert 0.0 <= losses["prediction_agreement"].item() <= 1.0

    losses["overall"].backward()
    assert detector.backbone.proj.weight.grad is None
    assert detector.backbone.scale.grad is None
    assert detector.backbone.proj.bias.grad is not None
    assert detector.backbone.norm.bias.grad is not None
    assert detector.head.weight.grad is not None


def test_eval_uses_one_view_and_disables_consistency(detector):
    detector.eval()
    batch = make_batch()
    batch.pop("image_strong")

    with torch.no_grad():
        predictions = detector(batch, inference=True)
        losses = detector.get_losses(batch, predictions)

    assert predictions["cls_strong"] is None
    assert losses["loss_consistency"].item() == 0.0
    assert losses["alpha_consistency"].item() == 0.0
    assert torch.allclose(losses["overall"], losses["loss_ce"])


def test_training_requires_strong_view(detector):
    detector.train()
    batch = make_batch()
    batch.pop("image_strong")

    with pytest.raises(RuntimeError, match="image_strong"):
        detector(batch)


def test_consistency_collate_keeps_views_separate():
    pytest.importorskip("lmdb")
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

    batch = [
        (torch.ones(3, 2, 2), torch.full((3, 2, 2), 2.0), 0, None, None),
        (torch.full((3, 2, 2), 3.0), torch.full((3, 2, 2), 4.0), 1, None, None),
    ]

    collated = DeepfakeAbstractBaseDataset.collate_fn(batch)

    assert collated["image"].shape == (2, 3, 2, 2)
    assert collated["image_strong"].shape == (2, 3, 2, 2)
    assert collated["label"].tolist() == [0, 1]
    assert collated["image"][0, 0, 0, 0].item() == 1.0
    assert collated["image_strong"][0, 0, 0, 0].item() == 2.0
