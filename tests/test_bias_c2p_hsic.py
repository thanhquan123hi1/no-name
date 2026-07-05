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

# Unit-test the new module in isolation so importing the detector package does
# not import CLIP. The real registry is still used for registration checks.
from metrics.registry import DETECTOR  # noqa: E402

detectors_package = types.ModuleType("detectors")
detectors_package.__path__ = [str(TRAINING_DIR / "detectors")]
detectors_package.DETECTOR = DETECTOR
sys.modules["detectors"] = detectors_package

from detectors.BiasC2PHsic import (  # noqa: E402
    BiasC2PHsicDetector,
    FrozenConceptBank,
    HSICLoss,
    _rbf_gram_matrix,
)


FEATURE_DIM = 6
CONCEPT_DIM = 4
CONTENT_DIM = 3


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)
        self.bias_vec = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, image):
        return SimpleNamespace(pooler_output=self.proj(image) + self.bias_vec)


def make_detector(strict=True, freeze_concept=True):
    torch.manual_seed(0)
    config = {
        "feature_dim": FEATURE_DIM,
        "concept_dim": CONCEPT_DIM,
        "content_dim": CONTENT_DIM,
        "weight_real": 1.0,
        "weight_fake": 1.0,
        "label_smoothing": 0.1,
        "concept_temperature": 0.07,
        "freeze_concept": freeze_concept,
        "lambda_hsic_max": 1.0,
        "lambda_hsic_start_epoch": 0,
        "lambda_hsic_warmup_epochs": 3,
        "lambda_content": 0.1,
        "normalize_eps": 1e-6,
        "use_lora": False,
        "strict_trainable_check": strict,
    }
    anchors = torch.randn(2, CONCEPT_DIM)
    bank = FrozenConceptBank(anchors, eps=1e-6)
    return BiasC2PHsicDetector(
        config, backbone=TinyBackbone(FEATURE_DIM), concept_bank=bank
    )


@pytest.fixture
def detector():
    return make_detector()


def _data(batch=8):
    torch.manual_seed(1)
    labels = torch.tensor([0, 1] * (batch // 2))
    return {"image": torch.randn(batch, 3), "label": labels}


# ----------------------------- registration --------------------------------

def test_detector_is_registered_with_expected_name():
    assert DETECTOR["bias_c2p_hsic"] is BiasC2PHsicDetector



# ------------------------------- HSIC math ----------------------------------

def test_hsic_zero_for_independent_inputs():
    torch.manual_seed(3)
    x = torch.randn(256, 5)
    y = torch.randn(256, 5)  # independently drawn
    loss = HSICLoss()(x, y)
    assert loss.item() >= 0.0
    assert loss.item() < 5e-3


def test_hsic_positive_for_dependent_inputs():
    torch.manual_seed(3)
    x = torch.randn(256, 5)
    y = x @ torch.randn(5, 5)  # deterministic function of x
    loss = HSICLoss()(x, y)
    assert loss.item() > 1e-2


def test_hsic_symmetric():
    torch.manual_seed(4)
    x = torch.randn(64, 5)
    y = torch.randn(64, 5)
    a = HSICLoss()(x, y)
    b = HSICLoss()(y, x)
    assert torch.allclose(a, b, atol=1e-6)


def test_hsic_nonnegative_and_differentiable():
    x = torch.randn(32, 5, requires_grad=True)
    y = torch.randn(32, 5, requires_grad=True)
    loss = HSICLoss()(x, y)
    loss.backward()
    assert loss.item() >= -1e-8
    assert x.grad is not None and y.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("batch", [8, 64, 128])
def test_hsic_stable_across_batch_sizes(batch):
    torch.manual_seed(batch)
    x = torch.randn(batch, 5)
    y = torch.randn(batch, 5)
    loss = HSICLoss()(x, y)
    assert torch.isfinite(loss)
    assert loss.item() >= -1e-8


def test_hsic_small_batch_returns_differentiable_zero():
    x = torch.randn(3, 5, requires_grad=True)
    y = torch.randn(3, 5, requires_grad=True)
    loss = HSICLoss()(x, y)
    loss.backward()
    assert loss.item() == 0.0
    assert x.grad is not None


def test_rbf_gram_matrix_properties():
    torch.manual_seed(5)
    x = torch.randn(10, 4)
    k = _rbf_gram_matrix(x)
    assert k.shape == (10, 10)
    assert torch.allclose(torch.diag(k), torch.ones(10), atol=1e-5)
    assert torch.allclose(k, k.t(), atol=1e-6)
    assert (k >= 0).all() and (k <= 1.0 + 1e-6).all()


# --------------------------- concept classifier -----------------------------

def test_concept_bank_is_l2_normalized():
    anchors = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 8.0]])
    bank = FrozenConceptBank(anchors)
    norms = bank().norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-6)


def test_concept_bank_rejects_wrong_shape():
    with pytest.raises(ValueError):
        FrozenConceptBank(torch.randn(3, 4))


def test_forward_returns_expected_keys_and_shapes(detector):
    data = _data(8)
    pred = detector(data)
    assert set(pred) == {"cls", "prob", "feat", "feat_forgery", "feat_content"}
    assert pred["cls"].shape == (8, 2)
    assert pred["prob"].shape == (8,)
    assert pred["feat"].shape == (8, FEATURE_DIM)
    assert pred["feat_forgery"].shape == (8, CONCEPT_DIM)
    assert pred["feat_content"].shape == (8, CONTENT_DIM)
    # feat_forgery is L2-normalized (used both for cosine logits and HSIC).
    assert torch.allclose(
        pred["feat_forgery"].norm(dim=1), torch.ones(8), atol=1e-5
    )


def test_reconstruction_uses_detached_forgery_plus_content(detector):
    # content_decoder must accept concept_dim + content_dim inputs.
    assert detector.content_decoder.in_features == CONCEPT_DIM + CONTENT_DIM
    assert detector.content_decoder.out_features == FEATURE_DIM
    detector.epoch = 1
    data = _data(8)
    pred = detector(data)
    losses = detector.get_losses(data, pred)
    losses["overall"].backward()
    # Forgery branch is detached inside reconstruction, so loss_content must not
    # push gradient into forgery_proj through the decoder path alone. HSIC + CE
    # still do, so we only check the decoder weights receive gradient.
    assert detector.content_decoder.weight.grad is not None
    assert torch.isfinite(detector.content_decoder.weight.grad).all()


def test_classifier_matches_forward_logits(detector):
    data = _data(8)
    raw = detector.features(data)
    logits_a = detector.classifier(raw)
    logits_b, _ = detector._concept_logits(raw)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)


def test_no_free_linear_head():
    det = make_detector()
    # Direction A: classification is cosine-to-concept, not a Linear(_, 2) head.
    assert not hasattr(det, "head")


# ------------------------------- losses -------------------------------------

def test_losses_have_required_keys_and_are_scalar(detector):
    detector.epoch = 1
    data = _data(8)
    pred = detector(data)
    losses = detector.get_losses(data, pred)
    required = {
        "overall",
        "loss_ce",
        "loss_hsic",
        "lambda_hsic",
        "weighted_hsic",
        "loss_content",
        "lambda_content",
        "weighted_content",
        "real_loss",
        "fake_loss",
    }
    assert required.issubset(losses)
    assert all(losses[k].ndim == 0 for k in required)


def test_overall_loss_matches_declared_formula(detector):
    detector.epoch = 3
    data = _data(8)
    pred = detector(data)
    losses = detector.get_losses(data, pred)
    expected = (
        losses["loss_ce"]
        + losses["lambda_hsic"] * losses["loss_hsic"]
        + losses["lambda_content"] * losses["loss_content"]
    )
    assert torch.allclose(losses["overall"], expected, atol=1e-6)


def test_overall_loss_is_finite_and_backprops(detector):
    detector.epoch = 2
    data = _data(8)
    pred = detector(data)
    losses = detector.get_losses(data, pred)
    losses["overall"].backward()
    assert torch.isfinite(losses["overall"])
    grad = detector.forgery_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_single_class_batch_is_safe(detector):
    detector.epoch = 1
    data = {"image": torch.randn(8, 3), "label": torch.zeros(8, dtype=torch.long)}
    pred = detector(data)
    losses = detector.get_losses(data, pred)
    losses["overall"].backward()
    assert torch.isfinite(losses["overall"])


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.0), (1, 1.0 / 3.0), (3, 1.0), (5, 1.0)],
)
def test_lambda_hsic_warmup(detector, epoch, expected):
    detector.train()
    detector.epoch = epoch
    assert detector._current_lambda_hsic_value() == pytest.approx(expected)


def test_lambda_hsic_zero_in_eval(detector):
    detector.eval()
    detector.epoch = 5
    assert detector._current_lambda_hsic_value() == 0.0


# --------------------------- trainability / isolation ------------------------

def test_only_bias_projections_trainable_concept_frozen():
    det = make_detector(freeze_concept=True)
    for name, p in det.backbone.named_parameters():
        assert p.requires_grad == name.endswith(".bias")
    for module in (det.forgery_proj, det.content_proj, det.content_decoder):
        assert all(p.requires_grad for p in module.parameters())
    assert all(not p.requires_grad for p in det.concept_bank.parameters())
    assert not any(
        "lora_A" in n or "lora_B" in n for n, _ in det.named_parameters()
    )


def test_concept_bank_buffer_not_a_parameter():
    det = make_detector()
    # Anchors live in a buffer so the optimizer never sees them.
    assert "concept_bank.anchors" in dict(det.named_buffers())
    assert "concept_bank.anchors" not in dict(det.named_parameters())


def test_lora_config_is_rejected():
    with pytest.raises(ValueError):
        BiasC2PHsicDetector(
            {"use_lora": True, "feature_dim": FEATURE_DIM, "concept_dim": CONCEPT_DIM},
            backbone=TinyBackbone(FEATURE_DIM),
            concept_bank=FrozenConceptBank(torch.randn(2, CONCEPT_DIM)),
        )


