import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_detector import AbstractDetector
from detectors import DETECTOR


logger = logging.getLogger(__name__)


def _rbf_gram_matrix(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """RBF Gram matrix with a per-batch, detached median-heuristic bandwidth.

    Squared distances are computed directly (never via sqrt) so the gradient is
    finite even on the zero-distance diagonal. Only the bandwidth is detached;
    gradients still flow through the kernel values, which is what lets HSIC act
    as a differentiable dependence penalty.
    """
    x_sq = (x * x).sum(dim=1, keepdim=True)  # [B, 1]
    sq_dist = x_sq + x_sq.t() - 2.0 * (x @ x.t())  # [B, B]
    sq_dist = sq_dist.clamp_min(0.0)  # numerical guard, keeps diagonal at 0
    batch_size = x.size(0)
    with torch.no_grad():
        mask = torch.triu(
            torch.ones(batch_size, batch_size, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        off_diagonal = sq_dist[mask]
        if off_diagonal.numel() == 0:
            sigma_sq = x.new_tensor(1.0)
        else:
            sigma_sq = torch.clamp(torch.median(off_diagonal), min=eps)
    return torch.exp(-sq_dist / (2.0 * sigma_sq))


class HSICLoss(nn.Module):
    """Biased empirical HSIC with RBF kernels.

    HSIC(X, Y) = 0 iff X and Y are independent (for a characteristic kernel),
    and is strictly positive when they are dependent. Minimizing it therefore
    pushes the two feature branches to be statistically independent.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("HSIC inputs must be 2D tensors of shape [B, D].")
        if x.size(0) != y.size(0):
            raise ValueError("HSIC inputs must share the same batch dimension.")

        batch_size = x.size(0)
        # HSIC needs a few samples for a meaningful kernel estimate; return a
        # differentiable zero otherwise so backward() stays valid.
        if batch_size < 4:
            return (x.sum() + y.sum()) * 0.0

        k = _rbf_gram_matrix(x, self.eps)
        l = _rbf_gram_matrix(y, self.eps)
        centering = (
            torch.eye(batch_size, device=x.device, dtype=x.dtype) - 1.0 / batch_size
        )
        khl = k @ centering @ l @ centering
        return torch.trace(khl) / ((batch_size - 1) ** 2)


class FrozenConceptBank(nn.Module):
    """Holds fixed, L2-normalized real/fake concept anchors in CLIP text space.

    Anchors are stored as a buffer (not a parameter) so they are never updated
    by the optimizer. This is the fixed-anchor variant of concept guidance:
    the text-derived concepts stay frozen and the image side learns to align to
    them. Learnable prompts (full C2P) are intentionally out of scope here.
    """

    def __init__(self, anchors: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        if anchors.dim() != 2 or anchors.size(0) != 2:
            raise ValueError(
                "Concept anchors must have shape [2, D] ordered as (real, fake)."
            )
        self.eps = float(eps)
        self.register_buffer(
            "anchors", F.normalize(anchors, p=2, dim=1, eps=self.eps)
        )

    def forward(self) -> torch.Tensor:
        return self.anchors


@DETECTOR.register_module(module_name="bias_c2p_hsic")
class BiasC2PHsicDetector(AbstractDetector):
    """CLIP ViT-L/14 with bias tuning, cosine-to-concept, and HSIC decoupling.

    Direction A of the plan:
      * classification = cosine similarity of a forgery-projected image feature
        to fixed text concept anchors (no free linear head);
      * a separate content branch is pushed to be statistically independent of
        the forgery branch via an HSIC penalty, so the forgery concept cannot
        carry FF++ domain bias (compression / identity);
      * a reconstruction term rebuilds the CLS feature from [forgery.detach() +
        content], so the content branch only has to supply the *residual* that
        forgery lacks. This keeps content informative without fighting HSIC.

    This detector is fully self-contained: it shares no state with the existing
    Bias* detectors and uses its own trainability validator.
    """

    def __init__(
        self,
        config=None,
        backbone: Optional[nn.Module] = None,
        concept_bank: Optional[nn.Module] = None,
    ) -> None:
        super(BiasC2PHsicDetector, self).__init__()
        self.config = config or {}

        if bool(self.config.get("use_lora", False)):
            raise ValueError("BiasC2PHsicDetector does not support LoRA.")
        if not bool(self.config.get("train_backbone_bias", True)):
            raise ValueError("BiasC2PHsicDetector requires train_backbone_bias=true.")

        self.feature_dim = int(self.config.get("feature_dim", 1024))
        self.concept_dim = int(self.config.get("concept_dim", 768))
        self.content_dim = int(self.config.get("content_dim", 256))
        self.normalize_eps = float(self.config.get("normalize_eps", 1e-6))
        self.temperature = float(self.config.get("concept_temperature", 0.1))
        # Fixed-anchor variant: concepts are always frozen (buffer, no params).
        self.freeze_concept = True
        self.warm_start_forgery_proj = bool(
            self.config.get("warm_start_forgery_proj", True)
        )
        self.strict_trainable_check = bool(
            self.config.get("strict_trainable_check", True)
        )
        self.epoch = int(self.config.get("start_epoch", 0))
        if self.temperature <= 0:
            raise ValueError("concept_temperature must be > 0.")

        logger.info("Loading CLIP ViT-L/14 for Bias + cosine-to-concept + HSIC.")
        self.backbone = (
            backbone if backbone is not None else self.build_backbone(self.config)
        )

        self.forgery_proj = nn.Linear(self.feature_dim, self.concept_dim)
        self.content_proj = nn.Linear(self.feature_dim, self.content_dim)
        # Reconstruct CLS from [forgery (detached) + content] -> residual coding.
        self.content_decoder = nn.Linear(
            self.concept_dim + self.content_dim, self.feature_dim
        )

        self.concept_bank = (
            concept_bank
            if concept_bank is not None
            else self.build_concept_bank(self.config)
        )

        self.build_loss(self.config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        self._setup_trainable_parameters()

    def build_backbone(self, config):
        # Import lazily so unit tests can inject a tiny backbone without CLIP.
        from transformers import CLIPModel

        model_name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        try:
            clip_model = CLIPModel.from_pretrained(model_name)
        except Exception:
            clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)

        # Optionally warm-start the forgery projection from the frozen CLIP
        # visual_projection so cosine-to-concept is meaningful from step 0.
        if self.warm_start_forgery_proj:
            self._pending_visual_projection = (
                clip_model.visual_projection.weight.detach().clone()
            )
        return clip_model.vision_model

    def _maybe_warm_start_forgery_proj(self) -> None:
        weight = getattr(self, "_pending_visual_projection", None)
        if weight is None:
            return
        if tuple(weight.shape) == tuple(self.forgery_proj.weight.shape):
            with torch.no_grad():
                self.forgery_proj.weight.copy_(weight)
                self.forgery_proj.bias.zero_()
            logger.info("Warm-started forgery_proj from CLIP visual_projection.")
        else:
            logger.warning(
                "Skipping forgery_proj warm-start: shape mismatch %s vs %s.",
                tuple(weight.shape),
                tuple(self.forgery_proj.weight.shape),
            )
        self._pending_visual_projection = None

    def build_concept_bank(self, config) -> nn.Module:
        # Import lazily; this path is only used in real training, not unit tests.
        from transformers import CLIPModel, CLIPTokenizer

        model_name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        real_prompts = list(
            config.get(
                "concept_real_prompts",
                ["a real face", "an authentic photo of a face", "a genuine human face"],
            )
        )
        fake_prompts = list(
            config.get(
                "concept_fake_prompts",
                [
                    "a deepfake face",
                    "a manipulated face",
                    "a face forgery",
                    "a synthetic fake face",
                ],
            )
        )
        try:
            clip_model = CLIPModel.from_pretrained(model_name)
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
        except Exception:
            clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
            tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=True)
        clip_model.eval()
        with torch.no_grad():
            real_feats = clip_model.get_text_features(
                **tokenizer(real_prompts, padding=True, return_tensors="pt")
            )
            fake_feats = clip_model.get_text_features(
                **tokenizer(fake_prompts, padding=True, return_tensors="pt")
            )
            real_anchor = F.normalize(real_feats, p=2, dim=1).mean(dim=0)
            fake_anchor = F.normalize(fake_feats, p=2, dim=1).mean(dim=0)
            anchors = torch.stack([real_anchor, fake_anchor], dim=0)
        if anchors.size(1) != self.concept_dim:
            raise RuntimeError(
                "Concept anchor dim mismatch: text features have "
                f"{anchors.size(1)} dims but concept_dim={self.concept_dim}."
            )
        return FrozenConceptBank(anchors, eps=self.normalize_eps)

    def build_loss(self, config) -> None:
        class_weights = torch.tensor(
            [
                float(config.get("weight_real", 1.0)),
                float(config.get("weight_fake", 1.0)),
            ],
            dtype=torch.float32,
        )
        self.loss_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(config.get("label_smoothing", 0.1)),
        )
        self.loss_hsic = HSICLoss(eps=1e-12)

        self.lambda_hsic_max = float(config.get("lambda_hsic_max", 1.0))
        self.lambda_hsic_start_epoch = int(config.get("lambda_hsic_start_epoch", 0))
        self.lambda_hsic_warmup_epochs = int(
            config.get("lambda_hsic_warmup_epochs", 3)
        )
        self.lambda_content = float(config.get("lambda_content", 0.1))
        if self.lambda_hsic_max < 0:
            raise ValueError("lambda_hsic_max must be >= 0.")
        if self.lambda_hsic_warmup_epochs < 0:
            raise ValueError("lambda_hsic_warmup_epochs must be >= 0.")
        if self.lambda_content < 0:
            raise ValueError("lambda_content must be >= 0.")

    def _setup_trainable_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for name, parameter in self.backbone.named_parameters():
            if name.endswith(".bias"):
                parameter.requires_grad = True

        self._maybe_warm_start_forgery_proj()

        for module in (self.forgery_proj, self.content_proj, self.content_decoder):
            for parameter in module.parameters():
                parameter.requires_grad = True

        # Concepts are frozen anchors (buffer); nothing to toggle here.

        if self.strict_trainable_check:
            self._validate_trainable_parameters()

        self.trainable_param_summary = self._summarize_trainable_parameters()
        logger.info(
            "BiasC2PHsic initialized. Trainable params: %s / %s (%.4f%%).",
            f"{self.trainable_param_summary['trainable']:,}",
            f"{self.trainable_param_summary['total']:,}",
            self.trainable_param_summary["percent"],
        )

    def _validate_trainable_parameters(self) -> None:
        lora = [
            name
            for name, _ in self.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        if lora:
            raise RuntimeError(
                f"LoRA parameters are forbidden in BiasC2PHsicDetector: {lora[:20]}"
            )

        invalid_backbone = []
        for name, parameter in self.backbone.named_parameters():
            should_train = name.endswith(".bias")
            if parameter.requires_grad != should_train:
                invalid_backbone.append(name)
        if invalid_backbone:
            raise RuntimeError(
                "Backbone trainability must be restricted to parameters ending in "
                f"'.bias'. Invalid parameters: {invalid_backbone[:20]}"
            )

        for label, module in (
            ("forgery_proj", self.forgery_proj),
            ("content_proj", self.content_proj),
            ("content_decoder", self.content_decoder),
        ):
            frozen = [
                pname
                for pname, param in module.named_parameters()
                if not param.requires_grad
            ]
            if frozen:
                raise RuntimeError(f"{label} parameters must be trainable: {frozen}")

        # Concept anchors must never be trainable (they live in a buffer).
        active = [
            name
            for name, param in self.concept_bank.named_parameters()
            if param.requires_grad
        ]
        if active:
            raise RuntimeError(f"Concept bank must be frozen: {active}")

    def _summarize_trainable_parameters(self) -> Dict[str, float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "percent": 100.0 * trainable / max(total, 1),
        }

    def get_trainable_summary(self) -> Dict[str, float]:
        return self.trainable_param_summary

    def _current_lambda_hsic_value(self) -> float:
        if not self.training:
            return 0.0
        epoch = float(getattr(self, "epoch", 0))
        if self.lambda_hsic_warmup_epochs == 0:
            progress = 1.0 if epoch >= self.lambda_hsic_start_epoch else 0.0
        else:
            progress = (
                epoch - self.lambda_hsic_start_epoch
            ) / self.lambda_hsic_warmup_epochs
            progress = min(max(progress, 0.0), 1.0)
        return self.lambda_hsic_max * progress

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(images)
        if not hasattr(outputs, "pooler_output"):
            raise RuntimeError("The vision backbone must return pooler_output.")
        raw_features = outputs.pooler_output
        if raw_features.dim() != 2:
            raise ValueError("Backbone pooler_output must have shape [B, D].")
        if raw_features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected pooler feature dimension {self.feature_dim}, "
                f"got {raw_features.size(1)}."
            )
        return raw_features

    def _concept_logits(
        self, raw_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        forgery = self.forgery_proj(raw_features)
        forgery_norm = F.normalize(forgery, p=2, dim=1, eps=self.normalize_eps)
        anchors = self.concept_bank()
        anchors_norm = F.normalize(anchors, p=2, dim=1, eps=self.normalize_eps)
        logits = (forgery_norm @ anchors_norm.t()) / self.temperature
        return logits, forgery_norm

    def features(self, data_dict: dict) -> torch.Tensor:
        return self._encode(data_dict["image"])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        logits, _ = self._concept_logits(features)
        return logits

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self._encode(data_dict["image"])
        logits, forgery_norm = self._concept_logits(raw_features)
        content = self.content_proj(raw_features)
        fake_probability = torch.softmax(logits, dim=1)[:, 1]
        return {
            "cls": logits,
            "prob": fake_probability,
            "feat": raw_features,
            "feat_forgery": forgery_norm,
            "feat_content": content,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"].contiguous().view(-1)
        logits = pred_dict["cls"]
        forgery = pred_dict["feat_forgery"]
        content = pred_dict["feat_content"]
        raw_features = pred_dict["feat"]
        zero = logits.sum() * 0.0

        loss_ce = self.loss_ce(logits, labels)
        # Decouple the (normalized) forgery representation from content.
        loss_hsic = torch.nan_to_num(
            self.loss_hsic(forgery, content), nan=0.0, posinf=0.0, neginf=0.0
        )
        # Reconstruct CLS from [forgery.detach() + content]; content only codes
        # the residual, so this does not fight the HSIC independence objective.
        decoder_input = torch.cat([forgery.detach(), content], dim=1)
        reconstruction = self.content_decoder(decoder_input)
        loss_content = torch.nan_to_num(
            F.mse_loss(reconstruction, raw_features.detach()),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        lambda_hsic = logits.new_tensor(self._current_lambda_hsic_value())
        lambda_content = logits.new_tensor(self.lambda_content)
        weighted_hsic = lambda_hsic * loss_hsic
        weighted_content = lambda_content * loss_content
        overall_loss = loss_ce + weighted_hsic + weighted_content

        loss_dict = {
            "overall": overall_loss,
            "loss_ce": loss_ce,
            "loss_hsic": loss_hsic,
            "lambda_hsic": lambda_hsic,
            "weighted_hsic": weighted_hsic,
            "loss_content": loss_content,
            "lambda_content": lambda_content,
            "weighted_content": weighted_content,
        }

        with torch.no_grad():
            real_mask = labels.eq(0)
            fake_mask = labels.eq(1)
            loss_dict["real_loss"] = (
                self.loss_ce(logits[real_mask], labels[real_mask])
                if real_mask.any()
                else zero.detach()
            )
            loss_dict["fake_loss"] = (
                self.loss_ce(logits[fake_mask], labels[fake_mask])
                if fake_mask.any()
                else zero.detach()
            )

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        from metrics.base_metrics_class import calculate_metrics_for_train

        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(),
            pred_dict["cls"].detach(),
        )
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
