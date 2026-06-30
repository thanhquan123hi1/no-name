import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_detector import AbstractDetector
from detectors import DETECTOR


logger = logging.getLogger(__name__)


class WeakStrongKLLoss(nn.Module):
    """KL consistency from a detached weak-view teacher to a strong-view student."""

    def __init__(
        self,
        temperature: float = 1.0,
        confidence_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("consistency_temperature must be > 0.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("consistency_confidence_threshold must be in [0, 1].")

        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)

    def forward(
        self,
        weak_logits: torch.Tensor,
        strong_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weak_logits.shape != strong_logits.shape:
            raise ValueError(
                "Weak and strong logits must have the same shape, got "
                f"{tuple(weak_logits.shape)} and {tuple(strong_logits.shape)}."
            )
        if weak_logits.dim() != 2:
            raise ValueError("Consistency logits must have shape [B, C].")

        temperature = self.temperature

        teacher_prob = F.softmax(weak_logits / temperature, dim=1).detach()
        student_log_prob = F.log_softmax(strong_logits / temperature, dim=1)

        per_sample_kl = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="none",
        ).sum(dim=1) * (temperature**2)

        teacher_confidence = teacher_prob.max(dim=1).values
        selected = teacher_confidence.ge(self.confidence_threshold)

        if selected.any():
            loss = per_sample_kl[selected].mean()
        else:
            loss = strong_logits.sum() * 0.0

        selected_fraction = selected.to(strong_logits.dtype).mean()
        mean_teacher_confidence = teacher_confidence.mean()

        return loss, selected_fraction, mean_teacher_confidence


@DETECTOR.register_module(module_name="siglip_bias_consistency")
class SigLIPBiasConsistencyDetector(AbstractDetector):
    """
    SigLIP vision encoder with BitFit-style bias tuning, CE, and optional consistency.

    Trainable:
    - SigLIP backbone bias parameters only
    - classifier head
    """

    def __init__(self, config=None, backbone: Optional[nn.Module] = None) -> None:
        super(SigLIPBiasConsistencyDetector, self).__init__()
        self.config = config or {}

        if bool(self.config.get("use_lora", False)):
            raise ValueError("SigLIPBiasConsistencyDetector does not support LoRA.")

        if not bool(self.config.get("train_backbone_bias", True)):
            raise ValueError(
                "SigLIPBiasConsistencyDetector requires train_backbone_bias=true."
            )

        self.normalize_eps = float(self.config.get("normalize_eps", 1e-6))
        self.use_consistency = bool(self.config.get("use_consistency", True))

        if self.use_consistency and not bool(
            self.config.get("use_consistency_views", True)
        ):
            raise ValueError(
                "use_consistency=true requires use_consistency_views=true."
            )

        self.strict_trainable_check = bool(
            self.config.get("strict_trainable_check", True)
        )
        self.strict_siglip_architecture = bool(
            self.config.get("strict_siglip_architecture", False)
        )
        self.epoch = int(self.config.get("start_epoch", 0))

        logger.info("Loading SigLIP vision encoder for Bias + CE + Consistency.")
        self.backbone = (
            backbone if backbone is not None else self.build_backbone(self.config)
        )

        # Auto-detect feature dim from SigLIP config.
        # Example:
        # google/siglip-base-patch16-224 -> hidden_size usually 768
        # google/siglip-large-patch16-256 -> hidden_size usually 1024
        backbone_config = getattr(self.backbone, "config", None)
        detected_dim = getattr(backbone_config, "hidden_size", None)

        self.feature_dim = int(
            self.config.get(
                "feature_dim",
                detected_dim if detected_dim is not None else 1024,
            )
        )

        self.head = nn.Linear(self.feature_dim, 2)

        self.build_loss(self.config)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        self._setup_trainable_parameters()

        if self.strict_siglip_architecture:
            self._validate_siglip_architecture()

    def build_backbone(self, config):
        # Import lazily so unit tests can inject a tiny backbone without loading SigLIP.
        from transformers import SiglipVisionModel

        model_name = config.get(
            "siglip_model_name",
            "google/siglip-large-patch16-256",
        )

        try:
            siglip_vision_model = SiglipVisionModel.from_pretrained(model_name)
        except Exception:
            siglip_vision_model = SiglipVisionModel.from_pretrained(
                model_name,
                local_files_only=True,
            )

        # Use SigLIP vision encoder only, no text encoder, no projection head.
        return siglip_vision_model

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

        self.alpha_consistency_max = float(
            config.get("alpha_consistency_max", 0.5)
        )
        self.alpha_consistency_start_epoch = int(
            config.get("alpha_consistency_start_epoch", 0)
        )
        self.alpha_consistency_warmup_epochs = int(
            config.get("alpha_consistency_warmup_epochs", 3)
        )

        if self.alpha_consistency_max < 0:
            raise ValueError("alpha_consistency_max must be >= 0.")

        if self.alpha_consistency_warmup_epochs < 0:
            raise ValueError("alpha_consistency_warmup_epochs must be >= 0.")

        self.loss_consistency = WeakStrongKLLoss(
            temperature=float(config.get("consistency_temperature", 1.0)),
            confidence_threshold=float(
                config.get("consistency_confidence_threshold", 0.0)
            ),
        )

    def _setup_trainable_parameters(self) -> None:
        # Freeze all SigLIP parameters.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # BitFit: train only bias parameters in SigLIP backbone.
        for name, parameter in self.backbone.named_parameters():
            if name.endswith(".bias"):
                parameter.requires_grad = True

        # Train classifier head.
        for parameter in self.head.parameters():
            parameter.requires_grad = True

        if self.strict_trainable_check:
            self._validate_trainable_parameters()

        self.trainable_param_summary = self._summarize_trainable_parameters()

        logger.info(
            "SigLIPBiasConsistency initialized. Trainable params: %s / %s (%.4f%%).",
            f"{self.trainable_param_summary['trainable']:,}",
            f"{self.trainable_param_summary['total']:,}",
            self.trainable_param_summary["percent"],
        )

    def _validate_trainable_parameters(self) -> None:
        forbidden = [
            name
            for name, _ in self.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]

        if forbidden:
            raise RuntimeError(
                "LoRA parameters are forbidden in SigLIPBiasConsistencyDetector: "
                f"{forbidden[:20]}"
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

        frozen_head = [
            name
            for name, parameter in self.head.named_parameters()
            if not parameter.requires_grad
        ]

        if frozen_head:
            raise RuntimeError(
                f"Classifier parameters must all be trainable: {frozen_head}"
            )

    def _validate_siglip_architecture(self) -> None:
        siglip_config = getattr(self.backbone, "config", None)

        if siglip_config is None:
            raise RuntimeError("SigLIP backbone does not expose a vision config.")

        expected = {
            "hidden_size": int(
                self.config.get("expected_siglip_hidden_size", self.feature_dim)
            ),
            "image_size": int(
                self.config.get("expected_siglip_image_size", 256)
            ),
            "patch_size": int(
                self.config.get("expected_siglip_patch_size", 16)
            ),
        }

        mismatches = {
            name: (getattr(siglip_config, name, None), expected_value)
            for name, expected_value in expected.items()
            if getattr(siglip_config, name, None) != expected_value
        }

        if mismatches:
            raise RuntimeError(
                f"Unexpected SigLIP vision architecture (actual, expected): {mismatches}"
            )

        expected_bias_params = self.config.get("expected_backbone_bias_params", None)

        if expected_bias_params is not None:
            expected_bias_params = int(expected_bias_params)

            actual_bias_params = sum(
                parameter.numel()
                for name, parameter in self.backbone.named_parameters()
                if name.endswith(".bias")
            )

            if actual_bias_params != expected_bias_params:
                raise RuntimeError(
                    "Unexpected number of SigLIP backbone bias parameters: "
                    f"actual={actual_bias_params:,}, "
                    f"expected={expected_bias_params:,}."
                )

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

    def _current_alpha_consistency_value(self) -> float:
        if not self.training or not self.use_consistency:
            return 0.0

        epoch = float(getattr(self, "epoch", 0))

        if self.alpha_consistency_warmup_epochs == 0:
            progress = (
                1.0 if epoch >= self.alpha_consistency_start_epoch else 0.0
            )
        else:
            progress = (
                epoch - self.alpha_consistency_start_epoch
            ) / self.alpha_consistency_warmup_epochs
            progress = min(max(progress, 0.0), 1.0)

        return self.alpha_consistency_max * progress

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            pixel_values=images,
            return_dict=True,
        )

        # SiglipVisionModel usually returns pooler_output.
        # Fallback to mean pooling if pooler_output is missing.
        if getattr(outputs, "pooler_output", None) is not None:
            raw_features = outputs.pooler_output
        elif getattr(outputs, "last_hidden_state", None) is not None:
            raw_features = outputs.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError(
                "SigLIP vision backbone must return pooler_output or last_hidden_state."
            )

        if raw_features.dim() != 2:
            raise ValueError("Backbone feature output must have shape [B, D].")

        if raw_features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {raw_features.size(1)}. "
                "Set config['feature_dim'] to match the selected SigLIP model."
            )

        normalized_features = F.normalize(
            raw_features,
            p=2,
            dim=1,
            eps=self.normalize_eps,
        )

        logits = self.head(normalized_features)

        return raw_features, normalized_features, logits

    def features(self, data_dict: dict) -> torch.Tensor:
        raw_features, _, _ = self._encode_images(data_dict["image"])
        return raw_features

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(
            features,
            p=2,
            dim=1,
            eps=self.normalize_eps,
        )
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        weak_raw, weak_norm, weak_logits = self._encode_images(data_dict["image"])

        strong_logits = None
        strong_raw = None
        strong_norm = None

        consistency_active = self.training and self.use_consistency and not inference

        if consistency_active:
            if data_dict.get("image_strong") is None:
                raise RuntimeError(
                    "Consistency training requires data_dict['image_strong']. "
                    "Enable use_consistency_views in the dataset config."
                )

            strong_raw, strong_norm, strong_logits = self._encode_images(
                data_dict["image_strong"]
            )

            logits = 0.5 * (weak_logits + strong_logits)
        else:
            logits = weak_logits

        fake_probability = torch.softmax(logits, dim=1)[:, 1]

        return {
            "cls": logits,
            "prob": fake_probability,
            "feat": weak_raw,
            "feat_norm": weak_norm,
            "cls_weak": weak_logits,
            "cls_strong": strong_logits,
            "feat_strong": strong_raw,
            "feat_norm_strong": strong_norm,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"].contiguous().view(-1)

        ensemble_logits = pred_dict["cls"]
        weak_logits = pred_dict["cls_weak"]
        strong_logits = pred_dict.get("cls_strong")

        zero = ensemble_logits.sum() * 0.0

        consistency_active = self.training and self.use_consistency

        if consistency_active:
            if strong_logits is None:
                raise RuntimeError(
                    "Consistency is enabled but strong-view logits were not computed."
                )

            loss_ce_weak = self.loss_ce(weak_logits, labels)
            loss_ce_strong = self.loss_ce(strong_logits, labels)
            loss_ce = 0.5 * (loss_ce_weak + loss_ce_strong)

            (
                loss_consistency,
                consistency_selected_fraction,
                teacher_confidence,
            ) = self.loss_consistency(weak_logits, strong_logits)

            loss_consistency = torch.nan_to_num(
                loss_consistency,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        else:
            loss_ce = self.loss_ce(ensemble_logits, labels)
            loss_ce_weak = loss_ce
            loss_ce_strong = zero
            loss_consistency = zero
            consistency_selected_fraction = zero.detach()
            teacher_confidence = zero.detach()

        alpha_consistency = ensemble_logits.new_tensor(
            self._current_alpha_consistency_value()
        )

        weighted_consistency = alpha_consistency * loss_consistency
        overall_loss = loss_ce + weighted_consistency

        loss_dict = {
            "overall": overall_loss,
            "loss_ce": loss_ce,
            "loss_ce_weak": loss_ce_weak,
            "loss_ce_strong": loss_ce_strong,
            "loss_consistency": loss_consistency,
            "alpha_consistency": alpha_consistency,
            "weighted_consistency": weighted_consistency,
            "consistency_selected_fraction": consistency_selected_fraction.detach(),
            "teacher_confidence": teacher_confidence.detach(),
        }

        with torch.no_grad():
            real_mask = labels.eq(0)
            fake_mask = labels.eq(1)

            loss_dict["real_loss"] = (
                self.loss_ce(ensemble_logits[real_mask], labels[real_mask])
                if real_mask.any()
                else zero.detach()
            )

            loss_dict["fake_loss"] = (
                self.loss_ce(ensemble_logits[fake_mask], labels[fake_mask])
                if fake_mask.any()
                else zero.detach()
            )

            loss_dict["prediction_agreement"] = (
                weak_logits.argmax(dim=1)
                .eq(strong_logits.argmax(dim=1))
                .float()
                .mean()
                if strong_logits is not None
                else ensemble_logits.new_tensor(1.0)
            )

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        from metrics.base_metrics_class import calculate_metrics_for_train

        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(),
            pred_dict["cls"].detach(),
        )

        return {
            "acc": acc,
            "auc": auc,
            "eer": eer,
            "ap": ap,
        }
