import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_detector import AbstractDetector
from detectors import DETECTOR


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="bias_only")
class BiasOnlyDetector(AbstractDetector):
    """CLIP ViT-L/14 with BitFit-style bias-only backbone tuning and CE."""

    def __init__(self, config=None, backbone: Optional[nn.Module] = None) -> None:
        super(BiasOnlyDetector, self).__init__()
        self.config = config or {}

        if bool(self.config.get("use_lora", False)):
            raise ValueError("BiasOnlyDetector does not support LoRA.")
        if not bool(self.config.get("train_backbone_bias", True)):
            raise ValueError("BiasOnlyDetector requires train_backbone_bias=true.")

        self.feature_dim = int(self.config.get("feature_dim", 1024))
        self.normalize_eps = float(self.config.get("normalize_eps", 1e-6))
        self.strict_trainable_check = bool(
            self.config.get("strict_trainable_check", True)
        )
        self.strict_clip_architecture = bool(
            self.config.get("strict_clip_architecture", False)
        )

        logger.info("Loading CLIP ViT-L/14 for BiasOnly CE baseline.")
        self.backbone = (
            backbone if backbone is not None else self.build_backbone(self.config)
        )
        self.head = nn.Linear(self.feature_dim, 2)

        self.build_loss(self.config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        self._setup_trainable_parameters()
        if self.strict_clip_architecture:
            self._validate_clip_architecture()

    def build_backbone(self, config):
        from transformers import CLIPModel

        model_name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        try:
            clip_model = CLIPModel.from_pretrained(model_name)
        except Exception:
            clip_model = CLIPModel.from_pretrained(
                model_name,
                local_files_only=True,
            )
        return clip_model.vision_model

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

    def _setup_trainable_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for name, parameter in self.backbone.named_parameters():
            if name.endswith(".bias"):
                parameter.requires_grad = True

        for parameter in self.head.parameters():
            parameter.requires_grad = True

        if self.strict_trainable_check:
            self._validate_trainable_parameters()

        self.trainable_param_summary = self._summarize_trainable_parameters()
        logger.info(
            "BiasOnly initialized. Trainable params: %s / %s (%.4f%%).",
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
                f"LoRA parameters are forbidden in BiasOnlyDetector: {forbidden[:20]}"
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

    def _validate_clip_architecture(self) -> None:
        clip_config = getattr(self.backbone, "config", None)
        if clip_config is None:
            raise RuntimeError("CLIP backbone does not expose a vision config.")

        expected = {
            "hidden_size": int(self.config.get("expected_clip_hidden_size", 1024)),
            "intermediate_size": int(
                self.config.get("expected_clip_intermediate_size", 4096)
            ),
            "num_hidden_layers": int(
                self.config.get("expected_clip_num_hidden_layers", 24)
            ),
            "num_attention_heads": int(
                self.config.get("expected_clip_num_attention_heads", 16)
            ),
            "image_size": int(self.config.get("expected_clip_image_size", 224)),
            "patch_size": int(self.config.get("expected_clip_patch_size", 14)),
        }
        mismatches = {
            name: (getattr(clip_config, name, None), expected_value)
            for name, expected_value in expected.items()
            if getattr(clip_config, name, None) != expected_value
        }
        if mismatches:
            raise RuntimeError(
                f"Unexpected CLIP vision architecture (actual, expected): {mismatches}"
            )

        expected_bias_params = int(
            self.config.get("expected_backbone_bias_params", 272384)
        )
        actual_bias_params = sum(
            parameter.numel()
            for name, parameter in self.backbone.named_parameters()
            if name.endswith(".bias")
        )
        if actual_bias_params != expected_bias_params:
            raise RuntimeError(
                "Unexpected number of CLIP backbone bias parameters: "
                f"actual={actual_bias_params:,}, expected={expected_bias_params:,}."
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

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        normalized_features = F.normalize(
            features,
            p=2,
            dim=1,
            eps=self.normalize_eps,
        )
        return self.head(normalized_features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features, normalized_features, logits = self._encode_images(
            data_dict["image"]
        )
        fake_probability = torch.softmax(logits, dim=1)[:, 1]
        return {
            "cls": logits,
            "prob": fake_probability,
            "feat": raw_features,
            "feat_norm": normalized_features,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"].contiguous().view(-1)
        logits = pred_dict["cls"]
        zero = logits.sum() * 0.0
        loss_ce = self.loss_ce(logits, labels)

        loss_dict = {
            "overall": loss_ce,
            "loss_ce": loss_ce,
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
        return {
            "acc": acc,
            "auc": auc,
            "eer": eer,
            "ap": ap,
        }
