import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_detector import AbstractDetector
from detectors import DETECTOR


logger = logging.getLogger(__name__)


class RealAnchorSupConLoss(nn.Module):
    """Supervised contrastive loss in which only Real samples are anchors."""

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        real_label: int = 0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if base_temperature <= 0:
            raise ValueError("base_temperature must be > 0.")

        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.real_label = int(real_label)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("RealAnchorSupConLoss expects features with shape [B, D].")

        labels = labels.contiguous().view(-1)
        if labels.numel() != features.size(0):
            raise ValueError(
                "The number of labels must match the feature batch dimension."
            )

        # Normalize here even though the detector also exposes normalized features.
        # This keeps the loss safe when it is used independently.
        features = F.normalize(features, p=2, dim=1, eps=self.eps)
        real_mask = labels.eq(self.real_label)

        # Keep the zero connected to features so backward() remains valid.
        if int(real_mask.sum().item()) < 2:
            return features.sum() * 0.0

        batch_size = features.size(0)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        candidate_mask = ~self_mask

        logits = torch.matmul(features, features.T) / self.temperature
        denominator_logits = logits.masked_fill(~candidate_mask, float("-inf"))
        log_denominator = torch.logsumexp(denominator_logits, dim=1)
        log_prob = logits - log_denominator.unsqueeze(1)

        # Every valid anchor is Real, and its positives are the other Real samples.
        # Fake samples occur only in the denominator above.
        positive_mask = real_mask.unsqueeze(0).expand(batch_size, -1) & candidate_mask
        positive_count = positive_mask.sum(dim=1)
        valid_real_anchor = real_mask & positive_count.gt(0)

        positive_log_prob = log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        mean_log_prob_positive = positive_log_prob / positive_count.clamp_min(1)
        loss_per_anchor = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_positive

        # Average over valid Real anchors only; Fake rows never enter this mean.
        return loss_per_anchor[valid_real_anchor].mean()


@DETECTOR.register_module(module_name="bias_real_anchor")
class BiasRealAnchorDetector(AbstractDetector):
    """CLIP ViT-L/14 with bias-only backbone tuning and Real-anchor SupCon."""

    def __init__(self, config=None, backbone: Optional[nn.Module] = None) -> None:
        super(BiasRealAnchorDetector, self).__init__()
        self.config = config or {}

        if bool(self.config.get("use_lora", False)):
            raise ValueError("BiasRealAnchorDetector does not support LoRA.")

        self.feature_dim = int(self.config.get("feature_dim", 1024))
        self.normalize_eps = float(self.config.get("normalize_eps", 1e-6))
        self.strict_trainable_check = bool(
            self.config.get("strict_trainable_check", True)
        )
        self.epoch = int(self.config.get("start_epoch", 0))

        logger.info("Loading CLIP ViT-L/14 for bias-only Real-anchor SupCon.")
        self.backbone = backbone if backbone is not None else self.build_backbone(self.config)
        self.head = nn.Linear(self.feature_dim, 2)

        self.build_loss(self.config)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        self._setup_trainable_parameters()

    def build_backbone(self, config):
        # Import lazily so unit tests can inject a tiny backbone without loading CLIP.
        from transformers import CLIPModel

        model_name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        try:
            clip_model = CLIPModel.from_pretrained(model_name)
        except Exception:
            clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        return clip_model.vision_model

    def build_loss(self, config) -> None:
        class_weights = torch.tensor(
            [
                float(config.get("weight_real", 1.0)),
                float(config.get("weight_fake", 2.0)),
            ],
            dtype=torch.float32,
        )
        self.loss_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(config.get("label_smoothing", 0.1)),
        )

        self.alpha_real_max = float(config.get("alpha_real_max", 0.05))
        self.alpha_real_start_epoch = int(config.get("alpha_real_start_epoch", 0))
        self.alpha_real_warmup_epochs = int(
            config.get("alpha_real_warmup_epochs", 5)
        )
        if self.alpha_real_max < 0:
            raise ValueError("alpha_real_max must be >= 0.")
        if self.alpha_real_warmup_epochs < 0:
            raise ValueError("alpha_real_warmup_epochs must be >= 0.")

        self.loss_real_anchor_supcon = RealAnchorSupConLoss(
            temperature=float(config.get("temperature", 0.07)),
            base_temperature=float(config.get("base_temperature", 0.07)),
            real_label=0,
            eps=self.normalize_eps,
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
            "BiasRealAnchor initialized. Trainable params: %s / %s (%.4f%%).",
            f"{self.trainable_param_summary['trainable']:,}",
            f"{self.trainable_param_summary['total']:,}",
            self.trainable_param_summary["percent"],
        )

    def _validate_trainable_parameters(self) -> None:
        lora_parameters = [
            name
            for name, _ in self.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        lora_modules = [
            name
            for name, module in self.named_modules()
            if hasattr(module, "lora_A") or hasattr(module, "lora_B")
        ]
        if lora_parameters or lora_modules:
            raise RuntimeError(
                "LoRA parameters/modules are forbidden in BiasRealAnchorDetector: "
                f"{(lora_parameters + lora_modules)[:20]}"
            )

        projection_modules = [
            name
            for name, module in self.named_modules()
            if module.__class__.__name__ == "ProjectionHead"
        ]
        if hasattr(self, "projection_head") or projection_modules:
            raise RuntimeError("ProjectionHead is forbidden in BiasRealAnchorDetector.")

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
            name for name, parameter in self.head.named_parameters() if not parameter.requires_grad
        ]
        if frozen_head:
            raise RuntimeError(
                f"Classifier parameters must all be trainable: {frozen_head}"
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

    def _current_alpha_real_value(self) -> float:
        epoch = float(getattr(self, "epoch", 0))
        if self.alpha_real_warmup_epochs == 0:
            progress = 1.0 if epoch >= self.alpha_real_start_epoch else 0.0
        else:
            progress = (
                epoch - self.alpha_real_start_epoch
            ) / self.alpha_real_warmup_epochs
            progress = min(max(progress, 0.0), 1.0)
        return self.alpha_real_max * progress

    def features(self, data_dict: dict) -> torch.Tensor:
        outputs = self.backbone(data_dict["image"])
        if not hasattr(outputs, "pooler_output"):
            raise RuntimeError("The vision backbone must return pooler_output.")
        features = outputs.pooler_output
        if features.dim() != 2:
            raise ValueError("Backbone pooler_output must have shape [B, D].")
        if features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected pooler feature dimension {self.feature_dim}, "
                f"got {features.size(1)}."
            )
        return features

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self.features(data_dict)
        normalized_features = F.normalize(
            raw_features,
            p=2,
            dim=1,
            eps=self.normalize_eps,
        )
        logits = self.classifier(normalized_features)
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
        normalized_features = pred_dict["feat_norm"]

        loss_ce = self.loss_ce(logits, labels)
        loss_real_anchor_supcon = self.loss_real_anchor_supcon(
            normalized_features,
            labels,
        )
        loss_real_anchor_supcon = torch.nan_to_num(
            loss_real_anchor_supcon,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        alpha_real = logits.new_tensor(self._current_alpha_real_value())
        weighted_real_anchor_supcon = alpha_real * loss_real_anchor_supcon
        overall_loss = loss_ce + weighted_real_anchor_supcon
        zero = logits.new_zeros(())

        loss_dict = {
            "overall": overall_loss,
            "loss_ce": loss_ce,
            "loss_real_anchor_supcon": loss_real_anchor_supcon,
            "alpha_real": alpha_real,
            "weighted_real_anchor_supcon": weighted_real_anchor_supcon,
        }

        with torch.no_grad():
            real_mask = labels.eq(0)
            fake_mask = labels.eq(1)
            loss_dict["real_loss"] = (
                self.loss_ce(logits[real_mask], labels[real_mask])
                if real_mask.any()
                else zero
            )
            loss_dict["fake_loss"] = (
                self.loss_ce(logits[fake_mask], labels[fake_mask])
                if fake_mask.any()
                else zero
            )

            diagnostic_features = F.normalize(
                normalized_features.detach(),
                p=2,
                dim=1,
                eps=self.normalize_eps,
            )
            cosine = torch.matmul(diagnostic_features, diagnostic_features.T)
            upper_triangle = torch.triu(
                torch.ones_like(cosine, dtype=torch.bool),
                diagonal=1,
            )
            real_real_pairs = (
                real_mask.unsqueeze(1) & real_mask.unsqueeze(0) & upper_triangle
            )
            real_fake_pairs = real_mask.unsqueeze(1) & fake_mask.unsqueeze(0)
            fake_fake_pairs = (
                fake_mask.unsqueeze(1) & fake_mask.unsqueeze(0) & upper_triangle
            )

            loss_dict["num_real"] = logits.new_tensor(int(real_mask.sum().item()))
            loss_dict["num_fake"] = logits.new_tensor(int(fake_mask.sum().item()))
            if real_real_pairs.any():
                loss_dict["mean_real_real_cosine"] = cosine[real_real_pairs].mean()
            if real_fake_pairs.any():
                loss_dict["mean_real_fake_cosine"] = cosine[real_fake_pairs].mean()
            if fake_fake_pairs.any():
                loss_dict["mean_fake_fake_cosine"] = cosine[fake_fake_pairs].mean()

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
