import copy
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR

from .BiasArtifactConsistency import BiasArtifactConsistencyDetector


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="bias_ema_teacher_consistency")
class BiasEMATeacherConsistencyDetector(BiasArtifactConsistencyDetector):
    """Bias-only CLIP with an EMA weak-view teacher for strong-view consistency."""

    def __init__(self, config=None, backbone: nn.Module = None) -> None:
        super().__init__(config=config, backbone=backbone)

        self.ema_decay = float(self.config.get("ema_decay", 0.999))
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
        )
        self._build_ema_teacher()

    def _build_ema_teacher(self) -> None:
        self.ema_backbone = copy.deepcopy(self.backbone)
        self.ema_head = copy.deepcopy(self.head)
        for parameter in self.ema_backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.ema_head.parameters():
            parameter.requires_grad = False
        self.ema_backbone.eval()
        self.ema_head.eval()

        self.trainable_param_summary = self._summarize_trainable_parameters()
        logger.info(
            "EMA teacher initialized for BiasEMATeacherConsistency with decay=%s.",
            self.ema_decay,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "ema_backbone"):
            self.ema_backbone.eval()
            self.ema_head.eval()
        return self

    @torch.no_grad()
    def update_ema_teacher(self) -> None:
        self._update_ema_module(self.ema_backbone, self.backbone)
        self._update_ema_module(self.ema_head, self.head)
        self.ema_update_count.add_(1)

    @torch.no_grad()
    def sync_ema_teacher(self) -> None:
        self.ema_backbone.load_state_dict(self.backbone.state_dict())
        self.ema_head.load_state_dict(self.head.state_dict())
        self.ema_update_count.zero_()

    def _update_ema_module(self, ema_module: nn.Module, student_module: nn.Module) -> None:
        decay = self.ema_decay
        for ema_parameter, student_parameter in zip(
            ema_module.parameters(),
            student_module.parameters(),
        ):
            ema_parameter.data.mul_(decay).add_(
                student_parameter.data,
                alpha=1.0 - decay,
            )

        for ema_buffer, student_buffer in zip(
            ema_module.buffers(),
            student_module.buffers(),
        ):
            ema_buffer.copy_(student_buffer)

    @torch.no_grad()
    def _encode_images_with_ema_teacher(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.ema_backbone(images)
        if not hasattr(outputs, "pooler_output"):
            raise RuntimeError("The EMA vision teacher must return pooler_output.")
        raw_features = outputs.pooler_output
        if raw_features.dim() != 2:
            raise ValueError("EMA teacher pooler_output must have shape [B, D].")
        if raw_features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected EMA teacher feature dimension {self.feature_dim}, "
                f"got {raw_features.size(1)}."
            )

        normalized_features = F.normalize(
            raw_features,
            p=2,
            dim=1,
            eps=self.normalize_eps,
        )
        logits = self.ema_head(normalized_features)
        return raw_features, normalized_features, logits

    def forward(self, data_dict: dict, inference=False) -> dict:
        weak_raw, weak_norm, weak_logits = self._encode_images(data_dict["image"])

        strong_logits = None
        strong_raw = None
        strong_norm = None
        teacher_logits = None
        teacher_raw = None
        teacher_norm = None
        consistency_active = self.training and self.use_consistency and not inference
        if consistency_active:
            if data_dict.get("image_strong") is None:
                raise RuntimeError(
                    "EMA teacher consistency training requires "
                    "data_dict['image_strong']. Enable use_consistency_views."
                )
            strong_raw, strong_norm, strong_logits = self._encode_images(
                data_dict["image_strong"]
            )
            teacher_raw, teacher_norm, teacher_logits = (
                self._encode_images_with_ema_teacher(data_dict["image"])
            )
            logits = (
                self.weak_ce_weight * weak_logits
                + self.strong_ce_weight * strong_logits
            )
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
            "cls_teacher": teacher_logits,
            "feat_strong": strong_raw,
            "feat_norm_strong": strong_norm,
            "feat_teacher": teacher_raw,
            "feat_norm_teacher": teacher_norm,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"].contiguous().view(-1)
        ensemble_logits = pred_dict["cls"]
        weak_logits = pred_dict["cls_weak"]
        strong_logits = pred_dict.get("cls_strong")
        teacher_logits = pred_dict.get("cls_teacher")
        zero = ensemble_logits.sum() * 0.0

        diagnostic_names = (
            "consistency_confident_fraction",
            "consistency_selected_fraction",
            "consistency_selected_real_fraction",
            "consistency_selected_fake_fraction",
            "teacher_confidence",
            "teacher_accuracy",
        )
        diagnostics = {name: zero.detach() for name in diagnostic_names}

        consistency_active = self.training and self.use_consistency
        if consistency_active:
            if strong_logits is None or teacher_logits is None:
                raise RuntimeError(
                    "EMA consistency is enabled but strong-view or teacher logits "
                    "were not computed."
                )
            loss_ce_weak = self.loss_ce(weak_logits, labels)
            loss_ce_strong = self.loss_ce(strong_logits, labels)
            loss_ce = (
                self.weak_ce_weight * loss_ce_weak
                + self.strong_ce_weight * loss_ce_strong
            )
            loss_consistency, diagnostics = self.loss_consistency(
                teacher_logits,
                strong_logits,
                labels,
            )
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
            "weak_ce_weight": ensemble_logits.new_tensor(self.weak_ce_weight),
            "strong_ce_weight": ensemble_logits.new_tensor(self.strong_ce_weight),
            "loss_consistency": loss_consistency,
            "alpha_consistency": alpha_consistency,
            "weighted_consistency": weighted_consistency,
            "ema_decay": ensemble_logits.new_tensor(self.ema_decay),
            "ema_update_count": self.ema_update_count.detach().clone(),
        }
        loss_dict.update(
            {name: value.detach() for name, value in diagnostics.items()}
        )

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
                teacher_logits.argmax(dim=1)
                .eq(strong_logits.argmax(dim=1))
                .float()
                .mean()
                if teacher_logits is not None and strong_logits is not None
                else ensemble_logits.new_tensor(1.0)
            )
            loss_dict["student_view_agreement"] = (
                weak_logits.argmax(dim=1)
                .eq(strong_logits.argmax(dim=1))
                .float()
                .mean()
                if strong_logits is not None
                else ensemble_logits.new_tensor(1.0)
            )

        return loss_dict
