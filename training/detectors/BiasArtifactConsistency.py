import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR

from .BiasConsistency import BiasConsistencyDetector


logger = logging.getLogger(__name__)


class CorrectnessAwareKLLoss(nn.Module):
    """Weak-to-strong KL restricted to confident, label-correct teachers."""

    def __init__(
        self,
        temperature: float = 1.0,
        confidence_threshold: float = 0.8,
        require_teacher_correct: bool = True,
        class_balanced: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("consistency_temperature must be > 0.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("consistency_confidence_threshold must be in [0, 1].")

        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.require_teacher_correct = bool(require_teacher_correct)
        self.class_balanced = bool(class_balanced)

    @staticmethod
    def _masked_fraction(
        selected: torch.Tensor,
        population: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if population.any():
            return selected[population].to(dtype).mean()
        return selected.new_zeros((), dtype=dtype)

    def forward(
        self,
        weak_logits: torch.Tensor,
        strong_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if weak_logits.shape != strong_logits.shape:
            raise ValueError(
                "Weak and strong logits must have the same shape, got "
                f"{tuple(weak_logits.shape)} and {tuple(strong_logits.shape)}."
            )
        if weak_logits.dim() != 2:
            raise ValueError("Consistency logits must have shape [B, C].")

        labels = labels.contiguous().view(-1)
        if labels.numel() != weak_logits.size(0):
            raise ValueError("Labels must match the consistency batch dimension.")

        temperature = self.temperature
        teacher_prob = F.softmax(weak_logits / temperature, dim=1).detach()
        student_log_prob = F.log_softmax(strong_logits / temperature, dim=1)
        per_sample_kl = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="none",
        ).sum(dim=1) * (temperature**2)

        teacher_confidence, teacher_prediction = teacher_prob.max(dim=1)
        confident = teacher_confidence.ge(self.confidence_threshold)
        teacher_correct = teacher_prediction.eq(labels)
        selected = confident
        if self.require_teacher_correct:
            selected = selected & teacher_correct

        if selected.any():
            if self.class_balanced:
                class_losses = []
                for class_index in labels.unique(sorted=True):
                    class_selected = selected & labels.eq(class_index)
                    if class_selected.any():
                        class_losses.append(per_sample_kl[class_selected].mean())
                loss = torch.stack(class_losses).mean()
            else:
                loss = per_sample_kl[selected].mean()
        else:
            # Preserve a differentiable zero for the strong student branch.
            loss = strong_logits.sum() * 0.0

        dtype = strong_logits.dtype
        real_population = labels.eq(0)
        fake_population = labels.eq(1)
        diagnostics = {
            "consistency_confident_fraction": confident.to(dtype).mean(),
            "consistency_selected_fraction": selected.to(dtype).mean(),
            "consistency_selected_real_fraction": self._masked_fraction(
                selected, real_population, dtype
            ),
            "consistency_selected_fake_fraction": self._masked_fraction(
                selected, fake_population, dtype
            ),
            "teacher_confidence": teacher_confidence.mean(),
            "teacher_accuracy": teacher_correct.to(dtype).mean(),
        }
        return loss, diagnostics


@DETECTOR.register_module(module_name="bias_artifact_consistency")
class BiasArtifactConsistencyDetector(BiasConsistencyDetector):
    """Bias-only CLIP with artifact-preserving supervised consistency."""

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
            label_smoothing=float(config.get("label_smoothing", 0.05)),
        )

        weak_weight = float(config.get("weak_ce_weight", 0.75))
        strong_weight = float(config.get("strong_ce_weight", 0.25))
        if weak_weight < 0 or strong_weight < 0:
            raise ValueError("weak_ce_weight and strong_ce_weight must be >= 0.")
        weight_sum = weak_weight + strong_weight
        if weight_sum <= 0:
            raise ValueError("At least one CE view weight must be positive.")
        self.weak_ce_weight = weak_weight / weight_sum
        self.strong_ce_weight = strong_weight / weight_sum

        self.alpha_consistency_max = float(
            config.get("alpha_consistency_max", 0.2)
        )
        self.alpha_consistency_start_epoch = int(
            config.get("alpha_consistency_start_epoch", 2)
        )
        self.alpha_consistency_warmup_epochs = int(
            config.get("alpha_consistency_warmup_epochs", 3)
        )
        if self.alpha_consistency_max < 0:
            raise ValueError("alpha_consistency_max must be >= 0.")
        if self.alpha_consistency_warmup_epochs < 0:
            raise ValueError("alpha_consistency_warmup_epochs must be >= 0.")

        self.loss_consistency = CorrectnessAwareKLLoss(
            temperature=float(config.get("consistency_temperature", 1.0)),
            confidence_threshold=float(
                config.get("consistency_confidence_threshold", 0.8)
            ),
            require_teacher_correct=bool(
                config.get("consistency_require_teacher_correct", True)
            ),
            class_balanced=bool(
                config.get("consistency_class_balanced", True)
            ),
        )

    def forward(self, data_dict: dict, inference=False) -> dict:
        weak_raw, weak_norm, weak_logits = self._encode_images(data_dict["image"])

        strong_logits = None
        strong_raw = None
        strong_norm = None
        consistency_active = self.training and self.use_consistency and not inference
        if consistency_active:
            if data_dict.get("image_strong") is None:
                raise RuntimeError(
                    "Artifact-aware consistency training requires "
                    "data_dict['image_strong']. Enable use_consistency_views."
                )
            strong_raw, strong_norm, strong_logits = self._encode_images(
                data_dict["image_strong"]
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
            "feat_strong": strong_raw,
            "feat_norm_strong": strong_norm,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"].contiguous().view(-1)
        ensemble_logits = pred_dict["cls"]
        weak_logits = pred_dict["cls_weak"]
        strong_logits = pred_dict.get("cls_strong")
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
            if strong_logits is None:
                raise RuntimeError(
                    "Consistency is enabled but strong-view logits were not computed."
                )
            loss_ce_weak = self.loss_ce(weak_logits, labels)
            loss_ce_strong = self.loss_ce(strong_logits, labels)
            loss_ce = (
                self.weak_ce_weight * loss_ce_weak
                + self.strong_ce_weight * loss_ce_strong
            )
            loss_consistency, diagnostics = self.loss_consistency(
                weak_logits,
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
                weak_logits.argmax(dim=1)
                .eq(strong_logits.argmax(dim=1))
                .float()
                .mean()
                if strong_logits is not None
                else ensemble_logits.new_tensor(1.0)
            )

        return loss_dict

