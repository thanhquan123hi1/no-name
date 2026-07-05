import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR

from .BiasEMATeacherConsistency import BiasEMATeacherConsistencyDetector


logger = logging.getLogger(__name__)


class CorrectnessAwareJSDLoss(nn.Module):
    """JSD consistency restricted to confident, label-correct EMA teachers."""

    def __init__(
        self,
        temperature: float = 1.0,
        confidence_threshold: float = 0.8,
        require_teacher_correct: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("consistency_temperature must be > 0.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("consistency_confidence_threshold must be in [0, 1].")

        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.require_teacher_correct = bool(require_teacher_correct)

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
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                "Teacher and student logits must have the same shape, got "
                f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}."
            )
        if teacher_logits.dim() != 2:
            raise ValueError("Consistency logits must have shape [B, C].")

        labels = labels.contiguous().view(-1)
        if labels.numel() != teacher_logits.size(0):
            raise ValueError("Labels must match the consistency batch dimension.")

        temperature = self.temperature
        teacher_prob = F.softmax(teacher_logits / temperature, dim=1).detach()
        student_prob = F.softmax(student_logits / temperature, dim=1)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        mixture_prob = 0.5 * (teacher_prob + student_prob)
        teacher_log_prob = torch.log(teacher_prob.clamp_min(1e-12))
        log_mixture = torch.log(mixture_prob.clamp_min(1e-12))

        teacher_to_mix = (teacher_prob * (teacher_log_prob - log_mixture)).sum(
            dim=1
        )
        student_to_mix = (student_prob * (student_log_prob - log_mixture)).sum(
            dim=1
        )
        per_sample_jsd = 0.5 * (teacher_to_mix + student_to_mix) * (
            temperature**2
        )

        teacher_confidence, teacher_prediction = teacher_prob.max(dim=1)
        confident = teacher_confidence.ge(self.confidence_threshold)
        teacher_correct = teacher_prediction.eq(labels)
        selected = confident
        if self.require_teacher_correct:
            selected = selected & teacher_correct

        if selected.any():
            loss = per_sample_jsd[selected].mean()
        else:
            # Preserve a differentiable zero for the student branch.
            loss = student_logits.sum() * 0.0

        dtype = student_logits.dtype
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


@DETECTOR.register_module(module_name="bias_ema_jsd_consistency")
class BiasEMAJSDConsistencyDetector(BiasEMATeacherConsistencyDetector):
    """Bias-only CLIP with EMA-teacher Jensen-Shannon consistency."""

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

        self.loss_consistency = CorrectnessAwareJSDLoss(
            temperature=float(config.get("consistency_temperature", 1.0)),
            confidence_threshold=float(
                config.get("consistency_confidence_threshold", 0.8)
            ),
            require_teacher_correct=bool(
                config.get("consistency_require_teacher_correct", True)
            ),
        )
