import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA adapter around an existing nn.Linear layer."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0.")

        self.base = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        self.reset_parameters()

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_output = F.linear(lora_hidden, self.lora_B) * self.scaling
        return base_output + lora_output


class ProjectionHead(nn.Module):
    """Projection head for contrastive embeddings: 1024 -> 512 -> 128."""

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AsymmetricSupConLoss(nn.Module):
    """
    Asymmetric supervised contrastive loss for deepfake detection.

    Real-real pairs are pulled strongly, fake-fake pairs are pulled weakly, and
    real-fake pairs are explicitly repelled with a cosine-margin term.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        real_label: int = 0,
        real_weight: float = 1.0,
        fake_weight: float = 0.25,
        cross_weight: float = 1.0,
        cross_margin: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if base_temperature <= 0:
            raise ValueError("base_temperature must be > 0.")

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.real_label = real_label
        self.real_weight = real_weight
        self.fake_weight = fake_weight
        self.cross_weight = cross_weight
        self.cross_margin = cross_margin
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        return_parts: bool = False,
    ):
        if features.dim() != 2:
            raise ValueError("AsymmetricSupConLoss expects features with shape [B, D].")

        features = F.normalize(features, p=2, dim=1, eps=self.eps)
        labels = labels.contiguous().view(-1)
        device = features.device
        batch_size = features.size(0)

        zero = features.new_zeros(())
        if batch_size < 2:
            parts = {"loss_asy_pull": zero, "loss_asy_cross": zero}
            return (zero, parts) if return_parts else zero

        labels_col = labels.view(-1, 1)
        same_class = labels_col.eq(labels_col.T)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        logits_mask = ~eye_mask

        cosine_sim = torch.matmul(features, features.T)
        logits = cosine_sim / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        positive_mask = same_class & logits_mask
        positive_count = positive_mask.float().sum(dim=1)
        valid_positive_anchor = positive_count > 0

        if valid_positive_anchor.any():
            mean_log_prob_pos = (
                positive_mask.float() * log_prob
            ).sum(dim=1) / positive_count.clamp_min(1.0)

            is_real_anchor = labels.eq(self.real_label)
            anchor_weight = torch.where(
                is_real_anchor,
                torch.full_like(mean_log_prob_pos, self.real_weight),
                torch.full_like(mean_log_prob_pos, self.fake_weight),
            )
            pull_loss_per_anchor = -(
                self.temperature / self.base_temperature
            ) * mean_log_prob_pos

            pull_loss = (
                pull_loss_per_anchor[valid_positive_anchor]
                * anchor_weight[valid_positive_anchor]
            ).sum() / valid_positive_anchor.float().sum().clamp_min(1.0)
        else:
            pull_loss = zero

        cross_mask = labels_col.ne(labels_col.T) & logits_mask
        if cross_mask.any():
            cross_loss = (
                F.relu(cosine_sim - self.cross_margin) * cross_mask.float()
            ).sum() / cross_mask.float().sum().clamp_min(1.0)
        else:
            cross_loss = zero

        loss = pull_loss + self.cross_weight * cross_loss
        parts = {"loss_asy_pull": pull_loss, "loss_asy_cross": cross_loss}
        return (loss, parts) if return_parts else loss


@DETECTOR.register_module(module_name="bias_lora_asy")
class BiasLoraAsyDetector(AbstractDetector):
    def __init__(self, config=None):
        super(BiasLoraAsyDetector, self).__init__()
        self.config = config or {}

        logger.info("Loading CLIP ViT-L/14 for BiasLoraAsy LoRA-Bias + AsySupCon.")

        self.feature_dim = int(self.config.get("feature_dim", 1024))
        self.projection_hidden_dim = int(self.config.get("projection_hidden_dim", 512))
        self.projection_dim = int(self.config.get("projection_dim", 128))
        self.normalize_eps = float(self.config.get("normalize_eps", 1e-6))
        feature_dropout = float(self.config.get("feature_dropout", 0.0))
        projection_dropout = float(self.config.get("projection_dropout", feature_dropout))

        self.use_lora = bool(self.config.get("use_lora", True))
        self.train_backbone_bias = bool(self.config.get("train_backbone_bias", True))
        self.strict_trainable_check = bool(self.config.get("strict_trainable_check", True))
        self.use_asy_supcon = bool(self.config.get("use_asy_supcon", True))

        self.backbone = self.build_backbone(self.config)
        self.feature_dropout = nn.Dropout(p=feature_dropout) if feature_dropout > 0 else nn.Identity()
        self.projection_dropout = (
            nn.Dropout(p=projection_dropout) if projection_dropout > 0 else nn.Identity()
        )
        self.head = nn.Linear(self.feature_dim, 2)
        self.projection_head = ProjectionHead(
            in_dim=self.feature_dim,
            hidden_dim=self.projection_hidden_dim,
            out_dim=self.projection_dim,
        )

        self.build_loss(self.config)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        self.lora_module_names: List[str] = []
        self._setup_trainable_params()

    def build_backbone(self, config):
        model_name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        try:
            clip_model = CLIPModel.from_pretrained(model_name)
        except Exception:
            clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        return clip_model.vision_model

    def build_loss(self, config):
        weight_real = float(config.get("weight_real", 1.0))
        weight_fake = float(config.get("weight_fake", 2.0))
        class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32)
        label_smoothing = float(config.get("label_smoothing", 0.1))

        self.loss_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        self.alpha_asy = float(config.get("alpha_asy", config.get("lambda_supcon", 0.05)))
        self.alpha_asy_warmup_epochs = int(config.get("alpha_asy_warmup_epochs", 0))
        self.alpha_asy_decay_start_epoch = int(config.get("alpha_asy_decay_start_epoch", -1))
        self.alpha_asy_decay_end_epoch = int(
            config.get(
                "alpha_asy_decay_end_epoch",
                max(self.alpha_asy_decay_start_epoch + 1, int(config.get("nEpochs", 0))),
            )
        )
        self.alpha_asy_min = float(config.get("alpha_asy_min", self.alpha_asy))
        self.loss_asy_supcon = AsymmetricSupConLoss(
            temperature=float(config.get("temperature", 0.07)),
            base_temperature=float(config.get("base_temperature", 0.07)),
            real_label=int(config.get("real_label", 0)),
            real_weight=float(config.get("asy_real_weight", 1.0)),
            fake_weight=float(config.get("asy_fake_weight", 0.25)),
            cross_weight=float(config.get("asy_cross_weight", 1.0)),
            cross_margin=float(config.get("asy_cross_margin", 0.0)),
        )

    def _setup_trainable_params(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.use_lora:
            self.lora_module_names = self._inject_lora_to_last_blocks()

        if self.train_backbone_bias:
            for name, param in self.backbone.named_parameters():
                if name.endswith(".bias"):
                    param.requires_grad = True

        for name, param in self.backbone.named_parameters():
            if self._is_lora_parameter(name):
                param.requires_grad = True

        for param in self.head.parameters():
            param.requires_grad = True
        asy_supcon_active = self.use_asy_supcon and self.alpha_asy > 0
        for param in self.projection_head.parameters():
            param.requires_grad = asy_supcon_active

        if self.strict_trainable_check:
            self._validate_trainable_backbone()

        self.trainable_param_summary = self._summarize_trainable_parameters()
        logger.info(
            "BiasLoraAsy initialized. Trainable params: "
            f"{self.trainable_param_summary['trainable']:,} / "
            f"{self.trainable_param_summary['total']:,} "
            f"({self.trainable_param_summary['percent']:.4f}%)."
        )
        logger.info("LoRA modules: %s", self.lora_module_names)
        if not asy_supcon_active:
            logger.info("AsySupCon disabled; projection head is frozen and skipped.")

    def _inject_lora_to_last_blocks(self) -> List[str]:
        layers = getattr(getattr(self.backbone, "encoder", None), "layers", None)
        if layers is None:
            raise AttributeError("CLIP vision backbone does not expose encoder.layers.")

        total_layers = len(layers)
        last_k = int(self.config.get("lora_last_k_blocks", 4))
        if last_k <= 0:
            return []

        start_idx = max(0, total_layers - last_k)
        targets = self._get_lora_targets()
        rank = int(self.config.get("lora_rank", 8))
        alpha = float(self.config.get("lora_alpha", 16.0))
        dropout = float(self.config.get("lora_dropout", 0.05))

        injected_modules = []
        for layer_idx in range(start_idx, total_layers):
            self_attn = getattr(layers[layer_idx], "self_attn", None)
            if self_attn is None:
                raise AttributeError(f"CLIP layer {layer_idx} does not expose self_attn.")

            for target_name in targets:
                target_module = getattr(self_attn, target_name, None)
                if target_module is None:
                    raise AttributeError(
                        f"CLIP layer {layer_idx}.self_attn has no module named {target_name}."
                    )
                if isinstance(target_module, LoRALinear):
                    injected_modules.append(f"encoder.layers.{layer_idx}.self_attn.{target_name}")
                    continue
                if not isinstance(target_module, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear for {target_name}, got {type(target_module)}."
                    )

                setattr(
                    self_attn,
                    target_name,
                    LoRALinear(
                        base_layer=target_module,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                injected_modules.append(f"encoder.layers.{layer_idx}.self_attn.{target_name}")

        expected_modules = {
            f"encoder.layers.{idx}.self_attn.{target}"
            for idx in range(start_idx, total_layers)
            for target in targets
        }
        if set(injected_modules) != expected_modules:
            raise RuntimeError(
                "LoRA injection mismatch. "
                f"Expected {sorted(expected_modules)}, got {sorted(injected_modules)}."
            )
        return injected_modules

    def _get_lora_targets(self) -> List[str]:
        targets = self.config.get("lora_targets", ["q_proj", "v_proj"])
        if isinstance(targets, str):
            targets = [item.strip() for item in targets.split(",") if item.strip()]
        if not targets:
            raise ValueError("lora_targets must contain at least one target module.")
        invalid_targets = set(targets) - {"q_proj", "v_proj"}
        if invalid_targets:
            raise ValueError(
                "BiasLoraAsy LoRA is intentionally restricted to q_proj/v_proj. "
                f"Invalid lora_targets: {sorted(invalid_targets)}."
            )
        return list(targets)

    @staticmethod
    def _is_lora_parameter(name: str) -> bool:
        return name.endswith(".lora_A") or name.endswith(".lora_B")

    def _validate_trainable_backbone(self) -> None:
        unexpected_trainable = []
        for name, param in self.backbone.named_parameters():
            allowed_lora = self._is_lora_parameter(name)
            allowed_bias = self.train_backbone_bias and name.endswith(".bias")
            if param.requires_grad and not (allowed_lora or allowed_bias):
                unexpected_trainable.append(name)

        if unexpected_trainable:
            raise RuntimeError(
                "Unexpected trainable backbone parameters found. "
                "This would violate LoRA-Bias adaptation: "
                f"{unexpected_trainable[:20]}"
            )

        if self.use_lora:
            target_set = set(self._get_lora_targets())
            for module_name in self.lora_module_names:
                target_name = module_name.rsplit(".", 1)[-1]
                if target_name not in target_set:
                    raise RuntimeError(f"Unexpected LoRA target module: {module_name}")

    def _summarize_trainable_parameters(self) -> Dict[str, float]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "percent": 100.0 * trainable / max(total, 1),
        }

    def get_trainable_summary(self) -> Tuple[Dict[str, float], List[str]]:
        return self.trainable_param_summary, self.lora_module_names

    def _current_alpha_asy_value(self) -> float:
        if not self.use_asy_supcon or self.alpha_asy <= 0:
            return 0.0

        alpha = self.alpha_asy
        if self.training and self.alpha_asy_warmup_epochs > 0:
            epoch = int(getattr(self, "epoch", 0))
            alpha *= min(max(epoch / self.alpha_asy_warmup_epochs, 0.0), 1.0)
        if self.training and self.alpha_asy_decay_start_epoch >= 0:
            epoch = int(getattr(self, "epoch", 0))
            if epoch >= self.alpha_asy_decay_start_epoch:
                decay_span = max(
                    self.alpha_asy_decay_end_epoch - self.alpha_asy_decay_start_epoch,
                    1,
                )
                progress = min(
                    max((epoch - self.alpha_asy_decay_start_epoch) / decay_span, 0.0),
                    1.0,
                )
                alpha = (1.0 - progress) * alpha + progress * self.alpha_asy_min
        return alpha

    def _current_alpha_asy(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._current_alpha_asy_value(), dtype=torch.float32, device=device)

    def features(self, data_dict: dict) -> torch.Tensor:
        outputs = self.backbone(data_dict["image"])
        return outputs.pooler_output

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, data_dict: dict, inference=False) -> dict:
        raw_features = self.features(data_dict)
        norm_features = F.normalize(raw_features, p=2, dim=1, eps=self.normalize_eps)

        pred = self.classifier(self.feature_dropout(norm_features))
        prob = torch.softmax(pred, dim=1)[:, 1]

        proj_features = None
        if self.training and self.use_asy_supcon and self._current_alpha_asy_value() > 0:
            proj_features = self.projection_head(self.projection_dropout(norm_features))
            proj_features = F.normalize(proj_features, p=2, dim=1, eps=self.normalize_eps)

        return {
            "cls": pred,
            "prob": prob,
            "feat": raw_features,
            "feat_norm": norm_features,
            "feat_proj": proj_features,
        }

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]

        loss_ce = self.loss_ce(pred, label)
        alpha_asy_value = self._current_alpha_asy_value()
        alpha_asy = torch.tensor(alpha_asy_value, dtype=torch.float32, device=pred.device)

        zero = pred.new_zeros(())
        asy_parts = {"loss_asy_pull": zero, "loss_asy_cross": zero}
        if self.training and self.use_asy_supcon and alpha_asy_value > 0:
            feat_proj = pred_dict.get("feat_proj", None)
            if feat_proj is None:
                raise RuntimeError("AsySupCon is enabled but feat_proj was not computed.")
            loss_asy_supcon, asy_parts = self.loss_asy_supcon(
                feat_proj,
                label,
                return_parts=True,
            )
            loss_asy_supcon = torch.nan_to_num(loss_asy_supcon, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            loss_asy_supcon = zero

        overall_loss = loss_ce + alpha_asy * loss_asy_supcon

        loss_dict = {
            "overall": overall_loss,
            "loss_ce": loss_ce,
            "loss_asy_supcon": loss_asy_supcon,
            "alpha_asy": alpha_asy,
            "loss_asy_pull": asy_parts["loss_asy_pull"],
            "loss_asy_cross": asy_parts["loss_asy_cross"],
        }

        with torch.no_grad():
            mask_real = label == 0
            mask_fake = label == 1

            loss_dict["real_loss"] = (
                self.loss_ce(pred[mask_real], label[mask_real])
                if mask_real.sum() > 0
                else zero
            )
            loss_dict["fake_loss"] = (
                self.loss_ce(pred[mask_fake], label[mask_fake])
                if mask_fake.sum() > 0
                else zero
            )

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]

        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(),
            pred.detach(),
        )

        return {
            "acc": acc,
            "auc": auc,
            "eer": eer,
            "ap": ap,
        }
