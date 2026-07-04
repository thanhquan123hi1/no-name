import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR

from .BiasConsistency import BiasConsistencyDetector


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="bias_multilayer_consistency")
class BiasMultiLayerConsistencyDetector(BiasConsistencyDetector):
    """Bias-only CLIP with weak/strong consistency and multi-layer CLS fusion."""

    def __init__(self, config=None, backbone: Optional[nn.Module] = None) -> None:
        super().__init__(config=config, backbone=backbone)

        self.fusion_layers = self._parse_fusion_layers(
            self.config.get("fusion_layers", [12, 18, 24])
        )
        self.fusion_feature_dim = self.feature_dim * len(self.fusion_layers)
        self.head = nn.Linear(self.fusion_feature_dim, 2)

        self._setup_trainable_parameters()
        logger.info(
            "BiasMultiLayerConsistency uses CLIP hidden layers %s "
            "with fused feature dimension %s.",
            self.fusion_layers,
            self.fusion_feature_dim,
        )

    @staticmethod
    def _parse_fusion_layers(layers: Sequence[int]) -> Tuple[int, ...]:
        parsed_layers = tuple(int(layer) for layer in layers)
        if not parsed_layers:
            raise ValueError("fusion_layers must contain at least one layer index.")
        if len(set(parsed_layers)) != len(parsed_layers):
            raise ValueError("fusion_layers must not contain duplicate indices.")
        return parsed_layers

    def _encode_images(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.backbone(images, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "The vision backbone must return hidden_states when "
                "output_hidden_states=True."
            )

        cls_features = []
        num_hidden_states = len(hidden_states)
        for layer_index in self.fusion_layers:
            if layer_index < 0 or layer_index >= num_hidden_states:
                raise ValueError(
                    f"fusion layer index {layer_index} is out of range for "
                    f"{num_hidden_states} hidden-state tensors."
                )

            layer_hidden = hidden_states[layer_index]
            if layer_hidden.dim() != 3:
                raise ValueError(
                    "Each hidden state must have shape [B, tokens, D], got "
                    f"{tuple(layer_hidden.shape)}."
                )
            if layer_hidden.size(-1) != self.feature_dim:
                raise ValueError(
                    f"Expected hidden feature dimension {self.feature_dim}, "
                    f"got {layer_hidden.size(-1)}."
                )
            cls_features.append(layer_hidden[:, 0, :])

        raw_features = torch.cat(cls_features, dim=1)
        normalized_features = F.normalize(
            raw_features,
            p=2,
            dim=1,
            eps=self.normalize_eps,
        )
        logits = self.head(normalized_features)
        return raw_features, normalized_features, logits
