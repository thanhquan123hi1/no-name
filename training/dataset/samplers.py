from collections import Counter
from typing import Iterable, Tuple

import torch
from torch.utils.data import WeightedRandomSampler


def make_class_balanced_sample_weights(
    labels: Iterable[int],
) -> Tuple[torch.Tensor, Counter]:
    """Return inverse-frequency binary-class weights for a sampler."""

    binary_labels = [0 if int(label) == 0 else 1 for label in labels]
    if not binary_labels:
        raise ValueError("Cannot build a balanced sampler for an empty dataset.")

    class_counts = Counter(binary_labels)
    if len(class_counts) < 2:
        raise ValueError(
            "Class-balanced sampling requires at least one real and one fake sample."
        )

    weights = torch.tensor(
        [1.0 / class_counts[label] for label in binary_labels],
        dtype=torch.double,
    )
    return weights, class_counts


def build_class_balanced_sampler(
    labels: Iterable[int],
    seed: int,
) -> WeightedRandomSampler:
    """Sample real and fake classes with equal expected probability."""

    weights, _ = make_class_balanced_sample_weights(labels)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )

