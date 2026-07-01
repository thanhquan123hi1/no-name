import sys
from pathlib import Path

import pytest


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset.samplers import (  # noqa: E402
    build_class_balanced_sampler,
    make_class_balanced_sample_weights,
)


def test_sample_weights_give_equal_total_mass_to_real_and_fake():
    labels = [0, 0, 1, 1, 1, 1, 1, 1]

    weights, counts = make_class_balanced_sample_weights(labels)

    real_mass = weights[:2].sum().item()
    fake_mass = weights[2:].sum().item()
    assert counts == {0: 2, 1: 6}
    assert real_mass == pytest.approx(fake_mass)


def test_balanced_sampler_is_seed_reproducible():
    labels = [0, 0, 1, 1, 1, 1, 1, 1]

    first = list(build_class_balanced_sampler(labels, seed=7749))
    second = list(build_class_balanced_sampler(labels, seed=7749))

    assert first == second
    assert len(first) == len(labels)


def test_balanced_sampler_rejects_single_class_data():
    with pytest.raises(ValueError, match="at least one real and one fake"):
        make_class_balanced_sample_weights([1, 1, 1])

