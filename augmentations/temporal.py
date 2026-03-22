"""Temporal Domain Randomization for robot learning datasets.

Resamples episode frame sequences at randomised speed factors (0.75×–1.25×),
making the learned policy robust to variations in motor latency and control
frequency — a key challenge when deploying VLA models on real hardware.

Parquet/video sync note:
    `resample_frame_indices()` returns *global* frame indices into the source
    dataset. Each call to `source[global_idx]` retrieves both the decoded
    video frame and the corresponding parquet row (state, action, timestamp)
    as a single dict. Skipping or repeating an index therefore affects both
    modalities simultaneously — no separate alignment step is needed.
"""

from __future__ import annotations

import numpy as np


def resample_frame_indices(
    start: int,
    end: int,
    speed_factor: float,
) -> list[int]:
    """Return resampled global frame indices for temporal augmentation.

    Args:
        start: First global frame index of the episode (inclusive).
        end:   One-past-the-last global frame index (exclusive).
        speed_factor: >1.0 → faster (fewer frames); <1.0 → slower (more frames,
            via frame repetition); =1.0 → no change.

    Returns:
        List of global frame indices to use for the augmented episode.
        The state/action parquet data for each index is automatically
        co-selected when `source[idx]` is called in the main pipeline.
    """
    original_len = end - start
    new_len = max(3, int(round(original_len / speed_factor)))
    offsets = np.linspace(0, original_len - 1, new_len).round().astype(int)
    return [start + int(o) for o in offsets]


def sample_speed_factor(
    seed: int,
    speed_min: float = 0.75,
    speed_max: float = 1.25,
) -> float:
    """Sample a reproducible random speed factor for one episode variant.

    Args:
        seed: Deterministic seed (use base_seed per variant+episode).
        speed_min: Lower bound (0.75 = 25% slower than original).
        speed_max: Upper bound (1.25 = 25% faster than original).

    Returns:
        Float speed factor in [speed_min, speed_max].
    """
    rng = np.random.default_rng(seed)
    return float(rng.uniform(speed_min, speed_max))
