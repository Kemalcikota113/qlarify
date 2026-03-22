"""Numerical augmentation for state observations and actions."""

from __future__ import annotations

import torch


def augment_state(
    tensor: torch.Tensor,
    noise_scale: float = 0.01,
    seed: int | None = None,
) -> torch.Tensor:
    """Add small Gaussian noise to state observations.

    Simulates sensor noise and slight positional variance across episodes,
    improving model robustness to real-world measurement uncertainty.

    Args:
        tensor: State tensor (any shape), float32.
        noise_scale: Std dev of Gaussian noise as fraction of typical range.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Augmented tensor with same shape and dtype.
    """
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(tensor) * noise_scale
    return tensor + noise


def augment_action(
    tensor: torch.Tensor,
    noise_scale: float = 0.005,
    seed: int | None = None,
) -> torch.Tensor:
    """Add tiny Gaussian noise to action sequences.

    Very small noise to preserve task validity while increasing variation.
    Action noise should be significantly smaller than state noise.

    Args:
        tensor: Action tensor (any shape), float32.
        noise_scale: Std dev of Gaussian noise. Keep small (< 0.01).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Augmented tensor with same shape and dtype.
    """
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(tensor) * noise_scale
    return tensor + noise
