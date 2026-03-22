"""Episode quality scoring and filtering based on action smoothness."""

from __future__ import annotations

import numpy as np
import torch


def score_episode(frames: list[dict]) -> float:
    """Score an episode's quality by action smoothness (lower jerk = higher quality).

    Uses the second derivative (jerk) of the action sequence. Smooth, purposeful
    robot motion produces low jerk. Trembling or failed grasps produce high jerk.

    Args:
        frames: List of frame dicts from dataset[i], each containing "action" tensor.

    Returns:
        Float in [0, 1], higher = smoother/better quality.
        Returns 1.0 if episode is too short to compute jerk (< 3 frames).
    """
    if len(frames) < 3:
        return 1.0

    actions = np.stack([f["action"].numpy() for f in frames])  # (T, D)
    jerk = np.diff(actions, n=2, axis=0)                       # 2nd derivative: (T-2, D)
    mean_abs_jerk = float(np.mean(np.abs(jerk)))
    smoothness = 1.0 / (1.0 + mean_abs_jerk)
    return smoothness


def filter_episodes(
    episode_frames: list[tuple[int, list[dict]]],
    threshold: float = 0.3,
    verbose: bool = True,
) -> list[tuple[int, list[dict]]]:
    """Filter out low-quality episodes below a smoothness threshold.

    Args:
        episode_frames: List of (episode_idx, frames) tuples.
        threshold: Minimum smoothness score to keep [0, 1].
        verbose: Print per-episode scores.

    Returns:
        Filtered list of (episode_idx, frames) tuples.
    """
    scored = []
    for ep_idx, frames in episode_frames:
        score = score_episode(frames)
        scored.append((score, ep_idx, frames))
        if verbose:
            status = "KEEP" if score >= threshold else "DROP"
            print(f"  Episode {ep_idx:3d}: smoothness={score:.4f} [{status}]")

    kept = [(ep_idx, frames) for score, ep_idx, frames in scored if score >= threshold]

    if verbose:
        n_total = len(scored)
        n_kept = len(kept)
        print(f"\n  Quality filter: kept {n_kept}/{n_total} episodes (threshold={threshold})")

    return kept
