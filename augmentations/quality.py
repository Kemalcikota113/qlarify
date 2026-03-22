"""Episode quality scoring and filtering based on action smoothness."""

from __future__ import annotations

import numpy as np
import torch

from utils.dataset import load_all_actions_by_episode


def score_episode_actions(actions: torch.Tensor) -> float:
    """Score an episode's quality by action smoothness (lower jerk = higher quality).

    Uses the second derivative (jerk) of the action sequence. Smooth,
    purposeful robot motion produces low jerk. Trembling or failed grasps
    produce high jerk.

    Args:
        actions: Tensor of shape (T, D), float32.

    Returns:
        Float in [0, 1], higher = smoother/better quality.
        Returns 1.0 if episode is too short to compute jerk (< 3 frames).
    """
    if actions.shape[0] < 3:
        return 1.0

    a = actions.numpy()
    jerk = np.diff(a, n=2, axis=0)  # (T-2, D)
    mean_abs_jerk = float(np.mean(np.abs(jerk)))
    return 1.0 / (1.0 + mean_abs_jerk)


def score_and_filter_episodes(
    dataset,
    ep_indices: list[int],
    threshold: float = 0.3,
    verbose: bool = True,
) -> list[int]:
    """Score episodes from parquet action data and return indices above threshold.

    Loads each parquet shard once (not once per episode), then groups by
    episode. No video decoding — fast even for large datasets.

    Args:
        dataset: Source LeRobotDataset.
        ep_indices: Episode indices to score.
        threshold: Minimum smoothness score to keep [0, 1].
        verbose: Print per-episode scores.

    Returns:
        List of episode indices that passed the quality filter.
    """
    if verbose:
        print("  Loading action data from parquet (no video decode)...")
    try:
        actions_by_ep = load_all_actions_by_episode(dataset)
    except Exception as e:
        if verbose:
            print(f"  Warning: could not load actions ({e}), keeping all episodes.")
        return list(ep_indices)

    kept = []
    for ep_idx in ep_indices:
        ep_row = dataset.meta.episodes[ep_idx]
        ep_index = int(ep_row["episode_index"])
        actions = actions_by_ep.get(ep_index)
        score = score_episode_actions(actions) if actions is not None else 1.0
        status = "KEEP" if score >= threshold else "DROP"
        if verbose:
            print(f"  Episode {ep_idx:3d}: smoothness={score:.4f} [{status}]")
        if score >= threshold:
            kept.append(ep_idx)

    if verbose:
        print(f"\n  Quality filter: kept {len(kept)}/{len(ep_indices)} episodes "
              f"(threshold={threshold})")

    return kept
