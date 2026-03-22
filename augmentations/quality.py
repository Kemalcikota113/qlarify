"""Multi-dimensional episode quality scoring and filtering.

Produces a QualityReport per episode covering three axes:
  1. Kinematic Smoothness — jerk (2nd derivative) of action sequence
  2. Idle Detection       — fraction of frames with near-zero joint velocity
  3. Length Outlier       — episodes ±2σ from the dataset mean length

The overall_score combines smoothness and idle penalty and is used for
filtering. Length outlier status is printed but does not affect the score
(it is informational — very short/long episodes may still be valid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.dataset import load_all_actions_by_episode


@dataclass
class QualityReport:
    smoothness: float       # jerk-based [0,1], higher = smoother
    idle_ratio: float       # fraction of frames with near-zero velocity [0,1]
    is_length_outlier: bool # True if episode length is ±2σ from mean
    overall_score: float    # combined score used for filtering [0,1]
    episode_length: int     # number of frames in episode


def score_episode_multidim(
    actions: torch.Tensor,
    episode_length: int,
    mean_length: float,
    std_length: float,
    idle_velocity_threshold: float = 0.001,
    idle_penalty_above: float = 0.15,
) -> QualityReport:
    """Compute a multi-dimensional quality report for a single episode.

    Args:
        actions: Tensor of shape (T, D), float32.
        episode_length: Number of frames (== T).
        mean_length: Mean episode length across the dataset.
        std_length: Std dev of episode lengths (if 0, outlier check is skipped).
        idle_velocity_threshold: Per-frame mean joint speed below which a frame
            is considered idle (near-zero motion).
        idle_penalty_above: idle_ratio values above this trigger a score penalty.

    Returns:
        QualityReport with all metrics filled in.
    """
    a = actions.numpy()  # (T, D)

    # --- Kinematic smoothness (jerk = 2nd derivative) ---
    if a.shape[0] >= 3:
        jerk = np.diff(a, n=2, axis=0)
        smoothness = float(1.0 / (1.0 + np.mean(np.abs(jerk))))
    else:
        smoothness = 1.0

    # --- Idle detection (velocity = 1st derivative) ---
    if a.shape[0] >= 2:
        velocity = np.diff(a, n=1, axis=0)          # (T-1, D)
        per_frame_speed = np.mean(np.abs(velocity), axis=1)  # (T-1,)
        idle_ratio = float(np.mean(per_frame_speed < idle_velocity_threshold))
    else:
        idle_ratio = 0.0

    # --- Length outlier (±2σ from dataset mean) ---
    if std_length > 0:
        z = abs(episode_length - mean_length) / std_length
        is_length_outlier = z > 2.0
    else:
        is_length_outlier = False

    # --- Overall score: smoothness penalised by excess idle ratio ---
    idle_excess = max(0.0, idle_ratio - idle_penalty_above)
    overall_score = float(np.clip(smoothness * (1.0 - idle_excess * 2.0), 0.0, 1.0))

    return QualityReport(
        smoothness=smoothness,
        idle_ratio=idle_ratio,
        is_length_outlier=is_length_outlier,
        overall_score=overall_score,
        episode_length=episode_length,
    )


def score_and_filter_episodes(
    dataset: LeRobotDataset,
    ep_indices: list[int],
    threshold: float = 0.3,
    verbose: bool = True,
    report_only: bool = False,
) -> list[int]:
    """Score episodes and return indices whose overall_score >= threshold.

    Loads each parquet shard once (no video decoding) for fast scoring.

    Args:
        dataset: Source LeRobotDataset.
        ep_indices: Episode indices to evaluate.
        threshold: Minimum overall_score to keep [0, 1].
        verbose: Print per-episode report table.
        report_only: If True, print scores but keep ALL episodes (no filtering).

    Returns:
        Filtered list of episode indices.
    """
    if verbose:
        print("  Loading action data from parquet (no video decode)...")
    try:
        actions_by_ep = load_all_actions_by_episode(dataset)
    except Exception as e:
        if verbose:
            print(f"  Warning: could not load actions ({e}), keeping all episodes.")
        return list(ep_indices)

    # Compute episode lengths for outlier detection
    ep_lengths = []
    for ep_idx in ep_indices:
        ep_row = dataset.meta.episodes[ep_idx]
        ep_lengths.append(int(ep_row["dataset_to_index"]) - int(ep_row["dataset_from_index"]))
    mean_length = float(np.mean(ep_lengths)) if ep_lengths else 0.0
    std_length = float(np.std(ep_lengths)) if len(ep_lengths) > 1 else 0.0

    if verbose:
        header = (
            f"  {'Ep':>3}  {'smooth':>7}  {'idle':>6}  {'length':>7}  {'overall':>8}  decision"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

    kept = []
    for ep_idx, ep_length in zip(ep_indices, ep_lengths):
        ep_row = dataset.meta.episodes[ep_idx]
        ep_index = int(ep_row["episode_index"])
        actions = actions_by_ep.get(ep_index)

        if actions is not None:
            report = score_episode_multidim(
                actions, ep_length, mean_length, std_length
            )
        else:
            report = QualityReport(
                smoothness=1.0, idle_ratio=0.0,
                is_length_outlier=False, overall_score=1.0,
                episode_length=ep_length,
            )

        decision = "KEEP" if (report_only or report.overall_score >= threshold) else "DROP"
        outlier_tag = " [OUTLIER]" if report.is_length_outlier else ""

        if verbose:
            print(
                f"  {ep_idx:>3}  "
                f"{report.smoothness:>7.4f}  "
                f"{report.idle_ratio:>6.3f}  "
                f"{report.episode_length:>7}{outlier_tag:<10}  "
                f"{report.overall_score:>7.4f}  "
                f"{decision}"
            )

        if report_only or report.overall_score >= threshold:
            kept.append(ep_idx)

    if verbose:
        action = "reported" if report_only else f"kept {len(kept)}/{len(ep_indices)}"
        print(f"\n  Quality filter: {action} (threshold={threshold})")

    return kept
