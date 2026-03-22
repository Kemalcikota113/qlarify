"""LeRobot v3 dataset helpers for loading, iterating, and creating datasets."""

from __future__ import annotations

from typing import Generator

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def iter_episodes(
    dataset: LeRobotDataset,
    max_episodes: int | None = None,
) -> Generator[tuple[int, list[dict]], None, None]:
    """Yield (episode_idx, frames) for each episode using metadata boundaries.

    Args:
        dataset: Loaded LeRobotDataset.
        max_episodes: Cap number of episodes (None = all).

    Yields:
        (episode_idx, list of frame dicts from dataset[i])
    """
    total = dataset.meta.total_episodes
    if max_episodes is not None:
        total = min(total, max_episodes)

    for ep_idx in range(total):
        ep_row = dataset.meta.episodes.iloc[ep_idx]
        start = int(ep_row["dataset_from_index"])
        end = int(ep_row["dataset_to_index"])
        frames = [dataset[i] for i in range(start, end)]
        yield ep_idx, frames


def create_output_dataset(
    source: LeRobotDataset,
    repo_id: str,
) -> LeRobotDataset:
    """Create a new empty dataset mirroring the source's schema.

    Args:
        source: Source dataset to copy schema from.
        repo_id: HuggingFace repo ID for the new dataset (user/name).

    Returns:
        New empty LeRobotDataset ready for add_frame() calls.
    """
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=source.meta.fps,
        features=source.meta.features,
        robot_type=getattr(source.meta, "robot_type", None),
        use_videos=True,
        video_backend="torchvision",
    )
