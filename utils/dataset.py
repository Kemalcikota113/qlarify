"""LeRobot v3 dataset helpers for loading, iterating, and creating datasets."""

from __future__ import annotations

import shutil
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_episode_bounds(dataset: LeRobotDataset, ep_idx: int) -> tuple[int, int]:
    """Return (start_frame_idx, end_frame_idx) for an episode."""
    ep_row = dataset.meta.episodes[ep_idx]
    return int(ep_row["dataset_from_index"]), int(ep_row["dataset_to_index"])


def load_all_actions_by_episode(dataset: LeRobotDataset) -> dict[int, torch.Tensor]:
    """Load action tensors for all episodes from parquet, grouped by episode index.

    Reads each unique parquet shard once (not once per episode), making
    quality scoring fast even for large datasets.

    Returns:
        Dict mapping episode_index → Tensor of shape (T, action_dim).
    """
    import datasets as hf_datasets
    import collections

    # Collect unique parquet file paths
    seen_paths = set()
    all_rows: dict[int, list] = collections.defaultdict(list)

    for ep_idx in range(dataset.meta.total_episodes):
        ep_row = dataset.meta.episodes[ep_idx]
        ep_index = int(ep_row["episode_index"])
        data_path = dataset.root / dataset.meta.get_data_file_path(ep_index)
        seen_paths.add((ep_index, str(data_path)))

    loaded_files: dict[str, object] = {}
    for ep_index, path_str in seen_paths:
        if path_str not in loaded_files:
            ds = hf_datasets.load_dataset("parquet", data_files=path_str, split="train")
            loaded_files[path_str] = ds

    # Group actions by episode_index
    result: dict[int, torch.Tensor] = {}
    for path_str, ds in loaded_files.items():
        for row in ds:
            ep_idx = int(row["episode_index"])
            if ep_idx not in all_rows:
                all_rows[ep_idx] = []
            all_rows[ep_idx].append(row["action"])

    for ep_idx, action_list in all_rows.items():
        result[ep_idx] = torch.tensor(action_list, dtype=torch.float32)

    return result


def create_output_dataset(
    source: LeRobotDataset,
    repo_id: str,
    overwrite: bool = False,
) -> LeRobotDataset:
    """Create a new empty dataset mirroring the source's schema.

    Args:
        source: Source dataset to mirror schema from.
        repo_id: Output HuggingFace repo ID (e.g. ``user/dataset-name``).
        overwrite: If True, delete any existing local cache for ``repo_id``
            before creating. If False and the cache exists, raises a clear
            error rather than letting LeRobotDataset raise a cryptic one.
    """
    from huggingface_hub import constants as hf_constants
    import pathlib

    local_dir = pathlib.Path(hf_constants.HF_HOME) / "lerobot" / repo_id.replace("/", "--")
    if local_dir.exists():
        if overwrite:
            shutil.rmtree(local_dir)
        else:
            raise FileExistsError(
                f"Local cache already exists for '{repo_id}' at {local_dir}.\n"
                "Re-run with --overwrite to delete it and start fresh, "
                "or choose a different --output name."
            )

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=source.meta.fps,
        features=source.meta.features,
        robot_type=getattr(source.meta, "robot_type", None),
        use_videos=True,
        video_backend="torchvision",
    )
