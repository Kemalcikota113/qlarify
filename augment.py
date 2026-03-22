#!/usr/bin/env python3
"""LeRobot v3 Dataset Augmentation Tool.

Downloads a LeRobot v3 dataset from HuggingFace Hub, applies augmentations
(color/lighting, background replacement, state noise), optionally filters
low-quality episodes, and uploads the result as a new dataset.

Usage:
    python augment.py \\
        --source lerobot/aloha_static_cups_open \\
        --output YOUR_HF_USER/aloha_augmented \\
        --strategies color,background,noise \\
        --multiplier 3 \\
        --filter-quality \\
        --quality-threshold 0.3 \\
        --max-episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from augmentations.quality import filter_episodes, score_episode
from augmentations.state import augment_action, augment_state
from augmentations.video import augment_image, build_color_transform
from utils.dataset import create_output_dataset, iter_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a LeRobot v3 dataset and upload to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source dataset repo ID (e.g. lerobot/aloha_static_cups_open)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output repo ID (e.g. your-hf-user/aloha_augmented)",
    )
    parser.add_argument(
        "--strategies",
        default="color",
        help=(
            "Comma-separated augmentation strategies: "
            "color (ColorJitter+noise), background (rembg replacement), "
            "noise (state/action Gaussian noise). Default: color"
        ),
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="How many augmented copies to create per episode. Default: 2",
    )
    parser.add_argument(
        "--filter-quality",
        action="store_true",
        help="Filter out low-quality episodes before augmenting.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.3,
        help="Minimum smoothness score to keep an episode [0, 1]. Default: 0.3",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (useful for quick tests).",
    )
    parser.add_argument(
        "--state-noise-scale",
        type=float,
        default=0.01,
        help="Std dev of Gaussian noise on state observations. Default: 0.01",
    )
    parser.add_argument(
        "--action-noise-scale",
        type=float,
        default=0.005,
        help="Std dev of Gaussian noise on actions. Default: 0.005",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload dataset as private on HuggingFace Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but skip upload. Useful for testing.",
    )
    return parser.parse_args()


def get_image_keys(dataset: LeRobotDataset) -> list[str]:
    """Return all image/video feature keys from dataset schema."""
    return [
        key
        for key, feat in dataset.meta.features.items()
        if feat.get("dtype") in ("image", "video")
    ]


def get_state_keys(dataset: LeRobotDataset) -> list[str]:
    """Return non-image, non-metadata feature keys (state, action, etc.)."""
    skip = {"timestamp", "frame_index", "episode_index", "index", "task_index", "task"}
    return [
        key
        for key, feat in dataset.meta.features.items()
        if feat.get("dtype") not in ("image", "video") and key not in skip
    ]


def augment_frame(
    frame: dict,
    strategies: set[str],
    image_keys: list[str],
    state_keys: list[str],
    seed: int,
    color_transform,
    state_noise_scale: float,
    action_noise_scale: float,
) -> dict:
    """Apply all requested augmentations to a single frame dict.

    Returns a new frame dict ready to pass to dataset.add_frame().
    """
    result = {}

    # Augment image features
    for key in image_keys:
        if key in frame and isinstance(frame[key], torch.Tensor):
            result[key] = augment_image(
                frame[key],
                strategies=strategies,
                seed=seed,
                color_transform=color_transform,
            )
        else:
            result[key] = frame[key]

    # Augment state/action features
    for key in state_keys:
        if key not in frame or not isinstance(frame[key], torch.Tensor):
            result[key] = frame.get(key)
            continue
        if "noise" in strategies:
            if "action" in key:
                result[key] = augment_action(frame[key], noise_scale=action_noise_scale, seed=seed)
            else:
                result[key] = augment_state(frame[key], noise_scale=state_noise_scale, seed=seed)
        else:
            result[key] = frame[key]

    # Pass through metadata (task string is required by add_frame)
    for key in ("task", "timestamp"):
        if key in frame:
            result[key] = frame[key]

    return result


def run(args: argparse.Namespace) -> None:
    strategies = {s.strip().lower() for s in args.strategies.split(",")}
    valid_strategies = {"color", "background", "noise"}
    unknown = strategies - valid_strategies
    if unknown:
        print(f"Unknown strategies: {unknown}. Valid: {valid_strategies}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  LeRobot Dataset Augmentation Tool")
    print(f"{'='*60}")
    print(f"  Source:      {args.source}")
    print(f"  Output:      {args.output}")
    print(f"  Strategies:  {', '.join(sorted(strategies))}")
    print(f"  Multiplier:  {args.multiplier}x")
    print(f"  Max episodes: {args.max_episodes or 'all'}")
    print(f"  Filter quality: {args.filter_quality} (threshold={args.quality_threshold})")
    print(f"  Dry run:     {args.dry_run}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 1. Load source dataset
    # -----------------------------------------------------------------------
    print(f"[1/5] Loading source dataset: {args.source}")
    source = LeRobotDataset(args.source)
    print(f"      {source.meta.total_episodes} episodes, {source.meta.total_frames} frames, {source.meta.fps} fps")

    image_keys = get_image_keys(source)
    state_keys = get_state_keys(source)
    print(f"      Image keys: {image_keys}")
    print(f"      State keys: {state_keys}")

    # -----------------------------------------------------------------------
    # 2. Load all episodes into memory (or stream, for large datasets)
    # -----------------------------------------------------------------------
    print(f"\n[2/5] Loading episodes (max={args.max_episodes or 'all'})...")
    all_episodes = list(
        tqdm(
            iter_episodes(source, max_episodes=args.max_episodes),
            total=min(source.meta.total_episodes, args.max_episodes or source.meta.total_episodes),
            desc="  Loading",
            unit="ep",
        )
    )
    print(f"      Loaded {len(all_episodes)} episodes.")

    # -----------------------------------------------------------------------
    # 3. Quality filtering
    # -----------------------------------------------------------------------
    if args.filter_quality:
        print(f"\n[3/5] Scoring episode quality (action smoothness)...")
        episodes_to_augment = filter_episodes(
            all_episodes,
            threshold=args.quality_threshold,
            verbose=True,
        )
    else:
        print(f"\n[3/5] Skipping quality filter (use --filter-quality to enable).")
        episodes_to_augment = all_episodes

    if not episodes_to_augment:
        print("ERROR: No episodes passed quality filter. Lower --quality-threshold.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 4. Create output dataset and apply augmentations
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Creating output dataset: {args.output}")
    out_dataset = create_output_dataset(source, args.output)

    color_transform = build_color_transform() if "color" in strategies else None

    total_episodes_out = len(episodes_to_augment) * args.multiplier
    print(f"      Will create {total_episodes_out} augmented episodes "
          f"({len(episodes_to_augment)} × {args.multiplier})\n")

    ep_counter = 0
    for variant_idx in range(args.multiplier):
        print(f"  Variant {variant_idx + 1}/{args.multiplier}:")
        for orig_ep_idx, frames in tqdm(episodes_to_augment, desc="    Episodes", unit="ep"):
            base_seed = variant_idx * 100_000 + orig_ep_idx * 1_000

            for frame_offset, frame in enumerate(frames):
                seed = base_seed + frame_offset
                aug_frame = augment_frame(
                    frame=frame,
                    strategies=strategies,
                    image_keys=image_keys,
                    state_keys=state_keys,
                    seed=seed,
                    color_transform=color_transform,
                    state_noise_scale=args.state_noise_scale,
                    action_noise_scale=args.action_noise_scale,
                )
                out_dataset.add_frame(aug_frame)

            out_dataset.save_episode()
            ep_counter += 1

    print(f"\n  Done. Created {ep_counter} episodes.")

    # -----------------------------------------------------------------------
    # 5. Finalize and upload
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Finalizing dataset...")
    out_dataset.finalize()
    print("      finalize() complete.")

    if args.dry_run:
        print("\n  DRY RUN — skipping upload.")
        print(f"  Dataset saved locally at: {out_dataset.root}")
        return

    print(f"  Uploading to HuggingFace Hub as '{args.output}' ...")
    out_dataset.push_to_hub(
        push_videos=True,
        private=args.private,
        license="apache-2.0",
    )

    # Print visualizer link
    hf_user = args.output.split("/")[0]
    dataset_name = args.output.split("/")[1]
    visualizer_url = (
        f"https://huggingface.co/spaces/lerobot/visualize_dataset"
        f"?path=%2F{hf_user}%2F{dataset_name}%2Fepisode_0"
    )
    print(f"\n{'='*60}")
    print(f"  Upload complete!")
    print(f"  Dataset: https://huggingface.co/datasets/{args.output}")
    print(f"\n  Visualizer link:")
    print(f"  {visualizer_url}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)
