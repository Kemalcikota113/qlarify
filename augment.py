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

import torch
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from augmentations.quality import score_and_filter_episodes
from augmentations.state import augment_action, augment_state
from augmentations.video import augment_image, build_color_transform
from utils.dataset import create_output_dataset, get_episode_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a LeRobot v3 dataset and upload to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", required=True,
        help="Source dataset repo ID (e.g. lerobot/aloha_static_cups_open)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output repo ID (e.g. your-hf-user/aloha_augmented)",
    )
    parser.add_argument(
        "--strategies", default="color",
        help=(
            "Comma-separated augmentation strategies: "
            "color (ColorJitter+noise on video frames), "
            "background (rembg background replacement), "
            "noise (Gaussian noise on state/action). Default: color"
        ),
    )
    parser.add_argument(
        "--multiplier", type=int, default=2,
        help="Augmented copies to create per episode. Default: 2",
    )
    parser.add_argument(
        "--filter-quality", action="store_true",
        help="Filter low-quality episodes by action smoothness before augmenting.",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.3,
        help="Min smoothness score to keep [0,1]. Default: 0.3",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Cap number of source episodes processed (useful for quick tests).",
    )
    parser.add_argument(
        "--state-noise-scale", type=float, default=0.01,
        help="Std dev of Gaussian noise on state observations. Default: 0.01",
    )
    parser.add_argument(
        "--action-noise-scale", type=float, default=0.005,
        help="Std dev of Gaussian noise on actions. Default: 0.005",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Upload dataset as private on HuggingFace Hub.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process data but skip upload. Useful for local testing.",
    )
    return parser.parse_args()


def get_image_keys(dataset: LeRobotDataset) -> list[str]:
    return [k for k, f in dataset.meta.features.items() if f.get("dtype") in ("image", "video")]


def get_data_keys(dataset: LeRobotDataset) -> list[str]:
    """Non-image feature keys that need to be passed to add_frame() explicitly."""
    auto_generated = {"frame_index", "episode_index", "index", "task_index", "task", "timestamp"}
    return [
        k for k, f in dataset.meta.features.items()
        if f.get("dtype") not in ("image", "video") and k not in auto_generated
    ]


def augment_frame(
    frame: dict,
    strategies: set[str],
    image_keys: list[str],
    data_keys: list[str],
    seed: int,
    color_transform,
    state_noise_scale: float,
    action_noise_scale: float,
    source_features: dict,
) -> dict:
    """Return a new augmented frame dict ready for dataset.add_frame()."""
    result = {}

    # Video augmentation — augment_image works in CHW, add_frame expects HWC
    for key in image_keys:
        t = frame.get(key)
        if isinstance(t, torch.Tensor):
            # t is float32 CHW [0,1] from dataset[i]
            aug = augment_image(t, strategies=strategies, seed=seed,
                                color_transform=color_transform)
            # Convert back to HWC uint8 as required by add_frame
            hwc = (aug.permute(1, 2, 0) * 255).clamp(0, 255).byte()
            result[key] = hwc
        else:
            result[key] = t

    # State / action augmentation (only float tensors)
    for key in data_keys:
        t = frame.get(key)
        if not isinstance(t, torch.Tensor):
            result[key] = t
            continue
        # Ensure shape matches feature schema (e.g. next.done loaded as () but schema wants (1,))
        expected_shape = tuple(source_features.get(key, {}).get("shape", t.shape))
        t_shaped = t.reshape(expected_shape) if t.shape != expected_shape else t

        if "noise" in strategies and t_shaped.is_floating_point():
            if "action" in key:
                result[key] = augment_action(t_shaped, noise_scale=action_noise_scale, seed=seed)
            else:
                result[key] = augment_state(t_shaped, noise_scale=state_noise_scale, seed=seed)
        else:
            result[key] = t_shaped

    # task string is required; timestamp is auto-generated — do not pass it
    if "task" in frame:
        result["task"] = frame["task"]

    return result


def run(args: argparse.Namespace) -> None:
    strategies = {s.strip().lower() for s in args.strategies.split(",")}
    valid = {"color", "background", "noise"}
    if unknown := strategies - valid:
        print(f"Unknown strategies: {unknown}. Valid: {valid}", file=sys.stderr)
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

    # -------------------------------------------------------------------------
    # 1. Load source dataset
    # -------------------------------------------------------------------------
    print(f"[1/5] Loading source dataset: {args.source}")
    source = LeRobotDataset(args.source)
    image_keys = get_image_keys(source)
    data_keys = get_data_keys(source)
    n_eps = min(source.meta.total_episodes, args.max_episodes or source.meta.total_episodes)
    ep_indices = list(range(n_eps))
    print(f"      {source.meta.total_episodes} episodes total, processing {n_eps}")
    print(f"      FPS: {source.meta.fps} | Images: {image_keys} | Data: {data_keys}")

    # -------------------------------------------------------------------------
    # 2. Quality filtering (reads parquet only — no video decode)
    # -------------------------------------------------------------------------
    if args.filter_quality:
        print(f"\n[2/5] Scoring episode quality (parquet action data, no video decode)...")
        ep_indices = score_and_filter_episodes(
            source, ep_indices,
            threshold=args.quality_threshold,
            verbose=True,
        )
        if not ep_indices:
            print("ERROR: No episodes passed quality filter. Lower --quality-threshold.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print(f"\n[2/5] Skipping quality filter (pass --filter-quality to enable).")

    # -------------------------------------------------------------------------
    # 3. Create output dataset
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Creating output dataset schema: {args.output}")
    out_dataset = create_output_dataset(source, args.output)
    color_transform = build_color_transform() if "color" in strategies else None

    total_out = len(ep_indices) * args.multiplier
    print(f"      Will produce {total_out} episodes ({len(ep_indices)} source × {args.multiplier}x)\n")

    # -------------------------------------------------------------------------
    # 4. Augment: stream frame-by-frame (constant memory regardless of dataset size)
    # -------------------------------------------------------------------------
    print(f"[4/5] Augmenting episodes...")
    ep_out_count = 0

    for variant_idx in range(args.multiplier):
        desc = f"  Variant {variant_idx + 1}/{args.multiplier}"
        for ep_idx in tqdm(ep_indices, desc=desc, unit="ep"):
            start, end = get_episode_bounds(source, ep_idx)
            base_seed = variant_idx * 100_000 + ep_idx * 1_000

            for frame_offset, global_idx in enumerate(range(start, end)):
                frame = source[global_idx]
                aug = augment_frame(
                    frame=frame,
                    strategies=strategies,
                    image_keys=image_keys,
                    data_keys=data_keys,
                    seed=base_seed + frame_offset,
                    color_transform=color_transform,
                    state_noise_scale=args.state_noise_scale,
                    action_noise_scale=args.action_noise_scale,
                    source_features=source.meta.features,
                )
                out_dataset.add_frame(aug)

            out_dataset.save_episode()
            ep_out_count += 1

    print(f"\n  Done. Created {ep_out_count} augmented episodes.")

    # -------------------------------------------------------------------------
    # 5. Finalize and push
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Finalizing...")
    out_dataset.finalize()
    print("      finalize() complete.")

    if args.dry_run:
        print(f"\n  DRY RUN — skipping upload.")
        print(f"  Dataset saved locally at: {out_dataset.root}")
        return

    print(f"  Uploading '{args.output}' to HuggingFace Hub...")
    out_dataset.push_to_hub(push_videos=True, private=args.private, license="apache-2.0")

    hf_user, dataset_name = args.output.split("/", 1)
    vis_url = (
        f"https://huggingface.co/spaces/lerobot/visualize_dataset"
        f"?path=%2F{hf_user}%2F{dataset_name}%2Fepisode_0"
    )
    print(f"\n{'='*60}")
    print(f"  Upload complete!")
    print(f"  Dataset:    https://huggingface.co/datasets/{args.output}")
    print(f"  Visualizer: {vis_url}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run(parse_args())
