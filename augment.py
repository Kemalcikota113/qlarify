#!/usr/bin/env python3
"""LeRobot v3 Dataset Augmentation Tool.

Downloads a LeRobot v3 dataset from HuggingFace Hub, applies augmentations
(color/lighting, background replacement, state noise, temporal domain
randomization), optionally filters low-quality episodes, and uploads the
result as a new dataset.

Usage:
    python augment.py \\
        --source lerobot/aloha_static_cups_open \\
        --output YOUR_HF_USER/aloha_augmented \\
        --strategies color,background,noise,tempo \\
        --multiplier 3 \\
        --filter-quality \\
        --quality-threshold 0.3 \\
        --max-episodes 5
"""

from __future__ import annotations

import argparse
import sys

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from augmentations.quality import score_and_filter_episodes
from augmentations.state import augment_action, augment_state
from augmentations.temporal import resample_frame_indices, sample_speed_factor
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
            "color (ColorJitter + Gaussian noise on video frames), "
            "background (spatial domain randomization via rembg), "
            "noise (Gaussian noise on state/action tensors), "
            "tempo (temporal domain randomization — speed variation). "
            "Default: color"
        ),
    )
    parser.add_argument(
        "--multiplier", type=int, default=2,
        help="Augmented copies to create per episode. Default: 2",
    )
    parser.add_argument(
        "--filter-quality", action="store_true",
        help="Filter low-quality episodes before augmenting (multi-dimensional scoring).",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.3,
        help="Min overall quality score to keep [0,1]. Default: 0.3",
    )
    parser.add_argument(
        "--quality-report-only", action="store_true",
        help="Print quality report for all episodes without filtering or augmenting.",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Cap number of source episodes processed.",
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
        "--bg-style", default="random", choices=["solid", "gradient", "random"],
        help="Background style for spatial domain randomization. Default: random",
    )
    parser.add_argument(
        "--background-episodes", type=int, default=None,
        help=(
            "Limit background replacement to the first N source episodes. "
            "Other episodes still get the remaining strategies. "
            "Useful to cap rembg compute time while still demonstrating spatial "
            "domain randomization in the visualizer."
        ),
    )
    parser.add_argument(
        "--speed-min", type=float, default=0.75,
        help="Min speed factor for temporal domain randomization. Default: 0.75",
    )
    parser.add_argument(
        "--speed-max", type=float, default=1.25,
        help="Max speed factor for temporal domain randomization. Default: 1.25",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Upload dataset as private on HuggingFace Hub.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process data but skip upload. Always test with this first.",
    )
    return parser.parse_args()


def get_image_keys(dataset: LeRobotDataset) -> list[str]:
    return [k for k, f in dataset.meta.features.items() if f.get("dtype") in ("image", "video")]


def get_data_keys(dataset: LeRobotDataset) -> list[str]:
    """Non-image feature keys to pass explicitly to add_frame()."""
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
    bg_style: str = "random",
) -> dict:
    """Return a new augmented frame dict ready for dataset.add_frame()."""
    result = {}

    # Video augmentation — augment_image works in CHW, add_frame expects HWC uint8
    for key in image_keys:
        t = frame.get(key)
        if isinstance(t, torch.Tensor):
            aug = augment_image(
                t, strategies=strategies, seed=seed,
                color_transform=color_transform, bg_style=bg_style,
            )
            result[key] = (aug.permute(1, 2, 0) * 255).clamp(0, 255).byte()
        else:
            result[key] = t

    # State / action augmentation (float tensors only)
    for key in data_keys:
        t = frame.get(key)
        if not isinstance(t, torch.Tensor):
            result[key] = t
            continue
        expected_shape = tuple(source_features.get(key, {}).get("shape", t.shape))
        t = t.reshape(expected_shape) if t.shape != expected_shape else t

        if "noise" in strategies and t.is_floating_point():
            if "action" in key:
                result[key] = augment_action(t, noise_scale=action_noise_scale, seed=seed)
            else:
                result[key] = augment_state(t, noise_scale=state_noise_scale, seed=seed)
        else:
            result[key] = t

    if "task" in frame:
        result["task"] = frame["task"]

    return result


def run(args: argparse.Namespace) -> None:
    strategies = {s.strip().lower() for s in args.strategies.split(",")}
    valid = {"color", "background", "noise", "tempo"}
    if unknown := strategies - valid:
        print(f"Unknown strategies: {unknown}. Valid: {valid}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  LeRobot Dataset Augmentation Tool")
    print(f"{'='*60}")
    print(f"  Source:       {args.source}")
    print(f"  Output:       {args.output}")
    print(f"  Strategies:   {', '.join(sorted(strategies))}")
    print(f"  Multiplier:   {args.multiplier}x")
    print(f"  Max episodes: {args.max_episodes or 'all'}")
    if args.quality_report_only:
        print(f"  Mode:         quality report only (no augmentation)")
    elif args.filter_quality:
        print(f"  Quality filter: ON (threshold={args.quality_threshold})")
    if "background" in strategies:
        print(f"  BG style:     {args.bg_style}")
    if "tempo" in strategies:
        print(f"  Speed range:  {args.speed_min}x – {args.speed_max}x")
    print(f"  Dry run:      {args.dry_run}")
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
    # 2. Quality scoring / filtering
    # -------------------------------------------------------------------------
    if args.quality_report_only or args.filter_quality:
        label = "report" if args.quality_report_only else "filter"
        print(f"\n[2/5] Multi-dimensional quality {label} (parquet, no video decode)...")
        ep_indices = score_and_filter_episodes(
            source, ep_indices,
            threshold=args.quality_threshold,
            verbose=True,
            report_only=args.quality_report_only,
        )
        if args.quality_report_only:
            print("\n  Report complete. Exiting (--quality-report-only).")
            return
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
    # 4. Augment — stream frame-by-frame (constant memory)
    # -------------------------------------------------------------------------
    print(f"[4/5] Augmenting episodes...")
    ep_out_count = 0
    # Track how many source episodes have had background applied
    bg_ep_count = 0
    bg_limit = args.background_episodes  # None = unlimited

    for variant_idx in range(args.multiplier):
        desc = f"  Variant {variant_idx + 1}/{args.multiplier}"
        for ep_idx in tqdm(ep_indices, desc=desc, unit="ep"):
            start, end = get_episode_bounds(source, ep_idx)
            base_seed = variant_idx * 100_000 + ep_idx * 1_000

            # Per-episode strategy set: drop background once limit is reached
            ep_strategies = set(strategies)
            if "background" in ep_strategies and bg_limit is not None:
                if bg_ep_count >= bg_limit:
                    ep_strategies = ep_strategies - {"background"}

            # Temporal domain randomization: resample frame sequence
            # source[global_idx] returns both video frame AND parquet row
            # as one dict — skipping an index skips both simultaneously.
            if "tempo" in ep_strategies:
                speed = sample_speed_factor(
                    seed=base_seed,
                    speed_min=args.speed_min,
                    speed_max=args.speed_max,
                )
                frame_indices = resample_frame_indices(start, end, speed)
            else:
                frame_indices = list(range(start, end))

            for frame_offset, global_idx in enumerate(frame_indices):
                frame = source[global_idx]
                aug = augment_frame(
                    frame=frame,
                    strategies=ep_strategies,
                    image_keys=image_keys,
                    data_keys=data_keys,
                    seed=base_seed + frame_offset,
                    color_transform=color_transform,
                    state_noise_scale=args.state_noise_scale,
                    action_noise_scale=args.action_noise_scale,
                    source_features=source.meta.features,
                    bg_style=args.bg_style,
                )
                out_dataset.add_frame(aug)

            if "background" in ep_strategies:
                bg_ep_count += 1
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
