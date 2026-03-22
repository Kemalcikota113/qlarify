# Qualia Challenge — LeRobot Dataset Augmentation Tool

## Project Goal
Build a CLI tool that downloads a LeRobot v3 dataset from HuggingFace Hub, applies augmentations (color/lighting, background replacement, state noise), filters low-quality episodes, uploads the result, and prints a visualizer link.

## File Structure
```
augment.py              ← main CLI entry point
augmentations/
  video.py              ← color jitter, background replacement (rembg)
  state.py              ← Gaussian noise on state/action tensors
  quality.py            ← episode smoothness scoring + filtering
utils/
  dataset.py            ← LeRobot helpers: iter_episodes, create_output_dataset
```

## CLI Usage
```bash
python augment.py \
  --source lerobot/aloha_static_cups_open \
  --output YOUR_HF_USER/aloha_augmented \
  --strategies color,background,noise \
  --multiplier 3 \
  --filter-quality \
  --quality-threshold 0.3 \
  --max-episodes 20
```

## LeRobot v3 API — Key Facts
- `LeRobotDataset(repo_id)` loads from HF Hub (cached locally)
- `dataset[idx]` → dict of tensors: images as **float32 CHW [0,1]**, states/actions as float32
- `frame["task"]` is a required string when calling `add_frame()`
- Episode boundaries: `dataset.meta.episodes` DataFrame, columns `dataset_from_index`, `dataset_to_index`
- Mirror schema: copy `source.meta.fps`, `source.meta.features`, `source.meta.robot_type`
- Pipeline: `add_frame()` per frame → `save_episode()` per episode → `finalize()` → `push_to_hub()`
- **CRITICAL**: Must call `finalize()` before `push_to_hub()` — skipping corrupts parquet files

## Environment Setup
```bash
source venv/bin/activate
huggingface-cli login   # or export HF_TOKEN=...
```

## Verification Smoke Test
```bash
python augment.py --source lerobot/aloha_static_cups_open \
  --output YOUR_USER/test-augmented \
  --strategies color --multiplier 1 --max-episodes 2
```
