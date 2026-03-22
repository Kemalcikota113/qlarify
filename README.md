# LeRobot Dataset Augmentation Tool

A CLI tool that downloads a [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset from HuggingFace Hub, applies multiple augmentation strategies to multiply and diversify the data, and uploads the result as a new dataset — ready to improve your robot's real-world robustness.

## What it does

**The core problem in robot learning:** collecting robot demonstration data is slow and expensive. A dataset of 100 episodes might take days to record. This tool lets you take those 100 episodes and produce 300-500 augmented variations, significantly increasing dataset diversity.

**Three augmentation strategies:**

| Strategy | What it does | Why it helps |
|----------|-------------|--------------|
| `color` | Random ColorJitter (brightness, contrast, saturation, hue) + Gaussian noise on video frames | Teaches the robot to ignore lighting conditions |
| `background` | AI-powered background removal (rembg/U2Net) + random solid color replacement | **Domain randomization** — the most effective robustness technique. The robot learns to focus on the task objects, not background cues |
| `noise` | Small Gaussian noise on joint state observations and actions | Simulates sensor noise and slight positional variance |

**Quality filtering:** Before augmenting, optionally score each episode by action smoothness (jerk = second derivative of the action sequence). Low-quality episodes (trembling, failed grasps) are filtered out — you augment the *best* data, not the worst.

**Automatic upload + visualizer link:** After augmentation, the dataset is automatically uploaded to HuggingFace Hub and a direct visualizer link is printed.

## Installation

```bash
git clone https://github.com/your-username/qualia-challenge
cd qualia-challenge
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # or: export HF_TOKEN=hf_...
```

## Usage

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

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | required | Source dataset repo ID on HuggingFace |
| `--output` | required | Output repo ID (`your-user/dataset-name`) |
| `--strategies` | `color` | Comma-separated: `color`, `background`, `noise` |
| `--multiplier` | `2` | How many augmented copies per source episode |
| `--filter-quality` | off | Enable action-smoothness quality filtering |
| `--quality-threshold` | `0.3` | Smoothness score cutoff [0,1] |
| `--max-episodes` | all | Cap source episodes (useful for quick tests) |
| `--state-noise-scale` | `0.01` | Std dev for state noise |
| `--action-noise-scale` | `0.005` | Std dev for action noise |
| `--private` | off | Upload as private dataset |
| `--dry-run` | off | Process locally, skip upload |

### Quick smoke test (2 episodes, no upload)

```bash
python augment.py \
  --source lerobot/aloha_static_cups_open \
  --output YOUR_USER/test-augmented \
  --strategies color \
  --multiplier 1 \
  --max-episodes 2 \
  --dry-run
```

### Full run with all strategies

```bash
python augment.py \
  --source lerobot/aloha_static_cups_open \
  --output YOUR_USER/aloha_augmented_full \
  --strategies color,background,noise \
  --multiplier 3 \
  --filter-quality \
  --quality-threshold 0.3 \
  --max-episodes 10
```

After completion, a visualizer link is printed:
```
Visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FYOUR_USER%2Faloha_augmented_full%2Fepisode_0
```

## Architecture

```
augment.py                  ← CLI entry point (argparse)
augmentations/
  video.py                  ← ColorJitter + rembg background replacement
  state.py                  ← Gaussian noise on state/action tensors
  quality.py                ← Episode smoothness scoring + filtering
utils/
  dataset.py                ← LeRobot v3 helpers (episode iteration, schema mirroring)
```

**Key design decisions:**
- **Streaming frame iteration**: frames are processed one at a time — constant memory regardless of dataset size (no pre-loading 400×4 images into RAM)
- **Fast quality scoring**: reads parquet action data directly, no video decode
- **Schema mirroring**: output dataset automatically copies fps, features, robot_type from source

## How I used AI coding agents

This tool was built in ~4 hours using **Claude Code** (Anthropic's CLI agent) as the primary coding tool.

**Phase 1 — API exploration (delegated entirely to Claude):**
I asked Claude to fetch and analyze the LeRobot v3 source code from GitHub and the HuggingFace docs. It returned a detailed technical summary of the `add_frame()` / `save_episode()` / `finalize()` pipeline, the exact tensor shapes expected (`CHW float32 [0,1]` vs `HWC uint8` for add_frame), and the episode metadata structure. This saved ~1 hour of manual API archaeology.

**Phase 2 — Architecture design:**
Rather than asking Claude to "write the code", I asked it to "architect a modular augmentation pipeline" and discussed the tradeoffs (pre-loading vs streaming, per-episode parquet reads vs single load). Claude identified the OOM risk of pre-loading 400×4 camera frames before I ran into it.

**Phase 3 — Iterative debugging:**
When `add_frame()` threw shape mismatches (`(3,480,640)` vs expected `(480,640,3)`), Claude diagnosed the CHW→HWC conversion issue and the `next.done` scalar reshape problem immediately from the error messages.

**Phase 4 — This README:**
Written by Claude based on the implemented code and the challenge requirements.

The key insight from this workflow: **AI agents are most valuable for API exploration and debugging**, not just code generation. Claude's ability to read library source code and docs and extract exactly the API patterns I needed was the highest-leverage use of the tool.
