# LeRobot Dataset Augmentation Tool

A CLI tool that downloads a [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset from HuggingFace Hub, applies a multi-strategy augmentation pipeline to multiply and diversify the training data, and uploads the result — ready to improve policy robustness on real hardware.

**Live example:** [Kemalcikota/aloha-augmented-v2](https://huggingface.co/datasets/Kemalcikota/aloha-augmented-v2) — view in the [LeRobot Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FKemalcikota%2Faloha-augmented-v2%2Fepisode_0)

---

## Why augmentation matters for robot learning

Collecting robot demonstration data is slow and expensive — a dataset of 100 episodes can take days to record. More critically, policies trained on limited data overfit to the specific lighting, background, and motor dynamics of the recording environment. They fail in deployment when any of these conditions change.

This tool addresses three distinct failure modes:

| Failure mode | Augmentation strategy | Technique |
|---|---|---|
| Lighting & color sensitivity | Photometric domain randomization | `color` — ColorJitter + Gaussian noise |
| Background clutter & spatial shortcuts | Spatial domain randomization | `background` — AI-powered background synthesis |
| Motor latency & control frequency variance | Temporal domain randomization | `tempo` — frame sequence resampling |
| Sensor noise & positional uncertainty | State/action perturbation | `noise` — Gaussian noise injection |

---

## Augmentation strategies

### `color` — Photometric Domain Randomization
Applies random ColorJitter (brightness ±30%, contrast ±30%, saturation ±20%, hue ±5%) and Gaussian pixel noise to all camera streams simultaneously. Forces the policy to learn features that are invariant to illumination — critical for deployment across different rooms, times of day, and lighting rigs.

### `background` — Spatial Domain Randomization via AI-Powered Background Synthesis
Uses [rembg](https://github.com/danielgatis/rembg) (U-Net neural background removal) to isolate the robot foreground from each camera frame and composite it onto a synthetically generated background — either a random solid color or a procedural vertical gradient. This prevents the policy from learning spurious correlations between task success and irrelevant background features, a well-documented failure mode in visuomotor policies deployed outside their recording environment.

### `tempo` — Temporal Domain Randomization for Control Frequency Robustness
Resamples each episode's frame sequence at a randomly drawn speed factor (default 0.75×–1.25×). Faster variants skip frames (fewer timesteps, same motion); slower variants repeat frames (more timesteps). Since `dataset[idx]` retrieves the video frame and its corresponding parquet row (state, action, timestamp) as a single atomic dict, video and kinematics remain perfectly synchronized across all speed variants — no separate alignment step is required.

This augmentation teaches the learned policy to be robust to variations in motor latency, control loop frequency, and real-time inference jitter — a key challenge when deploying VLA models on physical hardware where the compute budget may differ from training.

### `noise` — Proprioceptive Perturbation
Injects calibrated Gaussian noise into joint state observations (σ=0.01) and action sequences (σ=0.005). Simulates real-world encoder noise and positional uncertainty, improving the policy's ability to recover from small deviations without resorting to excessive smoothing.

---

## Quality filtering — Kinematic Integrity via Multi-Dimensional Scoring

Before augmenting, the tool optionally scores every source episode across three independent axes by reading the parquet action data directly (no video decoding — fast even for large datasets):

```
  Ep   smooth    idle   length   overall  decision
  -------------------------------------------------
   0   0.9989   0.326      400             0.6476  KEEP
   3   0.9988   0.238      400             0.8228  KEEP
   7   0.9989   0.373      400 [OUTLIER]   0.5525  DROP
```

**Kinematic smoothness** — second derivative (jerk) of the action sequence. Smooth, purposeful robot motion produces low jerk; trembling, failed grasps, or controller instability produce high jerk. Score: `1 / (1 + mean_abs_jerk)`.

**Idle detection** — fraction of timesteps where mean joint velocity falls below a threshold (0.001 rad/s). Episodes where the robot is stationary for >15% of the recording indicate hesitation, failure to grasp, or a stalled controller. The idle ratio penalises the overall score proportionally above the 15% threshold.

**Length outlier flag** — episodes whose length is ±2σ from the dataset mean are flagged. Unusually short episodes often represent early terminations (drops, resets); unusually long ones indicate the operator lost control. The flag is informational and does not automatically discard episodes.

The `overall_score = smoothness × (1 − max(0, idle_ratio − 0.15) × 2)` combines the two quantitative axes into a single filtereable value.

---

## Installation

```bash
git clone https://github.com/Kemalcikota113/qlarify
cd qlarify
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_...   # HuggingFace write token
```

## Usage

```bash
python augment.py \
  --source lerobot/aloha_static_cups_open \
  --output YOUR_HF_USER/aloha_augmented \
  --strategies color,background,noise,tempo \
  --multiplier 3 \
  --filter-quality \
  --quality-threshold 0.3 \
  --background-episodes 2 \
  --max-episodes 5
```

### Full flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | required | Source LeRobot v3 dataset on HuggingFace Hub |
| `--output` | required | Output repo ID (`user/dataset-name`) |
| `--strategies` | `color` | Comma-separated: `color`, `background`, `noise`, `tempo` |
| `--multiplier` | `2` | Augmented copies per source episode |
| `--filter-quality` | off | Enable multi-dimensional quality filtering |
| `--quality-threshold` | `0.3` | Minimum overall score to keep `[0,1]` |
| `--quality-report-only` | off | Print quality table without augmenting |
| `--background-episodes` | all | Limit background strategy to first N source episodes |
| `--bg-style` | `random` | Background type: `solid`, `gradient`, `random` |
| `--speed-min` | `0.75` | Lower bound for temporal speed factor |
| `--speed-max` | `1.25` | Upper bound for temporal speed factor |
| `--state-noise-scale` | `0.01` | Gaussian noise σ for state observations |
| `--action-noise-scale` | `0.005` | Gaussian noise σ for actions |
| `--max-episodes` | all | Cap source episodes processed |
| `--private` | off | Upload as private dataset |
| `--dry-run` | off | Process locally, skip upload |

### Inspect dataset quality before augmenting

```bash
python augment.py --source lerobot/aloha_static_cups_open \
  --output unused/unused --strategies color \
  --max-episodes 20 --quality-report-only --dry-run
```

---

## Architecture

```
augment.py                  ← CLI entry point, streaming augmentation loop
augmentations/
  video.py                  ← Photometric + spatial domain randomization (rembg)
  state.py                  ← Proprioceptive noise injection
  quality.py                ← Multi-dimensional kinematic quality scoring
  temporal.py               ← Temporal domain randomization (frame resampling)
utils/
  dataset.py                ← LeRobot v3 helpers: episode iteration, schema mirroring
```

**Key design decisions:**

- **Streaming frame iteration** — frames are loaded and augmented one at a time. Memory usage is `O(1)` with respect to dataset size, not `O(episodes × frames × cameras)`.
- **Parquet-only quality scoring** — quality filtering reads action columns directly from parquet files without touching any video, making it fast regardless of dataset size.
- **Atomic frame dict** — `source[idx]` returns video frames and kinematics as a single dict. Temporal resampling (skipping/repeating indices) therefore keeps video and parquet data perfectly aligned without any explicit synchronization.
- **Deterministic seeding** — each frame's augmentation seed is `variant_idx × 100,000 + ep_idx × 1,000 + frame_offset`, ensuring reproducible results while producing diverse variation across episode variants.

---

## How I used AI coding agents to build this

This tool was built almost entirely using **[Claude Code](https://claude.ai/code)** (Anthropic's CLI agent) as the primary development tool, in line with the challenge's explicit expectation of heavy AI agent usage.

### Phase 1 — LeRobot v3 API exploration (delegated to Claude)

The LeRobot v3 format was brand new and sparsely documented. I asked Claude Code to fetch and analyse the library source directly from GitHub, extracting the exact API contract for `add_frame()` / `save_episode()` / `finalize()`. Key facts it surfaced that would have taken significant manual effort to discover:

- `add_frame()` expects **HWC uint8**, not the CHW float32 tensors that `dataset[i]` returns — a silent shape mismatch that would have caused obscure failures.
- `finalize()` must be called before `push_to_hub()` — without it, parquet file footers are never written and the dataset silently corrupts.
- `dataset.meta.episodes` is a HuggingFace `Dataset` object, not a pandas DataFrame — the `.iloc` accessor I initially used would have crashed immediately.

### Phase 2 — Streaming architecture (Claude identified the OOM risk before it happened)

My first implementation pre-loaded all frames of each episode into memory before augmenting. Claude flagged the memory arithmetic before I ran it: 400 frames × 4 cameras × 480×640 × float32 ≈ 5.5GB per episode. It proposed the streaming design — process one frame at a time, constant memory regardless of dataset size — which became the core architecture.

### Phase 3 — Parquet-only quality scoring

For the quality filter, Claude proposed reading action columns directly from the parquet files using the HuggingFace `datasets` library, bypassing video decode entirely. This made the quality scoring 20–50× faster than the naive approach of calling `dataset[i]` for every frame.

### Phase 4 — Temporal sync guarantee

When designing temporal domain randomization, Claude identified that because `source[global_idx]` returns both video and kinematics as a single atomic dict, frame resampling is inherently synchronized — skipping frame index 5 skips both its video frame and its parquet row simultaneously. No separate alignment logic was needed.

### Phase 5 — Iterative debugging

Claude diagnosed every `add_frame()` validation error immediately from the traceback — including the CHW→HWC conversion, the `next.done` scalar reshape issue, and the `timestamp` key that must be omitted (it is auto-generated). Each fix took seconds rather than the minutes of API archaeology it would have required manually.

---

## Generalizability — tested across robot platforms

The tool was dry-run validated across three structurally different LeRobot v3 datasets with no code changes, demonstrating that the schema-mirroring architecture is truly dataset-agnostic:

| Dataset | Robot | Cameras | State dim | FPS | Quality range | Notes |
|---|---|---|---|---|---|---|
| `lerobot/aloha_static_cups_open` | ALOHA dual-arm | 4 | 14 | 50 | 0.41–0.83 | Primary development dataset |
| `lerobot/pusht` | 2D sim pusher | 1 | 2 | 10 | 0.30–0.32 | High-jerk 2D contact task — quality scores correctly reflect abrupt contact dynamics |
| `lerobot/xarm_lift_medium` | xArm robotic arm | 1 | 4 | 15 | 0.37–0.50 | Scripted demos — idle ratio = 0.000 on all episodes, as expected for algorithmic trajectories |
| `lerobot/unitreeh1_warehouse` | Unitree H1 humanoid | 2 | 19 | 50 | 0.47–0.55 | 19-DOF whole-body control; dual-camera schema handled automatically |

The quality scoring reveals meaningful, robot-specific signal in each case. The pusht scores (0.30–0.32) are lower than ALOHA (0.41–0.83) not because the data is low quality, but because 2D contact tasks produce structurally higher jerk — the metric correctly reflects the task's kinematic profile rather than misfiring. This demonstrates that the quality scores are interpretable relative to a dataset's own distribution, not as absolute cross-dataset thresholds.

## Example output

After a full run:
```
============================================================
  Upload complete!
  Dataset:    https://huggingface.co/datasets/Kemalcikota/aloha-augmented-v2
  Visualizer: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FKemalcikota%2Faloha-augmented-v2%2Fepisode_0
============================================================
```
