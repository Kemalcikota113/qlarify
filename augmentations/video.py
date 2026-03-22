"""Visual augmentations for robot camera frames."""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image


# ---------------------------------------------------------------------------
# Level 1: Color / lighting augmentation
# ---------------------------------------------------------------------------

def build_color_transform(
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.2,
    hue: float = 0.05,
    noise_sigma: float = 0.02,
) -> T.Compose:
    """Build a deterministic-looking ColorJitter + noise transform."""
    transforms = [
        T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
    ]
    if noise_sigma > 0:
        transforms.append(T.GaussianNoise(mean=0.0, sigma=noise_sigma))
    return T.Compose(transforms)


def color_augment(
    image_tensor: torch.Tensor,
    seed: int | None = None,
    transform: T.Compose | None = None,
) -> torch.Tensor:
    """Apply color/lighting augmentation to a single image frame.

    Args:
        image_tensor: Float32 CHW tensor in [0, 1].
        seed: Optional seed for reproducible augmentation.
        transform: Pre-built transform (built once and reused for speed).

    Returns:
        Augmented float32 CHW tensor in [0, 1].
    """
    if transform is None:
        transform = build_color_transform()
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    # torchvision v2 transforms accept CHW float tensors directly
    return torch.clamp(transform(image_tensor), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Level 2: Background replacement via rembg
# ---------------------------------------------------------------------------

_REMBG_SESSION = None  # lazy-loaded singleton


def _get_rembg_session():
    """Lazy-load rembg session (downloads U2Net model on first call)."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            from rembg import new_session
            print("  Loading rembg U2Net model (first-time download may take a moment)...")
            _REMBG_SESSION = new_session("u2netp")  # pruned model, ~3× faster on CPU
            print("  rembg model loaded.")
        except ImportError:
            raise ImportError(
                "rembg is required for background replacement. "
                "Install with: pip install rembg"
            )
    return _REMBG_SESSION


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert float32 CHW [0,1] tensor to uint8 HWC PIL Image."""
    img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np, mode="RGB")


def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert RGB PIL Image to float32 CHW [0,1] tensor."""
    img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1)


def build_gradient_background(
    size: tuple[int, int],
    seed: int | None = None,
) -> Image.Image:
    """Create a random vertical linear gradient RGBA background image.

    Picks two random muted RGB colors and interpolates linearly from top to
    bottom, producing varied but visually clean backgrounds for spatial
    domain randomization.

    Args:
        size: (width, height) of the output image.
        seed: RNG seed for reproducibility.

    Returns:
        RGBA PIL Image (alpha=255 everywhere) of the given size.
    """
    rng = np.random.default_rng(seed)
    color_top = rng.integers(50, 200, size=3)
    color_bot = rng.integers(50, 200, size=3)

    width, height = size
    gradient = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        t = y / max(height - 1, 1)
        rgb = ((1 - t) * color_top + t * color_bot).round().astype(np.uint8)
        gradient[y, :, :3] = rgb
        gradient[y, :, 3] = 255  # fully opaque

    return Image.fromarray(gradient, mode="RGBA")


def replace_background(
    image_tensor: torch.Tensor,
    bg_color: tuple[int, int, int] | None = None,
    bg_style: str = "random",
    seed: int | None = None,
) -> torch.Tensor:
    """Remove background with rembg and composite onto a generated background.

    Implements spatial domain randomization: the robot arm foreground is
    preserved while the workspace background is replaced, preventing the
    policy from learning background-specific shortcuts.

    Args:
        image_tensor: Float32 CHW [0, 1] tensor.
        bg_color: Fixed RGB color for solid background (random if None).
        bg_style: "solid" | "gradient" | "random" (50/50 solid vs gradient).
        seed: RNG seed for reproducible background generation.

    Returns:
        Augmented float32 CHW [0, 1] tensor with replaced background.
    """
    from rembg import remove

    rng = np.random.default_rng(seed)

    session = _get_rembg_session()
    pil_img = _tensor_to_pil(image_tensor)

    # Remove background → RGBA with transparent background
    fg_rgba = remove(pil_img, session=session)

    # Choose background type
    use_gradient = (
        bg_style == "gradient"
        or (bg_style == "random" and rng.random() > 0.5)
    )

    if use_gradient:
        background = build_gradient_background(pil_img.size, seed=int(rng.integers(0, 2**31)))
    else:
        if bg_color is None:
            bg_color = tuple(rng.integers(60, 200, size=3).tolist())
        background = Image.new("RGBA", pil_img.size, bg_color + (255,))

    composite = Image.alpha_composite(background, fg_rgba).convert("RGB")
    return _pil_to_tensor(composite)


# ---------------------------------------------------------------------------
# Combined augmentation dispatcher
# ---------------------------------------------------------------------------

def augment_image(
    image_tensor: torch.Tensor,
    strategies: set[str],
    seed: int | None = None,
    color_transform: T.Compose | None = None,
    bg_color: tuple[int, int, int] | None = None,
    bg_style: str = "random",
) -> torch.Tensor:
    """Apply requested augmentation strategies to a single camera frame.

    Args:
        image_tensor: Float32 CHW [0, 1].
        strategies: Set of strategy names: {"color", "background"}.
        seed: RNG seed (use different seeds per episode variant).
        color_transform: Pre-built color transform for reuse.
        bg_color: Fixed background color override for solid backgrounds.
        bg_style: "solid" | "gradient" | "random" for background style.

    Returns:
        Augmented float32 CHW [0, 1] tensor.
    """
    result = image_tensor

    if "background" in strategies:
        result = replace_background(result, bg_color=bg_color, bg_style=bg_style, seed=seed)

    if "color" in strategies:
        result = color_augment(result, seed=seed, transform=color_transform)

    return result
