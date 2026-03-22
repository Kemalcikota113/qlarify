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
            _REMBG_SESSION = new_session("u2net")
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


def replace_background(
    image_tensor: torch.Tensor,
    bg_color: tuple[int, int, int] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Remove background with rembg and composite onto a random solid color.

    This implements domain randomization: the robot arm is kept but the
    workspace background is replaced with a new color, forcing the model
    to focus on the robot rather than background cues.

    Args:
        image_tensor: Float32 CHW [0, 1] tensor.
        bg_color: RGB tuple for background. Random if None.
        seed: RNG seed for reproducible background color selection.

    Returns:
        Augmented float32 CHW [0, 1] tensor with replaced background.
    """
    from rembg import remove

    if seed is not None:
        np.random.seed(seed)

    session = _get_rembg_session()
    pil_img = _tensor_to_pil(image_tensor)

    # Remove background → RGBA image with transparent background
    fg_rgba = remove(pil_img, session=session)

    # Choose random background color (avoiding very dark or very light)
    if bg_color is None:
        bg_color = tuple(np.random.randint(60, 200, 3).tolist())

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
) -> torch.Tensor:
    """Apply requested augmentation strategies to a single camera frame.

    Args:
        image_tensor: Float32 CHW [0, 1].
        strategies: Set of strategy names: {"color", "background"}.
        seed: RNG seed (use different seeds per episode variant).
        color_transform: Pre-built color transform for reuse.
        bg_color: Fixed background color override (random if None).

    Returns:
        Augmented float32 CHW [0, 1] tensor.
    """
    result = image_tensor

    if "background" in strategies:
        result = replace_background(result, bg_color=bg_color, seed=seed)

    if "color" in strategies:
        result = color_augment(result, seed=seed, transform=color_transform)

    return result
