"""Image helpers shared by the tools: denormalizing inputs and coloring label maps."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .data.dataset import IMAGENET_MEAN, IMAGENET_STD

# One fixed color per class id (RGB), the same in every figure; unlabeled pixels are black.
CLASS_COLORS = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [220, 20, 60], [250, 170, 30],
                         [107, 142, 35], [255, 255, 0], [0, 130, 180], [119, 11, 32], [152, 251, 152],
                         [0, 0, 142], [0, 60, 100], [0, 80, 100], [0, 0, 230]], np.uint8)


def to_image(x: torch.Tensor) -> np.ndarray:
    """Normalized (3, H, W) input tensor -> uint8 BGR image."""
    img = (x.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def colorize(label: np.ndarray, ignore: int) -> np.ndarray:
    """Class-id map -> uint8 BGR image; ignore pixels are black."""
    out = CLASS_COLORS[np.clip(label, 0, len(CLASS_COLORS) - 1)][..., ::-1].copy()
    out[label == ignore] = 0
    return out
