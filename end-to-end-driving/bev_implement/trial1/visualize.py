import torch
import numpy as np
from PIL import Image


def _box_blur(gray, passes=1):
    out = gray.astype(np.float32)
    for _ in range(max(0, passes)):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return out


def bev_to_image(bev_tensor, vmin=None, vmax=None, blur_passes=0):
    """
    bev_tensor: (1, H, W) or (H, W)
    Returns PIL image
    """

    if bev_tensor.dim() == 3:
        bev_tensor = bev_tensor.squeeze(0)

    bev = bev_tensor.detach().cpu().numpy().astype(np.float32)

    if blur_passes > 0:
        bev = _box_blur(bev, passes=blur_passes)

    if vmin is None:
        vmin = float(np.percentile(bev, 1.0))
    if vmax is None:
        vmax = float(np.percentile(bev, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    bev = np.clip((bev - vmin) / (vmax - vmin), 0.0, 1.0)
    bev = (bev * 255).astype(np.uint8)

    return Image.fromarray(bev)
