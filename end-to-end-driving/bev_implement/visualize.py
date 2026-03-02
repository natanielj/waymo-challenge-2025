import torch
import numpy as np
from PIL import Image


def bev_to_image(bev_tensor):
    """
    bev_tensor: (1, H, W) or (H, W)
    Returns PIL image
    """

    if bev_tensor.dim() == 3:
        bev_tensor = bev_tensor.squeeze(0)

    bev = torch.sigmoid(bev_tensor)  # map to 0–1
    bev = bev.detach().cpu().numpy()

    bev = (bev - bev.min()) / (bev.max() - bev.min() + 1e-6)
    bev = (bev * 255).astype(np.uint8)

    return Image.fromarray(bev)