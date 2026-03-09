import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_waymo import WaymoDataset
from preprocess import preprocess


def create_pixel_grid(H, W, device):
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    return torch.stack([xs, ys, ones], dim=-1)


def main():

    dataset_path = "/home/nataniel/Desktop/work2/aiea/challenge_team/data/pytorch"
    dataset = WaymoDataset(dataset_path, "training")

    frame = dataset[0]

    # Use first camera only
    cam = frame.images[0]
    calib = frame.context.camera_calibrations[0]

    img = preprocess(cam.image)  # (3,H,W)
    H, W = img.shape[1:]

    fu, fv, cu, cv = calib.intrinsic[:4]

    K = torch.tensor([
        [fu, 0, cu],
        [0, fv, cv],
        [0, 0, 1]
    ], dtype=torch.float32)

    K_inv = torch.inverse(K)

    grid = create_pixel_grid(H, W, device="cpu")
    grid = grid.view(-1, 3).T  # (3, H*W)

    # Fake constant depth = 20 meters
    depth = 20.0
    rays = K_inv @ grid
    points_cam = rays * depth

    # Homogeneous
    ones = torch.ones((1, points_cam.shape[1]))
    points_cam_h = torch.cat([points_cam, ones], dim=0)

    cam2ego = torch.tensor(calib.extrinsic, dtype=torch.float32)

    points_ego = cam2ego @ points_cam_h
    points_ego = points_ego[:3]

    x = points_ego[0]
    y = points_ego[1]

    print("X range:", x.min().item(), x.max().item())
    print("Y range:", y.min().item(), y.max().item())

    # Plot scatter
    plt.figure(figsize=(6,6))
    plt.scatter(x.numpy(), y.numpy(), s=0.1)
    plt.xlabel("X (forward)")
    plt.ylabel("Y (left)")
    plt.title("Geometry Debug Projection")
    plt.show()


if __name__ == "__main__":
    main()