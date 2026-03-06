import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional 

# Lift function
def create_pixel_grid(self, H, W, device):
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=-1)  # (H,W,3)
    return grid


def lift_to_3d(features, depth_prob, K):
    B, N, D, Hf, Wf = depth_prob.shape
    device = depth_prob.device

    depth_values = torch.linspace(1.0, 50.0, D, device=device)

    grid = create_pixel_grid(Hf, Wf, device)
    grid = grid.view(-1,3).T  # (3,HW)

    points_all = []

    for b in range(B):
        for n in range(N):
            K_inv = torch.inverse(K[b,n])
            rays = K_inv @ grid
            rays = rays.view(3, Hf, Wf)

            for d in range(D):
                depth = depth_values[d]
                pts = rays * depth
                points_all.append(pts)

    return torch.stack(points_all)