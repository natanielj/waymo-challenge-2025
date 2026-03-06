import torch
from torch.utils.data import DataLoader
import argparse

from torch_waymo import WaymoDataset
from torch_waymo.protocol.dataset_proto import CameraName

from preprocess import preprocess
from lift_splat import MultiCamLiftSplatBEV
import numpy as np
import torch.nn.functional as F

import imageio
from visualize import bev_to_image

# Wrapper Dataset

class WaymoBEVDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split="training"):
        self.dataset = WaymoDataset(dataset_path, split)
        self.target_size = (256, 704)  # (H, W)
        self.camera_order = [
            CameraName.FRONT_LEFT,
            CameraName.FRONT,
            CameraName.FRONT_RIGHT,
            CameraName.SIDE_LEFT,
            CameraName.SIDE_RIGHT,
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frame = self.dataset[idx]

        images = []
        intrinsics = []
        cam2ego = []

        # Build lookup table for calibration
        calib_dict = {
            calib.name: calib
            for calib in frame.context.camera_calibrations
        }

        image_dict = {img.name: img for img in frame.images}

        for cam_name in self.camera_order:
            if cam_name not in image_dict:
                raise ValueError(f"Missing camera image for camera {cam_name} in frame {idx}")
            if cam_name not in calib_dict:
                raise ValueError(f"Missing camera calibration for camera {cam_name} in frame {idx}")

            cam = image_dict[cam_name]

            img = preprocess(cam.image)  # (3,H,W)

            orig_h, orig_w = img.shape[1:]

            img = F.interpolate(
                img.unsqueeze(0),
                size=self.target_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            images.append(img)

            calib = calib_dict[cam_name]

            fu, fv, cu, cv = calib.intrinsic[:4]

            scale_x = self.target_size[1] / orig_w
            scale_y = self.target_size[0] / orig_h

            K = torch.tensor([
                [fu * scale_x, 0.0, cu * scale_x],
                [0.0, fv * scale_y, cv * scale_y],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32)

            intrinsics.append(K)

            if hasattr(calib.extrinsic, "transform"):
                extrinsic_values = calib.extrinsic.transform
            else:
                extrinsic_values = calib.extrinsic
            T = torch.tensor(extrinsic_values, dtype=torch.float32).view(4, 4)
            cam2ego.append(T)
        
        images = torch.stack(images)
        intrinsics = torch.stack(intrinsics)
        cam2ego = torch.stack(cam2ego)

        return {
            "images": images,
            "intrinsics": intrinsics,
            "cam2ego": cam2ego,
            "scenario_name": frame.context.name,
        }

def save_gif(name, frames):
    safe_name = name.replace("/", "_")
    output_path = f"bev_{safe_name}.gif"
    imageio.mimsave(output_path, frames, fps=5)
    print(f"Saved {output_path}")


def _unnormalize_image(img):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0.0, 1.0)


def front_camera_ipm_bev(
    front_img,
    K,
    cam2ego,
    bev_x_range=(0.0, 50.0),
    bev_y_range=(-25.0, 25.0),
    bev_res=0.5,
    ground_z=-1.5,
):
    # front_img: (3, H, W), normalized by ImageNet stats
    # K: (3,3), cam2ego: (4,4)
    device = front_img.device
    dtype = front_img.dtype

    x_min, x_max = bev_x_range
    y_min, y_max = bev_y_range
    bev_h = int((x_max - x_min) / bev_res)
    bev_w = int((y_max - y_min) / bev_res)

    # BEV convention: top=far, bottom=near; left=left side of ego.
    xs = torch.linspace(x_max - 0.5 * bev_res, x_min + 0.5 * bev_res, bev_h, device=device, dtype=dtype)
    ys = torch.linspace(y_max - 0.5 * bev_res, y_min + 0.5 * bev_res, bev_w, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    ones = torch.ones_like(X)

    def project_with_ego2cam(ego2cam, z_plane):
        Z = torch.full_like(X, z_plane)
        points_ego = torch.stack([X, Y, Z, ones], dim=-1).view(-1, 4).T  # (4, H*W)
        points_cam = (ego2cam @ points_ego)[:3]  # (3, H*W)
        z = points_cam[2].clamp_min(1e-6)
        uv = K @ points_cam
        u = uv[0] / z
        v = uv[1] / z

        H, W = front_img.shape[1:]
        valid = (
            (points_cam[2] > 1e-4) &
            (u >= 0) & (u <= W - 1) &
            (v >= 0) & (v <= H - 1)
        )

        grid_x = (u / (W - 1)) * 2.0 - 1.0
        grid_y = (v / (H - 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, bev_h, bev_w, 2)

        img = _unnormalize_image(front_img)
        sampled = F.grid_sample(
            img.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        valid_map = valid.view(bev_h, bev_w).unsqueeze(0)
        sampled = sampled * valid_map
        valid_ratio = float(valid.float().mean().item())
        return sampled, valid_ratio

    # Some loaders expose cam->ego, others ego->cam; try both and keep the better projection.
    z_candidates = [ground_z, -1.0, -1.5, -2.0, 0.0]
    best_sampled = None
    best_ratio = -1.0

    for z_plane in z_candidates:
        sampled_a, ratio_a = project_with_ego2cam(torch.inverse(cam2ego), z_plane)
        if ratio_a > best_ratio:
            best_ratio = ratio_a
            best_sampled = sampled_a

        sampled_b, ratio_b = project_with_ego2cam(cam2ego, z_plane)
        if ratio_b > best_ratio:
            best_ratio = ratio_b
            best_sampled = sampled_b

    sampled = best_sampled

    return (sampled.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def render_bev_frames(bev_tensors):
    bev_np = np.stack([b.squeeze(0).detach().cpu().numpy() for b in bev_tensors], axis=0)
    vmin = float(np.percentile(bev_np, 1.0))
    vmax = float(np.percentile(bev_np, 99.0))
    frames = []
    for bev in bev_tensors:
        bev_img = bev_to_image(bev, vmin=vmin, vmax=vmax, blur_passes=1)
        frames.append(np.array(bev_img))
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="/workdir/data/pytorch-e2e/converted")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-scenarios", type=int, default=4)
    parser.add_argument("--bev-mode", type=str, default="front_ipm", choices=["front_ipm", "lift_splat"])
    parser.add_argument("--ground-z", type=float, default=-1.5)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    device = args.device

    dataset = WaymoBEVDataset(dataset_path, split=args.split)
    model = None
    if args.bev_mode == "lift_splat":
        model = MultiCamLiftSplatBEV().to(device)
        model.eval()

    current_scenario = None
    bev_tensors = []
    bev_frames = []
    saved_scenarios = 0
    front_idx = dataset.camera_order.index(CameraName.FRONT)

    for i in range(len(dataset)):
        sample = dataset[i]
        scenario_name = sample["scenario_name"]

        # If scenario changes → save previous GIF
        if current_scenario is not None and scenario_name != current_scenario:
            if args.bev_mode == "lift_splat":
                save_gif(current_scenario, render_bev_frames(bev_tensors))
            else:
                save_gif(current_scenario, bev_frames)
            saved_scenarios += 1
            if saved_scenarios >= args.max_scenarios:
                break
            bev_tensors = []
            bev_frames = []

        current_scenario = scenario_name

        images = sample["images"].unsqueeze(0).to(device)
        intrinsics = sample["intrinsics"].unsqueeze(0).to(device)
        cam2ego = sample["cam2ego"].unsqueeze(0).to(device)

        if args.bev_mode == "lift_splat":
            with torch.no_grad():
                bev = model(images, intrinsics, cam2ego)
            bev_tensors.append(bev[0].cpu())
        else:
            bev_rgb = front_camera_ipm_bev(
                sample["images"][front_idx].to(device),
                sample["intrinsics"][front_idx].to(device),
                sample["cam2ego"][front_idx].to(device),
                bev_x_range=(0.0, 50.0),
                bev_y_range=(-25.0, 25.0),
                bev_res=0.5,
                ground_z=args.ground_z,
            )
            bev_frames.append(bev_rgb)

        print(f"{scenario_name} | Frame {i}")

    # Save last scenario
    if saved_scenarios < args.max_scenarios:
        if args.bev_mode == "lift_splat" and bev_tensors:
            save_gif(current_scenario, render_bev_frames(bev_tensors))
            saved_scenarios += 1
        if args.bev_mode == "front_ipm" and bev_frames:
            save_gif(current_scenario, bev_frames)
            saved_scenarios += 1

    print(f"Saved {saved_scenarios} scenario GIF(s).")

if __name__ == "__main__":
    main()
