import torch
from torch.utils.data import DataLoader

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

        TARGET_SIZE = (256, 704)  # (H, W) choose something reasonable

        for cam in frame.images:

            img = preprocess(cam.image)  # (3,H,W)

            orig_h, orig_w = img.shape[1:]

            img = F.interpolate(
                img.unsqueeze(0),
                size=TARGET_SIZE,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            images.append(img)

            calib = calib_dict[cam.name]

            fu, fv, cu, cv = calib.intrinsic[:4]

            scale_x = TARGET_SIZE[1] / orig_w
            scale_y = TARGET_SIZE[0] / orig_h

            K = torch.tensor([
                [fu * scale_x, 0.0, cu * scale_x],
                [0.0, fv * scale_y, cv * scale_y],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32)

            intrinsics.append(K)

            T = torch.tensor(calib.extrinsic, dtype=torch.float32)
            cam2ego.append(T)
        
        images = torch.stack(images)
        intrinsics = torch.stack(intrinsics)
        cam2ego = torch.stack(cam2ego)

        return {
            "images": images,
            "intrinsics": intrinsics,
            "cam2ego": cam2ego
        }

def main():

    dataset_path = "/home/nataniel/Desktop/work2/aiea/challenge_team/data/pytorch"
    device = "cuda"

    dataset = WaymoBEVDataset(dataset_path, split="training")

    model = MultiCamLiftSplatBEV().to(device)
    model.eval()

    bev_frames = []

    for i in range(50):   # first 50 frames for demo

        sample = dataset[i]
        print(type(sample))
        print(sample)
        images = sample["images"].unsqueeze(0).to(device)
        intrinsics = sample["intrinsics"].unsqueeze(0).to(device)
        cam2ego = sample["cam2ego"].unsqueeze(0).to(device)

        with torch.no_grad():
            bev = model(images, intrinsics, cam2ego)

        bev_img = bev_to_image(bev[0])
        bev_frames.append(np.array(bev_img))

        print(f"Processed frame {i}")

    imageio.mimsave("bev_output.gif", bev_frames, fps=5)

    print("Saved bev_output.gif")

if __name__ == "__main__":
    main()