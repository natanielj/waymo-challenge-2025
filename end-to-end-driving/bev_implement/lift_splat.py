import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class MultiCamLiftSplatBEV(nn.Module):
    def __init__(
        self,
        depth_bins=32,
        depth_min=1.0,
        depth_max=50.0,
        bev_x_range=(0, 50),
        bev_y_range=(-25, 25),
        bev_res=0.5
    ):
        super().__init__()

        self.depth_bins = depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.x_min, self.x_max = bev_x_range
        self.y_min, self.y_max = bev_y_range
        self.res = bev_res

        self.bev_h = int((self.x_max - self.x_min) / self.res)
        self.bev_w = int((self.y_max - self.y_min) / self.res)

        # Shared CNN backbone
        resnet = models.resnet18(weights="IMAGENET1K_V1")

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.depth_head = nn.Conv2d(512, depth_bins, kernel_size=1)

        self.bev_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    
    # Pixel grid (feature resolution)
    
    def create_pixel_grid(self, H, W, device):
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        ones = torch.ones_like(xs)
        grid = torch.stack([xs, ys, ones], dim=-1)  # (H,W,3)
        return grid
    
    # Forward
    
    def forward(self, images, K, cam2ego):
        """
        images:   (B, N, 3, H, W)
        K:        (B, N, 3, 3)
        cam2ego:  (B, N, 4, 4)
        """

        B, N, C, H, W = images.shape
        device = images.device

        
        # Encode each camera
        
        images = images.view(B * N, C, H, W)
        features = self.encoder(images)  # (B*N,512,Hf,Wf)

        _, C_feat, Hf, Wf = features.shape

        depth_logits = self.depth_head(features)
        depth_prob = torch.softmax(depth_logits, dim=1)

        # reshape back
        features = features.view(B, N, C_feat, Hf, Wf)
        depth_prob = depth_prob.view(B, N, self.depth_bins, Hf, Wf)

        
        # Create frustum points
        
        depth_values = torch.linspace(
            self.depth_min,
            self.depth_max,
            self.depth_bins,
            device=device
        )

        grid = self.create_pixel_grid(Hf, Wf, device)   # (Hf,Wf,3)
        grid = grid.view(-1, 3).T                       # (3, Hf*Wf)

        # Inverse intrinsics
        K_inv = torch.inverse(K)   # (B,N,3,3)

        # Expand grid for batch + cameras
        grid = grid.unsqueeze(0).unsqueeze(0)           # (1,1,3,Hf*Wf)
        grid = grid.expand(B, N, -1, -1)                # (B,N,3,Hf*Wf)

        # Compute camera rays
        rays = torch.matmul(K_inv, grid)                # (B,N,3,Hf*Wf)

        # Depth bins

        depth_values = torch.linspace(
            self.depth_min,
            self.depth_max,
            self.depth_bins,
            device=device
        )

        # Expand for broadcasting
        depth_values = depth_values.view(1,1,self.depth_bins,1,1)

        # reshape rays to spatial
        rays = rays.view(B, N, 3, Hf, Wf)
        rays = rays.unsqueeze(2)   # (B,N,1,3,Hf,Wf)

        # Multiply by depth bins
        points_cam = rays * depth_values.unsqueeze(3)
        # (B,N,D,3,Hf,Wf)

        
        # cam → ego transform
        
        ones = torch.ones_like(points_cam[:,:, :, :1])
        points_cam_h = torch.cat([points_cam, ones], dim=3)

        # reshape for matmul
        points_cam_h = points_cam_h.permute(0,1,2,4,5,3)
        points_cam_h = points_cam_h.reshape(B, N, -1, 4)

        points_ego = torch.matmul(
            cam2ego.unsqueeze(2),
            points_cam_h.unsqueeze(-1)
        )

        points_ego = points_ego.squeeze(-1)[...,:3]
        points_ego = points_ego.view(B,N,self.depth_bins,Hf,Wf,3)

        
        # BEV projection
        
                
        x = points_ego[..., 0]
        y = points_ego[..., 1]

        # Compute grid indices
        x_idx = ((x - self.x_min) / self.res).long()
        y_idx = ((y - self.y_min) / self.res).long()

        valid = (
            (x_idx >= 0) & (x_idx < self.bev_h) &
            (y_idx >= 0) & (y_idx < self.bev_w)
        )

        # Expand features to match depth dimension
        feat = features.unsqueeze(2).expand(-1, -1, self.depth_bins, -1, -1, -1)
        feat = feat.permute(0,1,2,4,5,3)   # (B,N,D,Hf,Wf,C)

        # Flatten everything consistently
        x_idx = x_idx.reshape(B, -1)
        y_idx = y_idx.reshape(B, -1)
        valid = valid.reshape(B, -1)
        feat = feat.reshape(B, -1, feat.shape[-1])  # (B, total_points, C)

        bev = torch.zeros(B, feat.shape[-1], self.bev_h, self.bev_w, device=device)

        for b in range(B):
            xb = x_idx[b][valid[b]]
            yb = y_idx[b][valid[b]]
            fb = feat[b][valid[b]]

            bev[b].index_put_(
                (torch.arange(fb.shape[1], device=device).unsqueeze(1).expand(-1, fb.shape[0]),
                xb.unsqueeze(0).expand(fb.shape[1], -1),
                yb.unsqueeze(0).expand(fb.shape[1], -1)),
                fb.T,
                accumulate=True
            )

        bev_out = self.bev_head(bev)

        return bev_out