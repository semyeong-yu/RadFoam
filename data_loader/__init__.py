import os

import numpy as np
import einops
import torch

import radfoam

from .colmap import COLMAPDataset


dataset_dict = {
    "colmap": COLMAPDataset,
}


def get_up(c2ws):
    right = c2ws[:, :3, 0]
    down = c2ws[:, :3, 1]
    forward = c2ws[:, :3, 2]

    A = torch.einsum("bi,bj->bij", right, right).sum(dim=0)
    A += torch.einsum("bi,bj->bij", forward, forward).sum(dim=0) * 0.02

    l, V = torch.linalg.eig(A)

    min_idx = torch.argmin(l.real)
    global_up = V[:, min_idx].real
    global_up *= torch.einsum("bi,i->b", -down, global_up).sum().sign()

    return global_up


class DataHandler:
    def __init__(self, dataset_args, rays_per_batch, device="cuda"):
        self.args = dataset_args
        self.rays_per_batch = rays_per_batch
        self.device = torch.device(device)
        self.img_wh = None
        self.patch_size = 8

    def reload(self, split, downsample=None):
        data_dir = os.path.join(self.args.data_path, self.args.scene)
        dataset = dataset_dict[self.args.dataset]
        if downsample is not None:
            split_dataset = dataset(
                data_dir, split=split, downsample=downsample
            )
        else:
            split_dataset = dataset(data_dir, split=split)
        self.img_wh = split_dataset.img_wh
        self.fx = split_dataset.fx
        self.fy = split_dataset.fy
        self.c2ws = split_dataset.poses # (N, 3, 4)
        self.rays, self.rgbs = split_dataset.all_rays, split_dataset.all_rgbs

        self.viewer_up = get_up(self.c2ws) # global up-direction vector in world-coordinate
        self.viewer_pos = self.c2ws[0, :3, 3] # 첫 번째 image(camera)의 위치 (world-coordinate에서 기준점)
        self.viewer_forward = self.c2ws[0, :3, 2] # 첫 번째 image(camera)의 z-axis

        try:
            self.points3D = split_dataset.points3D
            self.points3D_colors = split_dataset.points3D_color
        except:
            self.points3D = None
            self.points3D_colors = None

        if split == "train":
            if self.args.patch_based:
                dw = self.img_wh[0] - (self.img_wh[0] % self.patch_size)
                dh = self.img_wh[1] - (self.img_wh[1] % self.patch_size)
                w_inds = np.linspace(0, self.img_wh[0] - 1, dw, dtype=int)
                h_inds = np.linspace(0, self.img_wh[1] - 1, dh, dtype=int)

                self.train_rays = self.rays[:, h_inds, :, :]
                self.train_rays = self.train_rays[:, :, w_inds, :]
                self.train_rgbs = self.rgbs[:, h_inds, :, :]
                self.train_rgbs = self.train_rgbs[:, :, w_inds, :]

                N, num_rayh, num_rayw, _ = self.train_rays.shape

                self.train_rays = einops.rearrange(
                    self.train_rays,
                    "n (x ph) (y pw) r -> (n x y) ph pw r", # (N * P_h * P_w, p, p, 6) = (total #patches, patch size, patch size, 6) where N : #images, P_h : #patches in H, P_w : #patches in W, p : patch size, 6 : ray origin and direction 
                    ph=self.patch_size,
                    pw=self.patch_size,
                )
                self.train_rgbs = einops.rearrange(
                    self.train_rgbs,
                    "n (x ph) (y pw) c -> (n x y) ph pw c",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )

                self.batch_size = self.rays_per_batch // (self.patch_size**2) # total #patches per batch

                self.batch_c2ws = self.c2ws.unsqueeze(1).unsqueeze(2).expand(N, num_rayh, num_rayw, 3, 4) # modified
                self.batch_c2ws = einops.rearrange(
                    self.batch_c2ws,
                    "n (x ph) (y pw) r t -> (n x y) ph pw r t", # (N * P_h * P_w, p, p, 3, 4) = (total #patches, patch size, patch size, 3, 4) where N : #images, P_h : #patches in H, P_w : #patches in W, p : patch size 
                    ph=self.patch_size,
                    pw=self.patch_size,
                )
            else:
                N, num_rayh, num_rayw, _ = self.rays.shape

                self.train_rays = einops.rearrange(
                    self.rays, "n h w r -> (n h w) r"
                )
                self.train_rgbs = einops.rearrange(
                    self.rgbs,
                     "n h w c -> (n h w) c"
                )

                if self.train_rays.shape[0] != self.train_rgbs.shape[0]:
                    print("Wrong Data Preprocessing - Need Debugging")

                self.batch_size = self.rays_per_batch # total #rays per batch

                self.batch_c2ws = self.c2ws.unsqueeze(1).unsqueeze(2).expand(N, num_rayh, num_rayw, 3, 4) # modified
                self.batch_c2ws = einops.rearrange(
                    self.batch_c2ws, "n h w r t -> (n h w) r t"
                )

    def get_iter(self):
        ray_batch_fetcher = radfoam.BatchFetcher(
            self.train_rays, self.batch_size, shuffle=True
        )
        rgb_batch_fetcher = radfoam.BatchFetcher(
            self.train_rgbs, self.batch_size, shuffle=True
        )
        pose_batch_fetcher = radfoam.BatchFetcher(
            self.batch_c2ws, self.batch_size, shuffle=True
        )
        
        # batch_idx = 0

        while True:
            ray_batch = ray_batch_fetcher.next()
            rgb_batch = rgb_batch_fetcher.next()
            pose_batch = pose_batch_fetcher.next()
            batch_idx = ray_batch_fetcher.get_last_indices()
            
            assert(torch.all(batch_idx < self.train_rays.shape[0]))
            assert(batch_idx.shape == (ray_batch.shape[0],) and batch_idx.ndim == 1)

            yield ray_batch, rgb_batch, pose_batch, batch_idx
            # B = ray_batch.shape[0]
            # batch_idx += B
