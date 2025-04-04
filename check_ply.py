import os
import uuid
import yaml
import gc
import numpy as np
from PIL import Image
import configargparse
import argparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
import radfoam

from deblurnerf.NeRF import DSKnet
from deblurnerf.run_nerf_helpers import *

from datetime import datetime

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

import open3d as o3d
import numpy as np

def compute_pcd_difference(ply_path1, ply_path2):
    pcd1 = o3d.io.read_point_cloud(ply_path1)
    pcd2 = o3d.io.read_point_cloud(ply_path2)

    points1 = np.asarray(pcd1.points)  # (N, 3)
    points2 = np.asarray(pcd2.points)  # (N, 3)

    if points1.shape != points2.shape:
        print("Point clouds have different shapes! Can't compare directly.")
        return None

    distances = np.linalg.norm(points1 - points2, axis=1)

    # 변화량 출력
    print(f"Mean Position Change: {np.mean(distances):.6f}")
    print(f"Max Position Change: {np.max(distances):.6f}")
    print(f"Min Position Change: {np.min(distances):.6f}")

    return distances

def main(ply_path, i):
    if i >= 10:
        prev_ply_path = os.path.dirname(ply_path) + f"/{i-10}_pcd.ply"
        compute_pcd_difference(ply_path, prev_ply_path)
    else:
        return 0

if __name__ == "__main__":
    # 예시 : python check_ply.py ./output/defocuscoral@0319_1657/20_pcd.ply 20
    parser = argparse.ArgumentParser(description="Process a file path and an integer parameter.")
    parser.add_argument("path", type=str, help="Path to a file or directory")
    parser.add_argument("number", type=int, help="An integer parameter")

    args = parser.parse_args()
    main(args.path, args.number)