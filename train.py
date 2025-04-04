import os
import uuid
import yaml
import gc
import numpy as np
from PIL import Image
import configargparse
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
import traceback


def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)
    # Setting up output directory
    if not pipeline_args.debug:
        # if len(pipeline_args.experiment_name) == 0:
        #     unique_str = str(uuid.uuid4())[:8]
        #     experiment_name = f"{dataset_args.scene}@{unique_str}"
        # else:
        #     experiment_name = pipeline_args.experiment_name
        experiment_name = f"{dataset_args.scene}@" + datetime.now().strftime("%m%d_%H%M")
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)
        os.makedirs(f"{out_dir}/ckpt", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataset
    iter2downsample = dict(
        zip(
            dataset_args.downsample_iterations,
            dataset_args.downsample,
        )
    )
    train_data_handler = DataHandler(
        dataset_args, rays_per_batch=10_000, device=device # rays_per_batch=1_000_000
    )
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )
    # test_ray_batch_fetcher = radfoam.BatchFetcher(
    #     test_data_handler.rays, batch_size=1, shuffle=False
    # )
    # test_rgb_batch_fetcher = radfoam.BatchFetcher(
    #     test_data_handler.rgbs, batch_size=1, shuffle=False
    # )

    # Define viewer settings
    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
    }

    # Setting up pipeline
    # rgb_loss = nn.SmoothL1Loss(reduction="none")
    rgb_loss = lambda x, y: torch.mean((x - y) ** 2) # modified

    # Setting up model
    model = RadFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors,
    )

    N, H, W, _ = train_data_handler.rays.shape
    model_dsk = DSKnet(N, H, W, model_args.num_pt, model_args.kernel_hwindow, 
                       random_hwindow=model_args.kernel_random_hwindow, 
                       in_embed=model_args.kernel_rand_embed, 
                       random_mode=model_args.kernel_random_mode, 
                       img_embed=model_args.kernel_img_embed,
                       spatial_embed=model_args.kernel_spatial_embed,
                       depth_embed=model_args.kernel_depth_embed,
                       num_hidden=model_args.kernel_num_hidden,
                       num_wide=model_args.kernel_num_wide,
                       short_cut=model_args.kernel_shortcut,
                       pattern_init_radius=model_args.kernel_pattern_init_radius,
                       isglobal=model_args.kernel_isglobal,
                       optim_trans=model_args.kernel_global_trans,
                       optim_spatialvariant_trans=model_args.kernel_spatialvariant_trans)

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
        init_num_points=model.primal_points.shape[0],
    )

    optimizer_dsk = torch.optim.Adam(model_dsk.parameters(), lr=model_args.kernel_lr_init, betas=(0.9, 0.999), weight_decay=1e-5)
    # scheduler_dsk = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dsk, T_max=pipeline_args.iterations, eta_min=1e-6)
    
    def lr_lambda(step):
        warmup_step = pipeline_args.densify_from
        total_step = pipeline_args.iterations
        init_lr = model_args.kernel_lr_init
        final_lr = model_args.kernel_lr_final
        # scale = 1.0
        # if curr_num_points > init_num_points:
        #     scale = init_num_points / curr_num_points
        if step < warmup_step:
            return float(step) / float(max(1, warmup_step)) # return scale * float(step) / float(max(1, warmup_step))                
        else:
            progress = (step - warmup_step) / float(max(1, total_step - warmup_step))
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return final_lr / init_lr + (1 - final_lr / init_lr) * cosine_decay # return scale * final_lr / init_lr + (1 - final_lr / init_lr) * cosine_decay

    scheduler_dsk = torch.optim.lr_scheduler.LambdaLR(optimizer_dsk, lr_lambda=lr_lambda)

    tonemapping = ToneMapping(model_args.tone_mapping_type)

    def test_render(
        test_data_handler, ray_batch_fetcher, rgb_batch_fetcher, debug=False, step=-1
    ):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        psnr_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]
                output, _, _, _, _ = model(ray_batch, start_points[i])

                opacity = output[..., -1:]
                if pipeline_args.white_background:
                    rgb_output = output[..., :3] + (1 - opacity)
                else:
                    rgb_output = output[..., :3]

                rgb_output = (rgb_output - rgb_output.mean(dim=0, keepdim=True)) / (rgb_output.std(dim=0, keepdim=True) + 1e-8)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean()
                psnr_list.append(img_psnr)
                torch.cuda.synchronize()

                if not debug:
                    if step == -1 or step % 500 == 0:
                        error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                        rgb_output = np.uint8(rgb_output.cpu() * 255)
                        rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                        im = Image.fromarray(
                            np.concatenate([rgb_output, rgb_batch, error], axis=1)
                        )
                        im.save(
                            f"{out_dir}/test/rgb_{step}_{i}_psnr_{img_psnr:.3f}.png"
                        )

        average_psnr = sum(psnr_list) / len(psnr_list)
        if not debug:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average PSNR: {average_psnr}")
            f.close()

        return average_psnr

    def train_loop(viewer):
        print("Training")

        torch.cuda.synchronize()

        data_iterator = train_data_handler.get_iter()
        ray_batch, rgb_batch, pose_batch, batch_idx = next(data_iterator) # stop at "yield" and continue from "yield"
        if torch.isnan(ray_batch).any():
            print("rat_batch NaN error!!")
            return 0
        if torch.isnan(rgb_batch).any():
            print("rgb_batch NaN error!!")
            return 0
        if torch.isnan(pose_batch).any():
            print("pose_batch NaN error!!")
            return 0
        
        N, H, W, _ = train_data_handler.rays.shape
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # (W, H)
        xs = np.tile((xs[None, ...] + 0.5), [N, 1, 1]) # (N, W, H)
        ys = np.tile((ys[None, ...] + 0.5), [N, 1, 1])
        rays_x = np.transpose(xs, (0, 2, 1))[..., None] # (N, H, W, 1)
        rays_y = np.transpose(ys, (0, 2, 1))[..., None] # (N, H, W, 1)
        ray_spatial = np.concatenate([rays_x, rays_y], axis=3) # (N, H, W, 2)
        ray_spatial = torch.from_numpy(ray_spatial).reshape(N*H*W, 2) # (N*H*W, 2)
        
        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        with tqdm.trange(pipeline_args.iterations) as train:
            torch.autograd.set_detect_anomaly(True) # for debugging # modified
            for i in train:
                # if i >= 2400:
                #     torch.autograd.set_detect_anomaly(True) # stop for debugging # modified
                
                if viewer is not None:
                    model.update_viewer(viewer)
                    viewer.step(i)

                if i in iter2downsample and i:
                    downsample = iter2downsample[i]
                    train_data_handler.reload(
                        split="train", downsample=downsample
                    )
                    data_iterator = train_data_handler.get_iter()
                    ray_batch, rgb_batch, pose_batch, batch_idx = next(data_iterator)
                    if torch.isnan(ray_batch).any():
                        print("rat_batch NaN error!!")
                        return 0
                    if torch.isnan(rgb_batch).any():
                        print("rgb_batch NaN error!!")
                        return 0
                    if torch.isnan(pose_batch).any():
                        print("pose_batch NaN error!!")
                        return 0
                    # ray_batch: (B, 6)
                    # pose_batch: (B, 3, 4)

                    N, H, W, _ = train_data_handler.rays.shape
                    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # (W, H)
                    xs = np.tile((xs[None, ...] + 0.5), [N, 1, 1]) # (N, W, H)
                    ys = np.tile((ys[None, ...] + 0.5), [N, 1, 1])
                    rays_x = np.transpose(xs, (0, 2, 1))[..., None] # (N, H, W, 1)
                    rays_y = np.transpose(ys, (0, 2, 1))[..., None] # (N, H, W, 1)
                    ray_spatial = np.concatenate([rays_x, rays_y], axis=3) # (N, H, W, 2)
                    ray_spatial = torch.from_numpy(ray_spatial).reshape(N*H*W, 2) # (N*H*W, 2)

                N, H, W, _ = train_data_handler.rays.shape
                focal_x = train_data_handler.fx
                focal_y = train_data_handler.fy
                K = np.array([
                    [focal_x, 0, 0.5 * W],
                    [0, focal_y, 0.5 * H],
                    [0, 0, 1]
                ])

                if i >= model_args.kernel_start_iter:
                    ray_info = ray_spatial[batch_idx] # (B, 2)

                    ray_batch, weight, align_loss = model_dsk(H, W, K, pose_batch[:, :3, :4], ray_info, batch_idx) # (B, M, 3, 2), (B, M), float
                    B, M, _, _ = ray_batch.shape
                    ray_batch = ray_batch.reshape(B*M, 6) # (B*M, 6)

                    depth_quantiles = (
                        torch.rand(*ray_batch.shape[:-1], 2, device=device)
                        .sort(dim=-1, descending=True)
                        .values
                    ) # (B*M, 2)

                    rgba_output, depth, _, _, _ = model(
                        ray_batch,
                        depth_quantiles=depth_quantiles,
                    )
                    rgba_output = rgba_output.reshape(B, M, 4)

                    # White background
                    rgb_pts = rgba_output[..., :3] # (B, M, 3)
                    opacity = rgba_output[..., -1:] # (B, M, 1)
                    
                    if pipeline_args.white_background:
                        rgb_pts = rgb_pts + (1 - opacity.expand(B, M, 3)) # (B, M, 3)
                    else:
                        rgb_pts = rgb_pts # (B, M, 3)

                    rgb_output = torch.sum(rgb_pts * weight[..., None], dim=1) # (B, 3)
                    rgb_output = tonemapping(rgb_output)

                    color_loss = rgb_loss(rgb_batch[:rgba_output.shape[0]], rgb_output)
                    opacity_loss = ((1 - opacity) ** 2).mean()

                    valid_depth_mask = (depth > 0).all(dim=-1)
                    quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                    quant_loss = (quant_loss * valid_depth_mask).mean()
                    w_depth = pipeline_args.quantile_weight * min(
                        2 * i / pipeline_args.iterations, 1
                    )
                else:
                    B, _ = ray_batch.shape # (B, 6)

                    depth_quantiles = (
                        torch.rand(*ray_batch.shape[:-1], 2, device=device)
                        .sort(dim=-1, descending=True)
                        .values
                    ) # (B, 2)

                    rgba_output, depth, _, _, _ = model(
                        ray_batch,
                        depth_quantiles=depth_quantiles,
                    )
                    rgba_output = rgba_output.reshape(B, 4)

                    # White background
                    rgb_pts = rgba_output[..., :3] # (B, 3)
                    opacity = rgba_output[..., -1:] # (B, 1)
                    
                    if pipeline_args.white_background:
                        rgb_output = rgb_pts + (1 - opacity.expand(B, 3)) # (B, 3)
                    else:
                        rgb_output = rgb_pts # (B, 3)

                    rgb_output = tonemapping(rgb_output)

                    color_loss = rgb_loss(rgb_batch[:rgba_output.shape[0]], rgb_output)
                    opacity_loss = ((1 - opacity) ** 2).mean()

                    valid_depth_mask = (depth > 0).all(dim=-1)
                    quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                    quant_loss = (quant_loss * valid_depth_mask).mean()
                    w_depth = pipeline_args.quantile_weight * min(
                        2 * i / pipeline_args.iterations, 1
                    )

                    align_loss = torch.tensor(0.0, device=color_loss.device)
                
                if torch.isnan(align_loss):
                    align_loss = torch.tensor(0.0, device=color_loss.device) # modified
                
                if i >= model_args.align_start_iter and i < model_args.align_end_iter:
                    loss = color_loss.mean() + 0.5 * opacity_loss + w_depth * quant_loss + pipeline_args.align_weight * align_loss
                else:
                    loss = color_loss.mean() + 0.5 * opacity_loss + w_depth * quant_loss

                model.optimizer.zero_grad(set_to_none=True)
                optimizer_dsk.zero_grad()
                # Hide latency of data loading behind the backward pass
                event = torch.cuda.Event()
                event.record()
                try:
                    loss.backward()
                except RuntimeError as e:
                    with open(f"{out_dir}/error.txt", "w") as f:
                        f.write(f"Error at iter. {i}: {e}\n")
                        traceback.print_exc(file=f)
                        f.close()
                total_norm = torch.nn.utils.clip_grad_norm_(model_dsk.parameters(), max_norm=1.0)
                event.synchronize()

                ray_batch, rgb_batch, pose_batch, batch_idx = next(data_iterator)
                if ray_batch.shape[0] == 0: # if B == 0:
                    train_data_handler.reload(
                        split="train", downsample=downsample
                    )
                    data_iterator = train_data_handler.get_iter()
                    ray_batch, rgb_batch, pose_batch, batch_idx = next(data_iterator)
                    if torch.isnan(ray_batch).any():
                        print("rat_batch NaN error!!")
                        return 0
                    if torch.isnan(rgb_batch).any():
                        print("rgb_batch NaN error!!")
                        return 0
                    if torch.isnan(pose_batch).any():
                        print("pose_batch NaN error!!")
                        return 0

                    N, H, W, _ = train_data_handler.rays.shape
                    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # (W, H)
                    xs = np.tile((xs[None, ...] + 0.5), [N, 1, 1]) # (N, W, H)
                    ys = np.tile((ys[None, ...] + 0.5), [N, 1, 1])
                    rays_x = np.transpose(xs, (0, 2, 1))[..., None] # (N, H, W, 1)
                    rays_y = np.transpose(ys, (0, 2, 1))[..., None] # (N, H, W, 1)
                    ray_spatial = np.concatenate([rays_x, rays_y], axis=3) # (N, H, W, 2)
                    ray_spatial = torch.from_numpy(ray_spatial).reshape(N*H*W, 2) # (N*H*W, 2)

                # radiant foam model
                model.optimizer.step()
                model.update_learning_rate(i, model.primal_points.shape[0])

                # deblur MLP module
                optimizer_dsk.step()
                scheduler_dsk.step()
                # decay_rate = 0.1
                # decay_steps = model_args.kernel_lr_decay * 100 # 25000
                # new_lrate = model_args.kernel_lr * (decay_rate ** (pipeline_args.iterations / decay_steps))
                # for param_group in optimizer_dsk.param_groups:
                #     param_group['lr'] = new_lrate

                train.set_postfix(color_loss=f"{color_loss.mean().item():.5f}")

                if not pipeline_args.debug:
                    writer.add_scalar("train/rgb_loss", color_loss.mean().item(), i)
                    writer.add_scalar("train/opacity_loss", 0.5* opacity_loss.item(), i)
                    writer.add_scalar("train/quant_loss", w_depth * quant_loss.item(), i)
                    writer.add_scalar("train/align_loss", pipeline_args.align_weight * align_loss.item(), i)
                    writer.add_scalar("train/gradient_norm", total_norm.item(), i)
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)

                    test_data_handler.reload(
                        split="test", downsample=min(dataset_args.downsample)
                    )
                    test_ray_batch_fetcher_local = radfoam.BatchFetcher(
                        test_data_handler.rays, batch_size=1, shuffle=False
                    )
                    test_rgb_batch_fetcher_local = radfoam.BatchFetcher(
                        test_data_handler.rgbs, batch_size=1, shuffle=False
                    )

                    test_psnr = test_render(
                        test_data_handler,
                        test_ray_batch_fetcher_local,
                        test_rgb_batch_fetcher_local,
                        False,
                        i
                    )
                    writer.add_scalar("test/psnr", test_psnr, i)

                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i, model.primal_points.shape[0]), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i, model.primal_points.shape[0]), i
                    )
                    writer.add_scalar(
                        "lr/attr_lr", model.attr_dc_scheduler_args(i, model.primal_points.shape[0]), i
                    )
                    writer.add_scalar(
                        "lr/deblur_dsk_lr", optimizer_dsk.param_groups[0]['lr'], i
                    )

                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0

                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (
                    iters_since_densification == next_densification_after
                    and model.primal_points.shape[0]
                    < 0.9 * model.num_final_points
                ):
                # if True:
                    point_error, point_contribution = model.collect_error_map(
                        train_data_handler, pipeline_args.white_background
                    )
                    model.prune_and_densify(
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                    )

                    model.update_triangulation(incremental=False)
                    triangulation_update_period = 1
                    gc.collect()

                    # Linear growth
                    iters_since_densification = 0
                    next_densification_after = int(
                        (
                            (pipeline_args.densify_factor - 1)
                            * model.primal_points.shape[0]
                            * (
                                pipeline_args.densify_until
                                - pipeline_args.densify_from
                            )
                        )
                        / (model.num_final_points - model.num_init_points)
                    )
                    next_densification_after = max(
                        next_densification_after, 100
                    )

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

                if i % 500 == 0:
                    model.save_ply(f"{out_dir}/ckpt/{i}_pcd.ply", i, pipeline_args)
                    model.save_pt(f"{out_dir}/ckpt/{i}_model.pt")

        # model.save_ply(f"{out_dir}/scene.ply")
        # model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    if pipeline_args.viewer:
        model.show(
            train_loop, iterations=pipeline_args.iterations, **viewer_options
        )
    else:
        train_loop(viewer=None)
    if not pipeline_args.debug:
        writer.close()

    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )
    test_ray_batch_fetcher_local = radfoam.BatchFetcher(
        test_data_handler.rays, batch_size=1, shuffle=False
    )
    test_rgb_batch_fetcher_local = radfoam.BatchFetcher(
        test_data_handler.rgbs, batch_size=1, shuffle=False
    )

    test_render(
        test_data_handler,
        test_ray_batch_fetcher_local,
        test_rgb_batch_fetcher_local,
        pipeline_args.debug
    )


def main():
    parser = configargparse.ArgParser(
        default_config_files=["configs/mipnerf360_indoor.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
