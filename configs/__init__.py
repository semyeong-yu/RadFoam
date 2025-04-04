import configargparse
import os
from argparse import Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(
        self, parser: configargparse.ArgParser, name: str, fill_none=False
    ):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            t = type(value)
            value = value if not fill_none else None
            if t == bool:
                group.add_argument(
                    "--" + key, default=value, action="store_true"
                )
            elif t == list:
                group.add_argument(
                    "--" + key,
                    nargs="+",
                    type=type(value[0]),
                    default=value,
                    help=f"List of {type(value[0]).__name__}",
                )
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class PipelineParams(ParamGroup):

    def __init__(self, parser):
        self.iterations = 20_000
        self.densify_from = 500 # 2_000
        self.densify_until = 11_000
        self.densify_factor = 1.15
        self.white_background = True
        self.quantile_weight = 1e-4
        self.experiment_name = ""
        self.debug = False
        self.viewer = False
        self.align_weight = 1e-2 # 1e-1 @ deblurnerf defocuscupcake config
        super().__init__(parser, "Setting Pipeline parameters")


class ModelParams(ParamGroup):

    def __init__(self, parser):
        self.sh_degree = 3
        self.init_points = 131_072
        self.final_points = 2_097_152
        self.activation_scale = 1.0
        self.device = "cuda"
        self.num_pt = 5
        self.kernel_hwindow = 10
        self.kernel_random_hwindow = 0.15
        self.kernel_rand_embed = 2
        self.kernel_random_mode = "input"
        self.kernel_img_embed = 32
        self.kernel_spatial_embed = 2
        self.kernel_depth_embed = 0
        self.kernel_num_hidden = 4
        self.kernel_num_wide = 64
        self.kernel_shortcut = True
        self.kernel_pattern_init_radius = 0.1
        self.kernel_isglobal = False
        self.kernel_global_trans = True
        self.kernel_spatialvariant_trans = True
        self.tone_mapping_type = "gamma"
        self.kernel_start_iter = 120
        self.align_start_iter = 0
        self.align_end_iter = 18_000
        self.kernel_lr_init = 5e-5
        self.kernel_lr_final = 1e-6
        self.kernel_lr_decay = 250
        super().__init__(parser, "Setting Model parameters")


class OptimizationParams(ParamGroup):

    def __init__(self, parser):
        self.points_lr_init = 1e-4 #2e-4
        self.points_lr_final = 5e-6
        self.density_lr_init = 5e-2 #1e-1
        self.density_lr_final = 1e-2
        self.attributes_lr_init = 2.5e-3 #5e-3
        self.attributes_lr_final = 5e-4
        self.sh_factor = 0.1
        self.freeze_points = 18_000
        super().__init__(parser, "Setting Optimization parameters")


class DatasetParams(ParamGroup):

    def __init__(self, parser):
        self.dataset = "colmap"
        self.data_path = "deblur_dataset/real_defocus_blur"
        self.scene = "defocuscupcake"
        # self.data_path = "data/mipnerf360"
        # self.scene = "bicycle"
        self.patch_based = False
        self.downsample = [4, 1]
        self.downsample_iterations = [0, 5_000]
        # self.downsample = [4, 2, 1]
        # self.downsample_iterations = [0, 150, 500]
        super().__init__(parser, "Setting Dataset parameters")
