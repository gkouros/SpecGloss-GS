#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for orig_key, orig_value in vars(self).items():
            shorthand = False
            key = orig_key
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(orig_value)
            # Keep original default for logic, but optionally pass None to argparse for cfg merging
            default_for_parser = None if fill_none else orig_value

            if t == bool:
                # Positive flag (enable)
                if shorthand:
                    group.add_argument("--" + key, "-" + key[0:1], dest=key, action="store_true",
                                       help=f"Enable {key}")
                else:
                    group.add_argument("--" + key, dest=key, action="store_true",
                                       help=f"Enable {key}")

                # Negative flag (disable). No short alias (avoids weird UX).
                group.add_argument("--no-" + key, dest=key, action="store_false",
                                   help=f"Disable {key}")

                # Respect either the real default or None (for cfg override path)
                group.set_defaults(**{key: default_for_parser})

            elif t == list:
                if shorthand:
                    group.add_argument("--" + key, "-" + key[0:1], default=default_for_parser, nargs="+")
                else:
                    group.add_argument("--" + key, default=default_for_parser, nargs="+")
            else:
                # Numbers/strings
                if shorthand:
                    group.add_argument("--" + key, "-" + key[0:1], default=default_for_parser, type=t)
                else:
                    group.add_argument("--" + key, default=default_for_parser, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.disney_brdf = False
        self.spec_gloss_brdf = True
        self.sh_degree = 3  # -1 for no SH, 0 for constant, 1 for linear, 2 for quadratic, etc.
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True

        self.brdf_envmap_res = 512
        self.brdf_envmap_scale = 0.1
        self.brdf_envmap_bias = 0.0
        self.min_roughness = 0.04
        self.max_roughness = 0.5
        self.clamp_roughness = False # if True, roughness is clamped to [min_roughness, max_roughness]
        self.softplus_env = False # Apply softplus on envmap base
        self.clamp_then_srgb = False  # If True, apply clamping before linear_to_srgb instead of the opposite
        self.default_roughness = 0.1  # 0.6 in GaussianShader
        self.default_metallic = 0.02  # default for dielectrics
        self.default_specular = 0.5
        self.roughness_bg_color = 0.0 # 0.0 for no background color, 1.0 for white background
        self.visibility_threshold = 0.95

        # flags for activating different features
        self.srgb = True
        self.use_residual = False

        self.unpremult_normals = True
        self.normalize_normals = False

        # 3DGS-DR solution for real scenes
        self.env_scope_center = [0.,0.,0.]
        self.env_scope_radius = 0.0

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

    def extract(self, args):
        g = super().extract(args)
        return g

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        MAX_ITERS = 100_000
        self.iterations = 50_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.feature_lr = 0.005
        MATERIAL_LR = 0.005
        self.base_color_lr = 1.5 * MATERIAL_LR
        self.roughness_lr = MATERIAL_LR
        self.metallic_lr = MATERIAL_LR
        self.specular_lr = MATERIAL_LR
        self.material_lr_decay = 0.1
        self.spec_score_lr = 0.1
        self.brdf_mlp_lr_init = 1e-2  # must be an order of magnitude lower than the LR of the materials
        self.brdf_mlp_lr_final = 1e-3
        self.brdf_mlp_lr_delay_mult = 0.01
        self.brdf_mlp_lr_max_steps = self.iterations
        self.fix_brdf_lr = False
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_charb = 0.0
        self.lambda_hdr = 0.0
        self.lambda_dist = 100.0  # 100 for RefGS, but 1000 for bounded and 100 for unbounded scenes for 2DGS
        self.lambda_normal = 0.05  # 0.05 for 2DGS
        self.lambda_normal_smooth = 0.0
        self.lambda_surf_normal_smooth = 0.0
        self.lambda_depth_smooth = 0.0
        self.lambda_normal_prior = 0.01  # 0.01 for bell
        self.lambda_normal_gradient = 0.0  # 0.05 for bell
        self.lambda_zero_one = 0.0  # 1e-2 seems to work better
        self.lambda_brdf_smooth = 0.0 # best for us 0.01  # 0.05 from GS-2DGS
        self.lambda_envmap_reg = 0.001  # 0.01 works best for bell scene, 0.001 for all scenes
        self.lambda_env_scope = 0.0  # loss weight for envscope-based roughness regularization of background
        self.lambda_env_scope_metallic = 0.0  # loss weight for envscope-based metallic regularization of background
        self.lambda_env_scope_specular = 0.0
        self.env_scope_specular_val = 0.0
        self.lambda_diffuse_prior = 0.05  # 0.05
        self.lambda_energy_conservation = 0.0
        self.lambda_neutral = 0.0
        self.use_ec_norm = False
        self.opacity_cull = 0.05  # 0.05 in 2DGS
        self.mask_loss_from_iter = 0
        self.mask_loss_until_iter = 7000
        self.normal_reg_from_iter = 0  # 7000 for 2DGS
        self.dist_loss_from_iter = 3000  # 3000 for 2DGS
        self.sparse_loss_from_iter = 0
        self.sparse_loss_until_iter = 0
        self.indirect_from_iter = 20_000
        self.residual_from_iter = MAX_ITERS
        self.normal_prior_from_iter = 0
        self.normal_prior_until_iter = MAX_ITERS
        self.diffuse_prior_from_iter = 0
        self.diffuse_prior_until_iter = 15_000
        self.brdf_smooth_from_iter = 0
        self.warmup_until_iter = 0
        self.envscope_roughness_threshold = 1.0
        self.brdf_envmap_upsample_interval = 15000  # the higher the faster and better
        self.no_base_color_from_iter = 0
        self.no_base_color_until_iter = 0
        self.no_roughness_from_iter = 0
        self.no_roughness_until_iter = 0
        self.no_metallic_from_iter = 0
        self.no_metallic_until_iter = 0
        self.no_specular_from_iter = 0
        self.no_specular_until_iter = 0
        self.clip_envmap = True  # clipping to [0, +inf)
        self.clip_envmap_max = False  # clipping to [0, 1]
        self.update_envmap_every = 1

        # mesh extraction params
        self.voxel_size = -1.0
        self.depth_trunc = -1.0
        self.sdf_trunc = -1.0
        self.mesh_res = 512
        self.num_cluster = 1
        self.mesh_extraction_interval = 2000

        # densification params
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.normal_prop_interval = 1000
        self.perturb_base_color = False
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
