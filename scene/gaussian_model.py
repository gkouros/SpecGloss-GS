import torch
import numpy as np
import torch.nn
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, flip_align_view
import nvdiffrast.torch
from torch import nn
import math
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_scaling_rotation
from utils.point_utils import initialize_color
import open3d as o3d
from scene.NVDIFFREC import create_trainable_env_map_rnd, load_env_map, save_env_map
from .raytracer import RayTracer

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.rotation_activation = torch.nn.functional.normalize

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.color_activation = torch.sigmoid
        self.inverse_color_activation = inverse_sigmoid

        self.roughness_activation = torch.sigmoid
        self.inverse_roughness_activation = inverse_sigmoid

        self.metallic_activation = torch.sigmoid
        self.inverse_metallic_activation = inverse_sigmoid

        self.metallic_activation = torch.sigmoid
        self.inverse_metallic_activation = inverse_sigmoid

        self.specular_activation = torch.sigmoid
        self.inverse_specular_activation = inverse_sigmoid

    def __init__(self, config):
        self.active_sh_degree = 0
        self.disney_brdf = getattr(config, "disney_brdf", True)
        self.spec_gloss_brdf = getattr(config, "spec_gloss_brdf", False) if self.disney_brdf else True
        self.max_sh_degree = getattr(config, "sh_degree", -1)
        self.brdf_envmap_res = getattr(config, "brdf_envmap_res", 64)
        self.brdf_envmap_scale = getattr(config, "brdf_envmap_scale", 1.0)
        self.brdf_envmap_bias = getattr(config, "brdf_envmap_bias", 0.0)
        self.default_roughness = getattr(config, "default_roughness", 0.5)
        self.default_metallic = getattr(config, "default_metallic", 0.02)
        self.default_specular = getattr(config, "default_specular", 0.5)
        self.roughness_bias = getattr(config, "roughness_bias", 0.0)
        self.min_roughness = getattr(config, "min_roughness", 0.08)
        self.max_roughness = getattr(config, "max_roughness", 0.5)
        self.clamp_roughness = getattr(config, "clamp_roughness", False)  # if True, roughness is clamped to [min_roughness, max_roughness]
        self.softplus_env = getattr(config, "softplus_env", False)  # if True, apply softplus on envmap base
        self.clamp_then_srgb = getattr(config, "clamp_then_srgb", False)  # If True, apply clamping before linear_to_srgb instead of the opposite
        self.roughness_bg_color = getattr(config, "roughness_bg_color", 0.0)  # 0.0 for no background color, 1.0 for white background
        self.visibility_threshold = getattr(config, "visibility_threshold", 0.95)

        self.srgb = getattr(config, "srgb", False)

        self.use_residual = False

        self.unpremult_normals = getattr(config, "unpremult_normals", False)  # if True, the normals are unpremultiplied by opacity
        self.normalize_normals = getattr(config, "normalize_normals", False)  # if True, the normals are normalized after unpremultiplication
        self.sigmoid_light = getattr(config, "sigmoid_light", False)  # if True, the light is sigmoid activated

        self.env_scope_radius = getattr(config, "env_scope_radius", 0.0)
        self.env_scope_center = torch.tensor(list(map(float, getattr(config, "env_scope_center", [0.,0.,0.]))), dtype=torch.float, device="cuda")

        self._xyz = torch.empty(0)
        self._features_sh_dc = torch.empty(0)
        self._features_sh_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)
        self._specular = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.ray_tracer = None

        # setup envmap
        self.brdf_mlp = create_trainable_env_map_rnd(self.brdf_envmap_res, scale=self.brdf_envmap_scale, bias=self.brdf_envmap_bias,
                                                     min_roughness=self.min_roughness, max_roughness=self.max_roughness, positive=self.softplus_env)

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_sh_dc,
            self._features_sh_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._base_color,
            self._roughness,
            self._specular,
            self._metallic,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.brdf_mlp.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_sh_dc,
            self._features_sh_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._base_color,
            self._roughness,
            self._specular,
            self._metallic,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            brdf_mlp_dict,
            self.spatial_lr_scale
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.brdf_mlp.load_state_dict(brdf_mlp_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features_sh(self):
        return torch.cat((self._features_sh_dc,  self._features_sh_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_base_color(self):
        return self.color_activation(self._base_color)

    @property
    def get_roughness(self):
        roughness = self.roughness_activation(self._roughness + self.roughness_bias)
        if self.clamp_roughness:
            roughness = roughness * (self.max_roughness - self.min_roughness) + self.min_roughness  # keep it within range
        return roughness

    @property
    def get_specular(self):
        return self.specular_activation(self._specular)

    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print("Increased SH degree to: ", self.active_sh_degree)

    def get_surfel_minimum_axis(self, viewdir=None):
        scales = self.get_scaling
        rotations = self.get_rotation
        R = build_rotation(rotations)
        # Find index of minimum scale for each surfel
        sorted_idx = torch.argsort(scales, descending=False, dim=-1)  # Sort scales per surfel
        R_xy = R[:, :, :2]  # Take the first two columns (corresponding to 2D surfel axes)
        R_sorted = torch.gather(R_xy, dim=2, index=sorted_idx[:, None, :].expand(-1, 3, -1))
        min_axis = R_sorted[:, :, 0]  # First column corresponds to the shortest axis

        if viewdir is not None:
            min_axis, _ = flip_align_view(min_axis, viewdir)
        return min_axis

    @property
    def get_mask(self):
        if self.env_scope_radius > 0:
            mask = torch.sum((self.get_xyz - self.env_scope_center[None])**2, dim=-1) < (self.env_scope_radius ** 2)
        else:
            mask = torch.ones(len(self.get_xyz), device=self.get_xyz.device).bool()
        return mask[..., None]

    def activate_residual(self):
        self.use_residual = True

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # Gaussian params
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # material properties
        base_colors = self.inverse_color_activation(fused_color if fused_color.any() else initialize_color(fused_point_cloud))
        roughnesses = self.inverse_roughness_activation(torch.full((fused_point_cloud.shape[0], 1), self.default_roughness, dtype=torch.float, device="cuda"))
        metallics = self.inverse_metallic_activation(torch.full((fused_point_cloud.shape[0], 1), self.default_metallic, dtype=torch.float, device="cuda"))
        speculars = self.inverse_specular_activation(torch.full((fused_point_cloud.shape[0], 3), self.default_specular, dtype=torch.float, device="cuda"))

        # misc features
        features_sh = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # [N, 3*(D+1)^2]

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._base_color = nn.Parameter(base_colors.requires_grad_(True))
        self._roughness = nn.Parameter(roughnesses.requires_grad_(True))
        self._metallic = nn.Parameter(metallics.requires_grad_(True))
        self._specular = nn.Parameter(speculars.requires_grad_(True))

        # indirect color features
        self._features_sh_dc = nn.Parameter(features_sh[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_sh_rest = nn.Parameter(features_sh[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args):
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"}, # roughness
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"}, # roughness
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},  # metallic
            {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},  # specular dielectric
            {'params': self.brdf_mlp.parameters(), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"},
            {'params': [self._features_sh_dc], 'lr': training_args.feature_lr, "name": "f_sh_dc"},
            {'params': [self._features_sh_rest], 'lr': training_args.feature_lr / 20, "name": "f_sh_rest"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.brdf_mlp_lr_init, lr_final=training_args.brdf_mlp_lr_final,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult, max_steps=training_args.brdf_mlp_lr_max_steps)
        self.metallic_scheduler_args = get_expon_lr_func(lr_init=training_args.metallic_lr, lr_final=training_args.metallic_lr * training_args.material_lr_decay,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult, max_steps=training_args.brdf_mlp_lr_max_steps)
        self.base_color_scheduler_args = get_expon_lr_func(lr_init=training_args.base_color_lr, lr_final=training_args.base_color_lr * training_args.material_lr_decay,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult, max_steps=training_args.brdf_mlp_lr_max_steps)
        self.specular_scheduler_args = get_expon_lr_func(lr_init=training_args.specular_lr, lr_final=training_args.specular_lr * training_args.material_lr_decay,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult, max_steps=training_args.brdf_mlp_lr_max_steps)

    def set_residual_mode(self, training_args):
        self.activate_residual()
        l = [
            {'params': [self._features_sh_dc], 'lr': training_args.feature_lr, "name": "f_sh_dc"}, # SH DC
            {'params': [self._features_sh_rest], 'lr': training_args.feature_lr / 20, "name": "f_sh_rest"}, # SH rest
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = self.xyz_scheduler_args(iteration)
            if not self.fix_brdf_lr and param_group["name"] == "brdf_mlp":
                param_group['lr'] = self.brdf_mlp_scheduler_args(iteration)
            if not self.fix_brdf_lr and param_group["name"] == "metallic":
                param_group['lr'] = self.metallic_scheduler_args(iteration)
            if not self.fix_brdf_lr and param_group["name"] == "base_color":
                param_group['lr'] = self.base_color_scheduler_args(iteration)
            if not self.fix_brdf_lr and param_group["name"] == "specular":
                param_group['lr'] = self.specular_scheduler_args(iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('radii2D')
        for i in range(self._base_color.shape[1]):
            l.append('base_color_{}'.format(i))
        l.append('roughness')
        l.append('metallic')
        for i in range(self._specular.shape[1]):
            l.append('specular_{}'.format(i))
        for i in range(self._features_sh_dc.shape[1]*self._features_sh_dc.shape[2]):
            l.append('f_sh_dc_{}'.format(i))
        for i in range(self._features_sh_rest.shape[1]*self._features_sh_rest.shape[2]):
            l.append('f_sh_rest_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_sh_dc = self._features_sh_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_sh_rest = self._features_sh_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        base_color = self._base_color.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        specular = self._specular.detach().cpu().numpy()
        radii = self.max_radii2D.detach().cpu().numpy()[..., np.newaxis]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, opacity, scale, rotation, radii, base_color, roughness, metallic, specular, f_sh_dc, f_sh_rest), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz)
        pcd_o3d.colors = o3d.utility.Vector3dVector(base_color)

        return pcd_o3d

    def reset_opacity(self):
        opacity_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_metallic(self, exclusive_mask=None):
        metallic_new = self.inverse_metallic_activation(torch.max(self.get_metallic, torch.ones_like(self.get_metallic) * self.default_metallic))
        if exclusive_mask is not None:
            metallic_new[exclusive_mask] = self._metallic[exclusive_mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(metallic_new, "metallic")
        self._metallic = optimizable_tensors["metallic"]

    def reset_scale(self, exclusive_mask=None):
        scale_new = self.enlarge_reflective_scales(exclusive_mask=exclusive_mask)
        optimizable_tensors = self.replace_tensor_to_optimizer(scale_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def enlarge_reflective_scales(self, exclusive_mask=None):
        ENLARGE_SCALE = 1.5
        METALLIC_MASK_THR = 0.02
        ROUGH_MASK_THR = 0.1

        metallic_mask = self.get_metallic.flatten() < METALLIC_MASK_THR
        roughness_mask = self.get_roughness.flatten() > ROUGH_MASK_THR
        combined_mask = torch.logical_or(metallic_mask, roughness_mask)
        if exclusive_mask is not None:
            combined_mask = torch.logical_or(combined_mask, exclusive_mask)
        scales = self.get_scaling
        rmin_axis = (torch.ones_like(scales) * ENLARGE_SCALE)
        scale_new = self.scaling_inverse_activation(scales * rmin_axis)
        scale_new[combined_mask] = self._scaling[combined_mask]
        return scale_new

    def perturb_base_color(self, exclusive_mask=None):
        METALLIC_MASK_THR = 0.02
        PERTURB_RANGE = 0.4
        metallic_mask = self.get_metallic.flatten() > METALLIC_MASK_THR
        if exclusive_mask is not None:
            metallic_mask = torch.logical_or(metallic_mask, exclusive_mask)
        bc = self._base_color.clone()
        new_bc = bc + (torch.rand_like(bc)*PERTURB_RANGE*2-PERTURB_RANGE)
        new_bc[metallic_mask] = bc[metallic_mask]
        optimizable_tensors = self.replace_tensor_to_optimizer(new_bc, "base_color")
        self._base_color = optimizable_tensors["base_color"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]
        raddii = np.asarray(plydata.elements[0]["radii2D"])

        self.active_sh_degree = self.max_sh_degree

        # handles both 1Ch and 3Ch specular
        specular_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("specular_")]
        specular_names = sorted(specular_names, key = lambda x: int(x.split('_')[-1]))
        specular = np.zeros((xyz.shape[0], len(specular_names)))
        for idx, attr_name in enumerate(specular_names):
            specular[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_sh_dc = np.zeros((xyz.shape[0], 3, 1 * (self.max_sh_degree >= 0)))
        if self.max_sh_degree >= 0:
            features_sh_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_sh_dc_0"])
            features_sh_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_sh_dc_1"])
            features_sh_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_sh_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_sh_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert (len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3) if self.max_sh_degree >= 0 else len(extra_f_names) == 0
        features_sh_rest = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_sh_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_sh_rest = features_sh_rest.reshape((features_sh_rest.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        base_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_color_")]
        base_color_names = sorted(base_color_names, key = lambda x: int(x.split('_')[-1]))
        base_color = np.zeros((xyz.shape[0], len(base_color_names)))
        for idx, attr_name in enumerate(base_color_names):
            base_color[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_sh_dc = nn.Parameter(torch.tensor(features_sh_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_sh_rest = nn.Parameter(torch.tensor(features_sh_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacity, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._base_color = nn.Parameter(torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._specular = nn.Parameter(torch.tensor(specular, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.tensor(raddii, dtype=torch.float, device="cuda")

    def save_mlp_checkpoints(self, path):
        save_env_map(os.path.join(path, "brdf_mlp.hdr"), self.brdf_mlp)  # save in HDR and PNG formats
        save_env_map(os.path.join(path, "brdf_mlp2.hdr"), self.brdf_mlp, rotate=True)  # save in HDR and PNG formats with rotated configuration
        torch.save(self.brdf_mlp.state_dict(), os.path.join(path, "brdf_mlp.pth"))  # save state in pth format
        # save env gain param to file

    def load_mlp_checkpoints(self, path, hdr_mode=False):
        if hdr_mode:
            self.load_env_map(path)
        else:
            chkpt = torch.load(os.path.join(path, "brdf_mlp.pth"))
            if int(chkpt["raw_base"].shape[1]) != self.brdf_mlp.current_res:
                self.brdf_mlp = create_trainable_env_map_rnd(self.brdf_envmap_res, initial_res=chkpt["raw_base"].shape[1],
                                                             scale=self.brdf_envmap_scale, bias=self.brdf_envmap_bias,
                                                             min_roughness=self.min_roughness, max_roughness=self.max_roughness)
            self.brdf_mlp.load_state_dict(chkpt)

    def load_env_map(self, path, tonemap=lambda x: x, rotate=False):
        if not any([path.lower().endswith(x) for x in [".hdr", ".hdri", ".exr", ".tiff", ".pfm", ".jpg", ".png"]]):
            path = os.path.join(path, "brdf_mlp.hdr")
        self.brdf_mlp = load_env_map(path, scale=1.0, res=[self.brdf_envmap_res]*2, tonemap=tonemap, rotate=rotate)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_sh_dc = optimizable_tensors["f_sh_dc"]
        self._features_sh_rest = optimizable_tensors["f_sh_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._specular = optimizable_tensors["specular"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_sh_dc, new_features_sh_rest, new_opacity, new_scaling, new_rotation, new_base_color, new_roughness, new_metallic, new_specular):
        d = {
            "xyz": new_xyz,
            "f_sh_dc": new_features_sh_dc,
            "f_sh_rest": new_features_sh_rest,
            "opacity": new_opacity,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "base_color" : new_base_color,
            "roughness" : new_roughness,
            "metallic" : new_metallic,
            "specular" : new_specular,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_sh_dc = optimizable_tensors["f_sh_dc"]
        self._features_sh_rest = optimizable_tensors["f_sh_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._base_color = optimizable_tensors["base_color"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._specular = optimizable_tensors["specular"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_sh_dc = self._features_sh_dc[selected_pts_mask].repeat(N,1,1)
        new_features_sh_rest = self._features_sh_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_base_color = self._base_color[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N,1)
        new_specular = self._specular[selected_pts_mask].repeat(N,1)

        self.densification_postfix(
            new_xyz=new_xyz,
            new_features_sh_dc=new_features_sh_dc,
            new_features_sh_rest=new_features_sh_rest,
            new_opacity=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            new_base_color=new_base_color,
            new_roughness=new_roughness,
            new_metallic=new_metallic,
            new_specular=new_specular,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if torch.sum(selected_pts_mask) == 0:
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_features_sh_dc = self._features_sh_dc[selected_pts_mask]
        new_features_sh_rest = self._features_sh_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_base_color = self._base_color[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_specular = self._specular[selected_pts_mask]

        self.densification_postfix(
            new_xyz=new_xyz,
            new_features_sh_dc=new_features_sh_dc,
            new_features_sh_rest=new_features_sh_rest,
            new_opacity=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            new_base_color=new_base_color,
            new_roughness=new_roughness,
            new_metallic=new_metallic,
            new_specular=new_specular,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def update_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        self.ray_tracer = RayTracer(vertices, faces)

    def load_mesh_from_ply(self, model_path, iteration):
        ply_path = os.path.join(model_path, f'{iteration:06d}.ply')
        if not os.path.exists(ply_path):
            print(f"Mesh file not found at: {ply_path}")
            return
        mesh = o3d.io.read_triangle_mesh(ply_path)
        self.update_mesh(mesh)
        print("Loaded mesh from:", ply_path)
