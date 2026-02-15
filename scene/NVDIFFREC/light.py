# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
from torch.nn import functional as F
import nvdiffrast.torch as dr

try:
    import util
    import renderutils as ru
except ImportError:
    from . import util
    from . import renderutils as ru
    from utils import tonemap
    from utils.image_utils import linear_to_srgb


######################################################################################
# Utility functions
######################################################################################

_DIR_CACHE = {}
def _cube_dirs(res, device):
    key = (res, device)
    if key not in _DIR_CACHE:
        gy, gx = torch.meshgrid(
            torch.linspace(-1+1/res, 1-1/res, res, device=device),
            torch.linspace(-1+1/res, 1-1/res, res, device=device),
            indexing='ij'
        )
        _DIR_CACHE[key] = [util.safe_normalize(util.cube_to_dir(s, gx, gy)) for s in range(6)]
    return _DIR_CACHE[key]

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        # return util.avg_pool_nhwc(cubemap, (2,2))  # slower
        return torch.nn.functional.avg_pool2d(cubemap.permute(0, 3, 1, 2), (2, 2)).permute(0, 2, 3, 1).contiguous()  # faster

    # @staticmethod
    # def backward(ctx, dout):
    #     res = dout.shape[1] * 2
    #     out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
    #     for s in range(6):
    #         gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
    #                                 torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
    #                                 )
    #                                 # indexing='ij')
    #         v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
    #         out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
    #     return out

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device=dout.device)
        dirs = _cube_dirs(res, dout.device)
        for s in range(6):
            out[s] = dr.texture(dout[None]*0.25, dirs[s][None].contiguous(),
                                filter_mode='linear', boundary_mode='cube')[0]
        return out

def _rodrigues_batch(axes, angles):
    # axes: [B,3] unit vectors, angles: [B,1]
    B = axes.shape[0]
    x, y, z = axes[:,0], axes[:,1], axes[:,2]
    zeros = torch.zeros(B, device=axes.device, dtype=axes.dtype)
    K = torch.stack([
        torch.stack([zeros, -z,    y   ], dim=-1),
        torch.stack([   z,  zeros, -x  ], dim=-1),
        torch.stack([  -y,   x,    zeros], dim=-1)
    ], dim=-2)                                    # [B,3,3]
    I = torch.eye(3, device=axes.device, dtype=axes.dtype).expand(B,3,3)
    s, c = torch.sin(angles), torch.cos(angles)   # [B,1]
    R = I + s[:,:,None]*K + (1.0 - c)[:,:,None]*(K @ K)   # [B,3,3]
    return R

def random_so3_batch(B, device, max_angle_deg=None, dtype=torch.float32):
    v = torch.randn(B,3, device=device, dtype=dtype)
    axes = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
    if max_angle_deg is None:
        angles = torch.rand(B,1, device=device, dtype=dtype) * (2*torch.pi)
    else:
        a = torch.deg2rad(torch.tensor(max_angle_deg, device=device, dtype=dtype))
        angles = (torch.rand(B,1, device=device, dtype=dtype) * 2*a) - a
    R = _rodrigues_batch(axes, angles)            # [B,3,3]
    M = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)
    M[:,:3,:3] = R                                 # zero translation; bottom row [0,0,0,1]
    return M                                       # [B,4,4]

def random_yaw_batch(B, device, max_yaw_deg=180, dtype=torch.float32):
    a = torch.deg2rad(torch.tensor(max_yaw_deg, device=device, dtype=dtype))
    yaw = (torch.rand(B,1, device=device, dtype=dtype) * 2*a) - a  # [-a,+a]
    c, s = torch.cos(yaw).squeeze(-1), torch.sin(yaw).squeeze(-1)
    M = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)
    M[:,0,0] = c; M[:,0,2] = s
    M[:,2,0] = -s; M[:,2,2] = c
    return M


def luminance(x, chw=False):
    if chw:
        return 0.2126*x[0] + 0.7152*x[1] + 0.0722*x[2]
    else:
        return 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]

def _solid_angle_weights(H, device, dtype):
    t = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=device, dtype=dtype)
    u, v = torch.meshgrid(t, t, indexing='ij')
    return 1.0 / torch.pow(1.0 + u*u + v*v, 1.5)  # Rory Driscoll weights

def env_power_mu(env_L0):
    # env_L0: (6,H,W,3)
    w = _solid_angle_weights(env_L0.shape[1], env_L0.device, env_L0.dtype)  # (H,W)
    lum = luminance(env_L0)  # (6,H,W)
    return (w.expand_as(lum)*lum).sum() / (w.sum()*6.0)

def tv2(face):  # (H,W,3)
    return (face[:-1,:,:]-face[1:,:,:]).abs().mean() + (face[:,:-1,:]-face[:,1:,:]).abs().mean()

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    def __init__(self, base, max_res=64, min_roughness=0.08, max_roughness=0.5, positive=False):
        super(EnvironmentLight, self).__init__()
        assert base.ndim == 4 and base.shape[0] == 6 and base.shape[1] == base.shape[2], "base must be (6, H, W, C) with square faces."
        self.mtx = None
        self.current_res = int(base.shape[1])
        self.max_res = int(max_res) if max_res is not None else self.current_res
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.positive = positive  # if True, apply softplus on base
        self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device=base.device)
        self.raw_base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)  # (6, 64, 64, 3)
        self.base = self.raw_base.clamp(min=0.0) if positive else self.raw_base
        self.register_parameter('raw_base', self.raw_base)
        self.log_exposure = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))  # starts at 1.0
        self.diffuse = None
        self.specular = None

    def xfm(self, mtx):
        self.mtx = mtx

    def maybe_rotate_env(self, p=0.5, mode='so3', max_angle_deg=None):
        if torch.rand(()) < p:
            mtx = random_so3_batch(1, device=torch.device('cuda'), max_angle_deg=max_angle_deg)
            self.xfm(mtx)     # shade3 will rotate R and N with this
        else:
            self.xfm(None)

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.raw_base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.max_roughness
                        , (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2)

    def build_mips(self, cutoff=0.99, update=True):
        if update:
            self.base = self.raw_base.clamp(min=1e-8) if self.positive else self.raw_base  # (6,64,64,3)
            self.specular = [self.base]  # [(6,64,64,3)]
            while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
                self.specular += [cubemap_mip.apply(self.specular[-1])]  # [..., [6,64,64,3], [6,32,32,3], [6,16,16,3]]

            self.diffuse = ru.diffuse_cubemap(self.specular[-1])  # (6,16,16,3)

            for idx in range(len(self.specular) - 1):
                roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
                self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff)
            self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
        else:
            self.diffuse = self.diffuse.detach()
            self.specular = [s.detach() for s in self.specular]

    def double_resolution(self):
        """
        Double per-face resolution (seam-aware resample), replace learnable Parameter,
        and rebuild mips. You MUST update your optimizer's param groups after calling this.
        """
        new_res = min(self.current_res * 2, self.max_res)
        if new_res == self.current_res:
            return False

        with torch.no_grad():
            up = self._resample_cubemap(self.raw_base, new_res)  # differentiable op not needed here
        # Replace Parameter (optimizer param group must be updated by the caller)
        self.raw_base = torch.nn.Parameter(up.detach().clone(), requires_grad=True)
        self.register_parameter('raw_base', self.raw_base)
        self.current_res = new_res

        # Rebuild mips after change
        self.build_mips()
        return True

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    @torch.no_grad()
    def _resample_cubemap(self, cubemap, res):
        """
        Seam-aware resample to (6, res, res, C) using cube boundary sampling.
        """
        assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2]
        device, dtype = cubemap.device, cubemap.dtype
        C = cubemap.shape[-1]
        out = torch.empty(6, res, res, C, device=device, dtype=dtype)

        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=device, dtype=torch.float32),
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=device, dtype=torch.float32),
            indexing='ij'
        )
        for s in range(6):
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(
                cubemap[None, ...], v[None, ...].contiguous(),
                filter_mode='linear', boundary_mode='cube'
            )[0]
        return out

    def regularize_env(self, mu_target=1.0, w_mu=1e-3, w_tv=5e-5, w_gray=1e-5):
        """
        Returns scalar regularizer for the environment + exposure.
        - Soft power prior on mean luminance at level-0 (no hard normalization)
        - TV prior (low-frequency bias)
        - Gray-world color prior
        - Exposure prior (keeps exposure near 1)
        """
        L0 = self.base                                  # (6,H,W,3)

        reg_mu = 0.0
        if w_mu > 0.0:
            mu = env_power_mu(L0)                             # scalar
            reg_mu   = w_mu * (torch.log(mu) - torch.log(torch.as_tensor(mu_target, device=mu.device)))**2

        # TV over all faces at L0 only (cheapest & effective)
        reg_tv = 0.0
        if w_tv > 0.0:
            for f in L0:
                reg_tv = reg_tv + tv2(f)
            reg_tv = w_tv * reg_tv

        reg_gray = 0.0
        if w_gray > 0.0:
            # Gray-world (very weak)
            H = L0.shape[1]; w = _solid_angle_weights(H, L0.device, L0.dtype)
            m = 0.0
            for f in L0:
                m = m + torch.stack([(w*f[...,c]).sum() for c in range(3)]) / w.sum()
            m = m / 6.0                                       # (3,)
            reg_gray = w_gray * ((m[0]-m[1])**2 + (m[1]-m[2])**2)

        return reg_mu + reg_tv + reg_gray

    def shade(self, viewdir, gb_normal, kd, kr, km, ks, disney_brdf=False, spec_gloss_brdf=True):
        wo = -viewdir  # (1,H,W,3)
        nrmvec = gb_normal  # (1,H,W,3)
        reflvec = util.safe_normalize(util.reflect(wo, nrmvec))
        NdotV = util.dot(wo, nrmvec).clamp(0.0, 1.0)  # (1,H,W,1)

        # clamp materials to avoid out-of-bounds values
        kd = kd.clamp(0.0, 1.0)
        kr = kr.clamp(0.0, 1.0)
        km = km.clamp(0.0, 1.0)
        ks = ks.clamp(0.0, 1.0)

        # Lookup FG term from lookup texture
        fg_uv = torch.cat((NdotV, kr), dim=-1).clamp(0,1)
        fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        # Ambient illumination lookup
        ambient = dr.texture(self.diffuse[None], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')

        # Roughness adjusted specular env lookup
        miplevel = self.get_mip(kr)
        spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

        # F0  = (1.0 - km) * 0.08 * ks + km * kd
        if spec_gloss_brdf and not disney_brdf:
            F0 = F0_d = ks  # (1,H,W,3)
            diff_col = kd #* (1.0 - luminance(F0)[..., None])  # (1,H,W,3)
        elif not spec_gloss_brdf and disney_brdf:
            F0 = F0_d = (1.0 - km) * 0.04 + km * kd  # (1,H,W,3)
            diff_col = kd * (1.0 - km)  # (1,H,W,3)
        elif spec_gloss_brdf and disney_brdf:
            F0 = (1.0 - km) * 0.04 + km * ks.detach()  # (1,H,W,3)
            diff_col = kd * (1.0 - km)
        else:
            raise RuntimeError("Either disney_brdf or spec_gloss_brdf must be True")

        diffuse_col = diff_col * ambient
        specular_col = (F0 * fg_lookup[...,0:1] + fg_lookup[...,1:2]) * spec
        shaded_col = diffuse_col + specular_col

        return shaded_col, {"diffuse": diffuse_col, "specular": specular_col,  "F0": F0, "direct_light": spec}


######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_map_hdr(fn, scale=1.0, res=[64, 64], tonemap=lambda x: x, rotate=False):
    # read envmap from HDR file
    latlong_img = util.load_image(fn)

    latlong_img = tonemap(latlong_img)  # apply tonemapping if needed

    # convert to tensor
    latlong_img = scale * torch.tensor(latlong_img, dtype=torch.float32, device='cuda').contiguous()

    # convert to cubemap-based light representation
    if rotate:
        cubemap = util.latlong_to_cubemap_orig(latlong_img, res)
    else:
        cubemap = util.latlong_to_cubemap(latlong_img, res)
    l = EnvironmentLight(cubemap)

    # generate cube mip maps
    l.build_mips()
    return l

def load_env_map(fn, scale=1.0, res=[64, 64], tonemap=lambda x: x, rotate=False):
    if any([fn.lower().endswith(x) for x in [".hdr", ".hdri", ".exr", ".tiff", ".pfm"]]):
        return _load_env_map_hdr(fn, scale, res, tonemap, rotate)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light, rotate=False):
    color = extract_env_map(light, [512, 1024], rotate).permute(1,2,0)
    util.save_image_raw(fn, color.detach().cpu().numpy())
    util.save_image(fn.replace('hdr', 'png'), color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_map_rnd(base_res, initial_res=64, scale=0.5, bias=0.25, min_roughness=0.08, max_roughness=0.5, positive=False):
    # create random cubemap with minimum resolution
    base = torch.ones(6, initial_res, initial_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    print(f"Initializing envmap to {base.shape}")
    env_map = EnvironmentLight(base, max_res=base_res, min_roughness=min_roughness, max_roughness=max_roughness, positive=positive)
    return env_map

def extract_env_map(light, resolution=[512, 1024], rotate=False, tonemapping=False, srgb=False):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if rotate:
        color = util.cubemap_to_latlong_orig(light.base, resolution)
    else:
        color = util.cubemap_to_latlong(light.base, resolution)
    if tonemapping:
        color = tonemap.gamma_tonemap(color)
    if srgb:
        color = linear_to_srgb(color)
    return color.permute(2,0,1)