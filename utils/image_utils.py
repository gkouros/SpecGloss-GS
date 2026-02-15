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

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


pca = PCA(n_components=3)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)
    return magnitude

def sobel_xy(img):  # simple H1 gradient; img: (B,3,H,W) linear
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=img.device, dtype=img.dtype)/8.0
    ky = kx.t()
    kx = kx.view(1,1,3,3); ky = ky.view(1,1,3,3)
    gx = torch.nn.functional.conv2d(img, kx.repeat(3,1,1,1), padding=1, groups=3)
    gy = torch.nn.functional.conv2d(img, ky.repeat(3,1,1,1), padding=1, groups=3)
    return gx, gy

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def pca_transform(x, return_tensor=False):
    C, H, W = x.shape
    x_pca = pca.fit_transform(x.reshape(C, -1).permute(1,0).cpu().numpy())  # (HxW,3)
    x_pca = (x_pca - x_pca.min()) / (x_pca.max() - x_pca.min())
    x_pca = x_pca.reshape(H,W,3)  # (H,W,3)
    if return_tensor:
        x_pca = torch.tensor(x_pca).permute(2,0,1).to(x.device)
    return x_pca


def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)

    return net_image

def _backend(x):
    """Return (mod, where, power, finfo_eps) for NumPy or Torch."""
    if isinstance(x, torch.Tensor):
        return torch, torch.where, torch.pow, torch.finfo(x.dtype).eps  # machine epsilon for that dtype
    else:
        return np, np.where, np.power, np.finfo(x.dtype).eps  # NumPy ndarray or scalar

# sRGB ↔ linear transfer                              IEC 61966-2-1
_BREAK_FWD  = 0.0031308   # linear-to-sRGB  breakpoint (≈12.92⁻¹ * 0.04045)
_BREAK_INV  = 0.04045     # sRGB-to-linear breakpoint (IEC spec)
_SCALE_FWD  = 12.92       # slope in the linear segment (forward)
_SCALE_INV  = 1.0 / _SCALE_FWD
_A          = 1.055       # scale for power segment
_GAMMA      = 2.4         # exponent for power segment

def linear_to_srgb(linear, eps=None):
    """
    Convert linear-light RGB → sRGB.
    Accepts NumPy ndarray or Torch tensor, in [0,1].
    """
    mod, where, power, eps_default = _backend(linear)
    if eps is None:
        eps = eps_default
    # Two branches: linear segment vs power-law segment
    srgb = where(
        linear <= _BREAK_FWD,
        _SCALE_FWD * linear,
        _A * power(mod.clip(linear, eps, None), 1.0 / _GAMMA) - (_A - 1.0),
    )
    return srgb

def srgb_to_linear(srgb, eps=None):
    """
    Convert sRGB → linear-light RGB.
    Accepts NumPy ndarray or Torch tensor, in [0,1].
    """
    mod, where, power, eps_default = _backend(srgb)
    if eps is None:
        eps = eps_default
    linear = where(
        srgb <= _BREAK_INV,
        _SCALE_INV * srgb,
        power(mod.clip((srgb + (_A - 1.0)), eps, None) / _A, _GAMMA),
    )
    return linear


# -------- sRGB OETF (display gamma), channel-first --------
def srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    # x: [C,H,W] linear in sRGB primaries
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(x.clamp_min(0.0), 1.0/2.4) - 0.055
    )

# -------- Luminance from linear sRGB, channel-first --------
def luminance(rgb_lin: torch.Tensor) -> torch.Tensor:
    # rgb_lin: [3,H,W]
    r, g, b = rgb_lin[0:1], rgb_lin[1:2], rgb_lin[2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b  # [1,H,W]

# -------- Reinhard with white point (returns scale L'/L), channel-first --------
def reinhard_whitepoint_scale(L: torch.Tensor, log_Lw: torch.Tensor) -> torch.Tensor:
    # L: [1,H,W]  (luminance), log_Lw: scalar Parameter
    eps = 1e-8
    Lw = torch.exp(log_Lw)  # scalar > 0
    Lp = L * (1.0 + L / (Lw * Lw + eps)) / (1.0 + L + eps)
    scale = (Lp / (L + eps)).clamp(0.0, 16.0)  # [1,H,W]
    return scale

# -------- Hable/Uncharted2 (returns scale L'/L), channel-first --------
def hable_scale(L: torch.Tensor, log_white: torch.Tensor) -> torch.Tensor:
    # L: [1,H,W], log_white: scalar Parameter (white normalization)
    eps = 1e-8
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    def curve(x):
        return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F
    W  = torch.exp(log_white)
    Lp = curve(L)
    Wp = curve(W)
    Lp = Lp / (Wp + eps)
    scale = (Lp / (L + eps)).clamp(0.0, 16.0)  # [1,H,W]
    return scale

class DisplayMapper(nn.Module):
    """
    Differentiable display mapping (channel-first).
    Inputs/outputs are CHW tensors:
      - rgb_linear : [3,H,W] (scene-linear HDR)
      - alpha      : [1,H,W] in [0,1] (optional)
      - bg_srgb    : [3,H,W] sRGB/LDR background (optional)

    Pipeline:
      linear → exposure → luminance-preserving tone map → sRGB OETF → (optional) composite → clamp
    """
    def __init__(self, curve: str = 'reinhard', init_log_exposure: float = 0.0, init_log_white: float = 0.0):
        super().__init__()
        assert curve in ('reinhard', 'hable')
        self.curve = curve
        self.log_exposure = nn.Parameter(torch.tensor(float(init_log_exposure)))
        self.log_white    = nn.Parameter(torch.tensor(float(init_log_white)))

    @torch.no_grad()
    def init_white_from_percentile(self, rgb_linear: torch.Tensor, p: float = 0.95):
        """
        Initialize white point to a luminance percentile of a sample HDR frame.
        rgb_linear: [3,H,W]
        """
        L = luminance(rgb_linear.clamp_min(0.0)).reshape(-1)
        L = L[torch.isfinite(L)]
        if L.numel() > 0:
            q = torch.quantile(L, p)
            self.log_white.copy_(torch.log(q.clamp_min(1e-6)))

    def forward(self, rgb_linear: torch.Tensor, alpha: torch.Tensor = None, bg_srgb: torch.Tensor = None) -> torch.Tensor:
        """
        Returns sRGB/LDR in [0,1], CHW.
        """
        # 1) Exposure in linear
        # y = torch.exp(self.log_exposure) * rgb_linear.clamp_min(0.0)  # [3,H,W]
        y = rgb_linear.clamp_min(0.0)  # [3,H,W]

        # 2) Tone map on luminance; preserve hue by scaling RGB with L'/L
        L = luminance(y)  # [1,H,W]
        if self.curve == 'reinhard':
            scale = reinhard_whitepoint_scale(L, self.log_white)  # [1,H,W]
        else:
            scale = hable_scale(L, self.log_white)                # [1,H,W]
        y_tm = y * scale  # broadcast over channels

        # 3) sRGB OETF
        out_srgb = srgb_oetf(y_tm)  # [3,H,W]

        # 4) Composite background AFTER OETF (matches LDR training images)
        if (alpha is not None) and (bg_srgb is not None):
            a = alpha.clamp(0.0, 1.0)           # [1,H,W]
            out_srgb = a * out_srgb + (1.0 - a) * bg_srgb

        # 5) Final clamp for SSIM/L1
        return out_srgb.clamp(0.0, 1.0)
