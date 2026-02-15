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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, tan
from kornia.filters import spatial_gradient
from utils.graphics_utils import sample_camera_rays


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def charbonnier(x, eps=1e-3):  # robust L1
    return torch.sqrt(x*x + eps*eps).mean()

def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input + 1e-10) + (1 - target) * torch.log(1 - input + 1e-10)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def weighted_entropy_loss(img):
    zero_epsilon = 1e-6
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = -torch.mean(val * torch.log(val) + (1 - val) * torch.log(1 - val))
    return loss

def entropy_loss(alpha, eps=1e-6):
    alpha = alpha.clamp(eps, 1 - eps)
    loss = -alpha * torch.log(alpha) - (1 - alpha) * torch.log(1 - alpha)
    loss = torch.mean(loss)
    return loss

def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def total_variation_add_grad(img, w):
    '''Add gradients by total variation loss in-place'''
    wx = wy = w
    loss = wx * F.smooth_l1_loss(img[:,:,1:], img[:,:,:-1], reduction='sum') + \
        wy * F.smooth_l1_loss(img[:,1:,:], img[:,:-1,:], reduction='sum')
    loss.backward(retain_graph=True)

def pairwise_cos(n):               # n: [3,H,W] unit
    v = (n[:,:-1,:] * n[:,1:,:]).sum(0).float()   # vertical cos
    h = (n[:,:,:-1] * n[:,:,1:]).sum(0).float()   # horizontal cos
    return v, h

def rotation_invariant_normal_loss(n_ir, n_s, sigma_n=0.1): # n_ir/s: [3,H,W] unit
    # pairwise
    v_ir, h_ir = pairwise_cos(n_ir)
    v_s,  h_s  = pairwise_cos(n_s)
    L_pair = ((v_ir - v_s)**2).mean() + ((h_ir - h_s)**2).mean()

    # bilateral weights (abs cos)
    w_v = torch.exp(-(1 - v_s.abs())/sigma_n)
    w_h = torch.exp(-(1 - h_s.abs())/sigma_n)
    dv = (n_ir[:,:-1,:]-n_ir[:,1:,:]).pow(2).sum(0)
    dh = (n_ir[:,:,:-1]-n_ir[:,:,1:]).pow(2).sum(0)
    L_bilat = (w_v*dv).mean() + (w_h*dh).mean()
    return L_pair, L_bilat

def lambda_schedule(t, T, lam_max=0.1, ramp_ratio=0.1, hold_ratio=0.4, decay_ratio=0.5):
    ramp, hold, decay = ramp_ratio*T, hold_ratio*T, decay_ratio*T
    if t < ramp:                     # 0 → lam_max
        return lam_max * (t / ramp)
    elif t < ramp + hold:            # keep
        return lam_max
    else:                            # lam_max → 0
        return lam_max * (1 - (t - ramp - hold) / decay)

def lambda_schedule_iter(t, lam_max=0.1, ramp=3000, hold=6000, decay=6000):
    if t < ramp:                     # 0 → lam_max
        return lam_max * (t / ramp)
    elif t < ramp + hold:            # keep
        return lam_max
    else:                            # lam_max → 0
        return lam_max * (1 - (t - ramp - hold) / decay)

def ranking_loss(error, penalize_ratio=0.7, extra_weights=None , type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
    if extra_weights is not None:
        weights = torch.index_select(extra_weights, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
        s_error = s_error * weights

    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)

def normal_gradient_loss(rend_normal, gt_normal):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(rend_normal.device) / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(rend_normal.device) / 4

    rend_grad_x = F.conv2d(rend_normal, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    rend_grad_y = F.conv2d(rend_normal, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    gt_grad_x = F.conv2d(gt_normal, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    gt_grad_y = F.conv2d(gt_normal, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    loss_x = F.mse_loss(rend_grad_x, gt_grad_x)
    loss_y = F.mse_loss(rend_grad_y, gt_grad_y)

    return loss_x + loss_y

def energy_conservation_loss(gaussians, vis_w=None):
    s = gaussians.get_specular
    b = gaussians.get_base_color
    s = s[vis_w] if vis_w is not None else s
    b = b[vis_w] if vis_w is not None else b

    # Compute the L2 norm of s and b
    s_norm = torch.norm(s, p=float("inf"), dim=1)
    b_norm = torch.norm(b, p=float("inf"), dim=1)

    # Calculate the energy conservation loss
    total_norm = s_norm + b_norm
    loss = torch.clamp(total_norm - 1, min=0)

    # Return the mean loss for batch processing
    return loss.mean()

def disney_energy_conservation_loss(gaussians, vis_w=None, margin=0.0, detach_m=True, norm=False):
    """
    Enforce ( (1-m)*base_color + F0 ) ≤ 1 per Gaussian (∞-norm over RGB).
    gaussians: your GaussianModel
    vis_w: (N,1) visibility weights (e.g., (radii>0).float().detach())
    """
    bc = gaussians.get_base_color          # (N,3), linear, in [0,1]
    m  = gaussians.get_metallic            # (N,1), in [0,1]
    m = m.detach() if detach_m else m     # stop “cheating” by raising metallic

    bc = bc[vis_w] if vis_w is not None else bc
    m = m[vis_w] if vis_w is not None else m

    d  = (1.0 - m) * bc                   # Disney diffuse albedo
    F0 = (1.0 - m) * 0.04 + m * bc       # Disney F0

    if norm:
        d = d.norm(dim=1, keepdim=True)
        F0 = F0.norm(dim=1, keepdim=True)

    over = (d + F0).amax(dim=1, keepdim=True) - (1.0 - margin)  # (N,1)
    per_g_loss = torch.relu(over)

    return per_g_loss.mean()


def neutral_specular_loss(gaussians, detach_m=True, vis_w=None):
    # Calculate the mean across the color channels (dim=1)
    if gaussians.disney_brdf:
        b = gaussians.get_base_color          # (N,3), linear, in [0,1]
        m  = gaussians.get_metallic            # (N,1), in [0,1]
        b = b[vis_w] if vis_w is not None else b
        m = m[vis_w] if vis_w is not None else m
        m = m.detach() if detach_m else m     # stop “cheating” by raising metallic
        F0 = (1.0 - m) * 0.04 + m * b       # Disney F0
    else:
        F0 = gaussians.get_specular

    # Repeat the mean to match s's shape for RGB comparison
    F0_neutral = F0.mean(dim=1, keepdim=True).expand(-1, 3)  # Shape: [N, 3]

    # Calculate the L2 loss between s and its neutral (gray) version
    loss_neutral = torch.norm(F0 - F0_neutral, p=float("inf"), dim=1)

    return loss_neutral.mean()
