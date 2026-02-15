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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixCorrect(znear, zfar, H, W, K):
    top = (K[1,2])/K[1,1] * znear
    bottom = -(H - K[1,2])/K[1,1] * znear
    right = (K[0,2])/K[0,0] * znear
    left = -(W - K[0,2])/K[0,0] * znear
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getIntrinsicMatrix(fov_x, fov_y, width, height):
    """
    Compute 3x3 camera intrinsic matrix K from FOV and image size.

    Args:
        fov_x (float): horizontal field of view in radians
        fov_y (float): vertical field of view in radians
        width (int): image width in pixels
        height (int): image height in pixels

    Returns:
        torch.Tensor: 3x3 intrinsic matrix
    """
    fx = fov2focal(fov_x, width)
    fy = fov2focal(fov_y, height)
    cx = width / 2.0
    cy = height / 2.0

    K = torch.tensor([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=torch.float32)

    return K

def rotation_matrix(axis: str, theta: float):
    """
    Generate a 4x4 homogeneous rotation matrix for the given axis and angle.

    Args:
        axis (str): Rotation axis ('x', 'y', or 'z').
        theta (float): Rotation angle in radians.

    Returns:
        torch.Tensor: 4x4 rotation matrix.
    """
    theta = torch.tensor(theta, dtype=torch.float32)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    if axis == 'x':
        return torch.tensor([
            [1,  0,         0,        0],
            [0,  cos_theta, -sin_theta, 0],
            [0,  sin_theta,  cos_theta, 0],
            [0,  0,         0,        1]
        ], dtype=torch.float32)

    elif axis == 'y':
        return torch.tensor([
            [ cos_theta, 0, sin_theta, 0],
            [ 0,         1, 0,         0],
            [-sin_theta, 0, cos_theta, 0],
            [ 0,         0, 0,         1]
        ], dtype=torch.float32)

    elif axis == 'z':
        return torch.tensor([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta,  cos_theta, 0, 0],
            [0,         0,         1, 0],
            [0,         0,         0, 1]
        ], dtype=torch.float32)

    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

pixel_camera = None
def sample_camera_rays(viewpoint_camera, normalize=True):
    R = torch.tensor(viewpoint_camera.R, dtype=torch.float32, device="cuda")
    T = torch.tensor(viewpoint_camera.T, dtype=torch.float32, device="cuda")
    H, W, K = viewpoint_camera.HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    if normalize:
        rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)  # (N,3)
    rays_d = rays_d.reshape(H,W,3)  # (H,W,3)
    return rays_d, rays_o
