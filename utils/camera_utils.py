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
import torch
from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    K = cam_info.K.copy() if cam_info.K is not None else None

    if args.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * args.resolution
        resolution = round(orig_w / scale), round(orig_h / scale)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if cam_info.K is not None:
        K[:2] = K[:2] / scale

    HWK = (resolution[1], resolution[0], K)

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution, rescale=True)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution, rescale=True)
        loaded_mask = None
        gt_image = resized_image_rgb

    delight_prior = None
    if cam_info.delight_prior is not None:
        delight_prior = PILtoTorch(cam_info.delight_prior, resolution, rescale=True)
        alpha = delight_prior[3:4] if delight_prior.shape[0] == 4 else 1.0
        delight_prior = delight_prior[:3] * alpha + (1.0 - alpha) * args.white_background  # blend with alpha channel

    normal_prior = None
    if cam_info.normal_prior is not None:
        normal_prior = PILtoTorch(cam_info.normal_prior, resolution, rescale=True)
        alpha = normal_prior[3:4] if normal_prior.shape[0] == 4 else 1.0
        normal_prior = normal_prior[:3]
        normal_prior = (normal_prior * 2 - 1)  # (3,H,W)
        normal_prior = -normal_prior  # invert to match with learnable normals
        normal_prior = normal_prior.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_info.R)).float())
        normal_prior = normal_prior.permute(2, 0, 1)  # (3,H,W)
        normal_prior = normal_prior * alpha #+ (1.0 - alpha) * args.white_background  # blend with alpha channel

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device, HWK=HWK,
                  normal_prior=normal_prior, delight_prior=delight_prior,
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def get_closest_camera(camera, cameras):
    closest_camera = None
    closest_dist = float('inf')
    def rot_gap(R_ref, R):
        angle = np.arccos(np.clip((np.trace(R.T @ R_ref) - 1)/2, -1.0, 1.0))
        return angle  # radians
    def trans_gap(t_ref, t):
        return np.linalg.norm(t_ref - t)
    def cost(R_ref, t_ref, R, t, w_r, w_t):
        return np.hypot(w_r*rot_gap(R_ref, R), w_t*trans_gap(t_ref, t))
    for c in cameras:
        dist = cost(camera.R, camera.T, c.R, c.T, w_r=1.0, w_t=1.0)
        if dist < closest_dist:
            closest_dist = dist
            closest_camera = c
    return closest_camera
