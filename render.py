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
from scene import Scene
import os
import numpy as np
import copy
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from copy import deepcopy
import open3d as o3d
from functools import partial
from scene.NVDIFFREC import util
from scene.NVDIFFREC.light import load_env_map
from utils.tonemap import estimate_tonemap, apply_tonemapping, save_tonemap_params, load_tonemap_params, gamma_tonemap

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--skip_misc", action="store_true")
    parser.add_argument("--disable_residual", action="store_true")
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--render_disney", action="store_true")
    parser.add_argument("--num_render", default=-1, type=int,  help="How many samples to render")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--gt_envmap_path", default="", help="The original envmap that was used to generate the scene")
    parser.add_argument("--relight_envmap_path", default="", help="The envmap to use to relight the scene")
    parser.add_argument("--relight_gt_path", default="", help="The relighted dataset to compare against")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset)

    if args.render_disney:
        gaussians.use_specular = False
        gaussians.disney_brdf = True
        gaussians.spec_gloss_brdf = False

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    if dataset.k_dim > 0 or dataset.sh_degree > 0:
        gaussians.activate_residual()
    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(args.model_path, 'train' + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test' + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color, num_render=args.num_render)

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir, skip_misc=args.skip_misc)

    if (not args.skip_test) or (len(scene.getTestCameras()) == 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir, skip_misc=args.skip_misc)

    if args.render_path and not args.relight_envmap_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj' + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_frames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
        gaussExtractor.reconstruction(cam_traj, traj=True)
        gaussExtractor.export_image(traj_dir, skip_misc=False, traj=True)
        # create_videos(base_dir=traj_dir,
        #             input_dir=traj_dir,
        #             out_name='render_traj',
        #             num_frames=n_frames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

    if args.relight_envmap_path:
        # deactivate residual since it's specifc to the original ground truth map
        relight_envmap_name = "".join(args.relight_envmap_path.split("/")[-1])
        print(f"Relighting using {relight_envmap_name}")
        dirname = "relight" if not args.render_path else "traj_relight"
        relight_dir = os.path.join(args.model_path, dirname + "_disney" * args.render_disney, relight_envmap_name, "ours_{}".format(scene.loaded_iter))
        os.makedirs(relight_dir, exist_ok=True)
        os.makedirs(os.path.join(relight_dir, 'envmaps'), exist_ok=True)

        if os.path.exists(args.gt_envmap_path):
            point_cloud_path = os.path.join(args.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter))

            # load gt envmap and save
            gt_envmap = util.load_image_raw(args.gt_envmap_path)
            # util.save_image_raw(os.path.join(relight_dir, 'envmaps', 'gt_envmap.hdr'), gt_envmap)
            util.save_image(os.path.join(relight_dir, 'envmaps', 'gt_envmap.png'), gt_envmap)

        # load relight envmap and save
        relight_envmap = util.load_image_raw(args.relight_envmap_path)
        # util.save_image_raw(os.path.join(relight_dir, 'envmaps', 'relight_envmap.hdr'), relight_envmap)
        util.save_image(os.path.join(relight_dir, 'envmaps', 'relight_envmap.png'), relight_envmap)

        # adapt scene for relighted dataset
        if args.relight_gt_path and not args.render_path:
            relight_dataset = deepcopy(dataset)
            relight_dataset.source_path = args.relight_gt_path
            del scene
            del dataset
            relight_scene = Scene(relight_dataset, gaussians, load_iteration=iteration, shuffle=False, relight=True)
        else:
            relight_scene = scene

        if args.render_path:
            n_frames = 240
            cam_traj = generate_path(relight_scene.getTrainCameras(), n_frames=n_frames)
        else:
            cam_traj = relight_scene.getTestCameras()

        rotate = "glossy_" in args.relight_gt_path  # only rotate envmap if glossy dataset
        roll = (lambda x: np.roll(x, x.shape[1]//4, axis=1)) if rotate else (lambda x: x)
        gaussians.load_env_map(args.relight_envmap_path, roll, rotate=rotate)

        gaussExtractor.reconstruction(cam_traj, relight=True, traj=args.render_path)
        gaussExtractor.export_image(relight_dir, skip_misc=True, traj=args.render_path)


    # apply scene editing if the multipliers differ from 1.0
    if args.edit:
        with torch.no_grad():
            # render scene with edited albedo
            edit_dir = f'edit/bgr'
            test_dir = os.path.join(args.model_path, edit_dir + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
            print(f"export edited images with settings {edit_dir} ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussians._base_color = gaussians._base_color[..., [2,1,0]]  # RGB to BGR
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir, skip_misc=True)

            # render scene with edited albedo
            edit_dir = f'edit/rbg'
            test_dir = os.path.join(args.model_path, edit_dir + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
            print(f"export edited images with settings {edit_dir} ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussians._base_color = gaussians._base_color[..., [2,0,1]]  # BGR to RBG
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir, skip_misc=True)

            # restore original albedo
            gaussians._base_color = gaussians._base_color[..., [0,2,1]]  # RBG to RGB

            # render scene with edited roughness
            edit_dir = f'edit/rougher'
            test_dir = os.path.join(args.model_path, edit_dir + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
            print(f"export edited images with settings {edit_dir} ...")
            os.makedirs(test_dir, exist_ok=True)
            temp_roughness = copy.deepcopy(gaussians.get_roughness)
            gaussians.use_residual = False
            gaussians._roughness = gaussians.inverse_roughness_activation(torch.ones_like(gaussians._roughness))
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir, skip_misc=True)
            gaussians.use_residual = True

            # render scene with edited roughness
            edit_dir = f'edit/smoother'
            test_dir = os.path.join(args.model_path, edit_dir + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
            print(f"export edited images with settings {edit_dir} ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussians._roughness = gaussians.inverse_roughness_activation(0.02 * torch.ones_like(gaussians._roughness))
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir, skip_misc=True)

            # restore original roughness
            gaussians._roughness = gaussians.inverse_roughness_activation(temp_roughness)

            # render scene with edited specular
            edit_dir = f'edit/plastic'
            test_dir = os.path.join(args.model_path, edit_dir + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
            print(f"export edited images with settings {edit_dir} ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussians.use_residual = False
            gaussians._specular = gaussians.inverse_specular_activation(0.02 * torch.ones_like(gaussians._specular))
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir, skip_misc=True)

            # render scene with edited roughness
            edit_dir = f'edit/mirror'
            test_dir = os.path.join(args.model_path, edit_dir + "_disney" * args.render_disney, "ours_{}".format(scene.loaded_iter))
            print(f"export edited images with settings {edit_dir} ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussians.use_residual = True
            gaussians._specular = gaussians.inverse_specular_activation(torch.ones_like(gaussians._specular))
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir, skip_misc=True)