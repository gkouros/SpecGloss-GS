import os
import torch
from random import randint
from utils.loss_utils import (
    l1_loss, first_order_edge_aware_loss, charbonnier,
    energy_conservation_loss, disney_energy_conservation_loss, neutral_specular_loss,
    ranking_loss, normal_gradient_loss
)
from utils.general_utils import safe_state
from fused_ssim import fused_ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import swap_env_param_in_group, colormap, tensor_to_viz
import uuid
from tqdm.auto import tqdm
from utils.image_utils import psnr, srgb_to_linear
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.NVDIFFREC.light import extract_env_map
from scene.cameras import Camera
from utils.camera_utils import get_closest_camera
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.system_utils import mkdir_p
from PIL import Image
import open3d as o3d
import numpy as np
import wandb
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, server=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        first_iter = scene.loaded_iter if scene.loaded_iter else first_iter

    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # create gaussian extractor to extract meshes from gaussians
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_light_for_log = 0.0
    ema_env_scope_for_log = 0.0
    ema_diffuse_for_log = 0.0

    first_iter += 1
    prev_camera = None
    closest_camera = None
    viewpoint_stack = None
    progress_bar = tqdm(initial=first_iter, total=opt.iterations, desc="Training progress")
    for iteration in range(first_iter, opt.iterations+1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Increase SH levels every 1000 iterations
        if iteration == opt.indirect_from_iter or (iteration > opt.indirect_from_iter and iteration % 1000 == 0):
            gaussians.oneupSHdegree()

        # enable residual mode for initial or indirect stage
        initial_stage = (iteration < opt.warmup_until_iter)
        indirect_stage = (iteration >= opt.indirect_from_iter)
        gaussians.use_residual = (initial_stage or indirect_stage)  # enable residual mode

        # Progressive Upsampling: double the resolution of the envmap
        if not initial_stage:
            if opt.brdf_envmap_upsample_interval > 0 and iteration % opt.brdf_envmap_upsample_interval == 0:
                prev_base = gaussians.brdf_mlp.raw_base
                if gaussians.brdf_mlp.double_resolution():
                    print(f"Cubemap size doubled from {prev_base.shape} to {gaussians.brdf_mlp.raw_base.shape}")
                    # update optimizer param groups with new envmap
                    swap_env_param_in_group(gaussians.optimizer, prev_base, gaussians.brdf_mlp.raw_base)

            # rebuild the mips for the envmap
            gaussians.brdf_mlp.build_mips(update=(iteration % opt.update_envmap_every == 0))

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  # list of training views
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # render the scene and scene attributes
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, initial_stage=initial_stage)

        # extract renderings
        alpha_map = render_pkg["rend_alpha"]  # (1,H,W)
        visibility_filter = render_pkg["visibility_filter"]  # (G,)
        image = render_pkg["render"]  # (3,H,W)
        image_lin = render_pkg["rend_linear"].clamp(min=0.0)

        # get GT image
        gt_image = viewpoint_cam.original_image[:3, :,:].cuda()  # (3,H,W)
        mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else torch.ones_like(alpha_map).cuda()
        gt_lin = srgb_to_linear(gt_image).clamp(min=0.0) * mask

        # Photometric losses
        Ll1 = (1 - opt.lambda_hdr) * l1_loss(image, gt_image) + opt.lambda_hdr * l1_loss(image_lin, gt_lin)
        if opt.lambda_charb:
            Ll1 += opt.lambda_charb * charbonnier(image_lin - gt_lin)
        dssim_loss = fused_ssim(image[None].clamp(0.0, 1.0), gt_image[None].clamp(0.0, 1.0))

        # Photometric loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - dssim_loss)

        # declare losses
        dist_loss, normal_loss, brdf_smooth_loss, light_loss, env_scope_loss, envmap_light_loss, diffuse_loss = \
            (torch.tensor(0.0, device=loss.device) for _ in range(7))

        # distortion loss
        if opt.lambda_dist > 0.0 and iteration >= opt.dist_loss_from_iter:
            dist_loss = opt.lambda_dist * render_pkg["rend_dist"].mean()

        # normals regularization
        if opt.lambda_normal > 0.0 and iteration > opt.normal_reg_from_iter:
            normal_loss = opt.lambda_normal * (1 - (render_pkg['rend_normal'] * render_pkg['surf_normal']).sum(dim=0))[None].mean()
        # normals prior
        if  viewpoint_cam.normal_prior is not None and opt.lambda_normal_prior > 0 and iteration >= opt.normal_prior_from_iter and iteration < opt.normal_prior_until_iter:
            normal_prior_error = 0.0
            prior_normal = viewpoint_cam.normal_prior * mask
            normal_prior_error += (1 - F.cosine_similarity(prior_normal, render_pkg['rend_normal'], dim=0))
            normal_prior_error += (1 - F.cosine_similarity(prior_normal, render_pkg['surf_normal_expected'], dim=0))
            normal_prior_error = ranking_loss(normal_prior_error[viewpoint_cam.normal_mask[0]], penalize_ratio=1.0, type='mean')
            normal_prior_loss = opt.lambda_normal_prior * normal_prior_error
            if opt.lambda_normal_gradient > 0.0:
                normal_prior_loss += opt.lambda_normal_gradient * normal_gradient_loss(prior_normal, render_pkg['surf_normal_median'])
            normal_loss += normal_prior_loss
        # smoothing of normals
        if opt.lambda_normal_smooth > 0.0 and iteration > opt.normal_reg_from_iter:
            normal_loss += opt.lambda_normal_smooth * first_order_edge_aware_loss(render_pkg['rend_normal'], gt_image)
        # smoothing of depth normals
        if opt.lambda_surf_normal_smooth > 0.0 and iteration > opt.normal_reg_from_iter:
            normal_loss += opt.lambda_surf_normal_smooth * first_order_edge_aware_loss(render_pkg['surf_normal'], gt_image)
        # smoothing of depth
        if opt.lambda_depth_smooth > 0.0 and iteration > opt.normal_reg_from_iter:
            normal_loss += opt.lambda_depth_smooth * first_order_edge_aware_loss(render_pkg['surf_depth'], gt_image)

        # brdf smoothness loss with edge-awareness
        if opt.lambda_brdf_smooth > 0.0 and iteration > opt.brdf_smooth_from_iter:
            brdf_smooth_loss += opt.lambda_brdf_smooth * first_order_edge_aware_loss(render_pkg["rend_base_color_linear"], gt_image)
            brdf_smooth_loss += opt.lambda_brdf_smooth * first_order_edge_aware_loss(render_pkg["rend_roughness"], render_pkg["rend_base_color_linear"])
            if gaussians.disney_brdf:
                brdf_smooth_loss += opt.lambda_brdf_smooth * first_order_edge_aware_loss(render_pkg["rend_metallic"], render_pkg["rend_base_color_linear"])
            if gaussians.spec_gloss_brdf or gaussians.use_specular:
                brdf_smooth_loss += opt.lambda_brdf_smooth * first_order_edge_aware_loss(render_pkg["rend_specular"], render_pkg["rend_base_color_linear"])

        # env-scope loss to penalize reflectivity of gaussians outside bounding volume for real scenes
        outside_mask = None
        if (opt.lambda_env_scope > 0 or opt.lambda_env_scope_metallic > 0 or opt.lambda_env_scope_specular > 0.0) and gaussians.env_scope_radius > 0.0:
            outside_mask = torch.sum((gaussians.get_xyz - gaussians.env_scope_center[None])**2, dim=-1) > gaussians.env_scope_radius ** 2
            roughness, metallic, specular = gaussians.get_roughness, gaussians.get_metallic, gaussians.get_specular
            glossiness = opt.envscope_roughness_threshold - roughness
            env_scope_loss += opt.lambda_env_scope * glossiness[outside_mask].mean()
            env_scope_loss += opt.lambda_env_scope_metallic * metallic[outside_mask].mean()
            env_scope_loss += opt.lambda_env_scope_specular * (specular[outside_mask] - opt.env_scope_specular_val).clamp_min(0.0).mean()

        # envmap light regularizer that encourages neutral white illumination
        if opt.lambda_envmap_reg > 0.0:
            envmap_light_loss += opt.lambda_envmap_reg * gaussians.brdf_mlp.regularizer()

        if opt.lambda_diffuse_prior > 0 and iteration >= opt.diffuse_prior_from_iter and iteration < opt.diffuse_prior_until_iter:
            diffuse_prior = viewpoint_cam.delight_prior.cuda() * mask  # (3,H,W)
            lambda_diffuse_prior = opt.lambda_diffuse_prior
            diffuse_loss += lambda_diffuse_prior * l1_loss(render_pkg["rend_diffuse_linear"], diffuse_prior)

        if opt.lambda_energy_conservation > 0.0: # and iteration > 7000:
            if gaussians.disney_brdf:
                loss += opt.lambda_energy_conservation * disney_energy_conservation_loss(gaussians, vis_w=visibility_filter, margin=0.0, detach_m=True, norm=opt.use_ec_norm)
            if gaussians.spec_gloss_brdf and gaussians.use_specular:
                loss += opt.lambda_energy_conservation * energy_conservation_loss(gaussians, vis_w=visibility_filter)

        if opt.lambda_neutral > 0.0:
            loss += opt.lambda_neutral * neutral_specular_loss(gaussians, vis_w=visibility_filter)

        # Total loss
        if iteration < opt.warmup_until_iter:
            total_loss = loss
        else:
            total_loss = loss + dist_loss + normal_loss \
                + env_scope_loss + light_loss + brdf_smooth_loss + envmap_light_loss + diffuse_loss

        # backpropagate loss
        total_loss.backward()

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_light_for_log = 0.4 * light_loss.item() + 0.6 * ema_light_for_log
            ema_env_scope_for_log = 0.4 * env_scope_loss.item() + 0.6 * ema_env_scope_for_log
            ema_diffuse_for_log = 0.4 * diffuse_loss.item() + 0.6 * ema_diffuse_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Total": f"{total_loss:.{5}f}",
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "lght": f"{ema_light_for_log:.{5}f}",
                    "diff": f"{ema_diffuse_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            elapsed = iter_start.elapsed_time(iter_end)

            # Log and save
            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/loss', ema_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/recon_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/light_loss', ema_light_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/env_scope_loss', ema_env_scope_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/diffuse_loss', ema_diffuse_for_log, iteration)
                tb_writer.add_scalar('iter_time', elapsed, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            if wandb.run:
                wandb.log({
                    "train/loss": ema_loss_for_log,
                    "train/rec_loss": Ll1.item(),
                    "train/dist_loss": ema_dist_for_log,
                    "train/normal_loss": ema_normal_for_log,
                    "train/light_loss": ema_light_for_log,
                    "train/env_scope_loss": ema_env_scope_for_log,
                    "train/diffuse_loss": ema_diffuse_for_log,
                    "train/iter_time": elapsed,
                    "train/#points": gaussians._xyz.size(0),
                })

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, render, (pipe, background))
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            opacity_reset = False
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], render_pkg["radii"][visibility_filter])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                # update outside mask after densification
                outside_mask = (gaussians.get_xyz - gaussians.env_scope_center[None]).norm(dim=-1) > gaussians.env_scope_radius if gaussians.env_scope_radius > 0.0 else None
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    opacity_reset = True
                    if opt.normal_prop_interval and not (opt.no_metallic_from_iter <= iteration < opt.no_metallic_until_iter) and gaussians.disney_brdf:
                        gaussians.reset_metallic(exclusive_mask=outside_mask)

                if not opacity_reset and opt.normal_prop_interval and iteration % opt.normal_prop_interval == 0:
                    # do not reset scale when opacity is reset
                    if opt.perturb_base_color:
                        gaussians.perturb_base_color(exclusive_mask=outside_mask)
                    gaussians.reset_scale(exclusive_mask=outside_mask)

            if not opacity_reset and ((iteration >= opt.indirect_from_iter and iteration % opt.mesh_extraction_interval == 0) or iteration == opt.indirect_from_iter) \
                    and not gaussians.env_scope_radius > 0.0:
                gaussExtractor.reconstruction(scene.getTrainCameras())
                if 'ref_real' in dataset.source_path:
                    mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                else:
                    depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                    voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                    sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                mesh_path = os.path.join(scene.model_path, "mesh")
                mkdir_p(mesh_path)
                ply_path = os.path.join(mesh_path, f'{iteration:06d}.ply')
                o3d.io.write_triangle_mesh(ply_path, mesh)
                gaussians.update_mesh(mesh)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # clip cube mipmap
            if opt.clip_envmap:
                gaussians.brdf_mlp.clamp_(min=0.0, max=1.0 if opt.clip_envmap_max else None)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        #viser!!
        if server is None or server["client"] is None:
            continue

        render_type = network_gui.on_gui_change()
        with torch.no_grad():
            client = server["client"]
            RT_w2v = viser.transforms.SE3(wxyz_xyz=np.concatenate([client.camera.wxyz, client.camera.position], axis=-1)).inverse()
            R = torch.tensor(RT_w2v.rotation().as_matrix().astype(np.float32)).numpy()
            R = R.T if "glossy_synthetic" in  dataset.source_path else R  # fix visualization control for glossy_synthetic dataset
            R = R.T if "ref_real" in  dataset.source_path else R  # fix visualization control for glossy_synthetic dataset
            T = torch.tensor(RT_w2v.translation().astype(np.float32)).numpy()
            FoVx = viewpoint_cam.FoVx
            FoVy = viewpoint_cam.FoVy

            camera = Camera(
                colmap_id=None,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=gt_image,
                gt_alpha_mask = None,
                image_name="",
                uid=None,
            )

            render_pkg = render(camera, gaussians, pipe, background, initial_stage=initial_stage)
            rendering = render_pkg["render"]

            output = None
            if render_type == "Rendered":
                output = tensor_to_viz(rendering)
            elif render_type in ["gt image", "stable normals", "stable delight diffuse"]:
                # find closest training view to the current camera
                if prev_camera is None or (camera.R != prev_camera.R).any() or (camera.T != prev_camera.T).any():
                    closest_camera = get_closest_camera(camera, scene.getTrainCameras())
                else:
                    prev_camera = camera
                if render_type == "gt image":
                    image = closest_camera.original_image[:3]  # (3,H,W)
                elif render_type == "stable delight diffuse":
                    image = (closest_camera.delight_prior[:3] if closest_camera.delight_prior is not None else torch.zeros_like(rendering))
                elif render_type == "stable normals":
                    image = (((closest_camera.normal_prior[:3] + 1) / 2.0) if closest_camera.normal_prior is not None else torch.zeros_like(rendering))
                else:
                    raise ValueError(f"Unsupported render type: {render_type}")
                output = tensor_to_viz(image)
            elif render_type == "render normal":
                output = tensor_to_viz((render_pkg["rend_normal"] + 1) / 2.0)
            elif render_type == "surf normal":
                output = tensor_to_viz((render_pkg["surf_normal"] + 1) / 2.0)
            elif render_type == "surf depth":
                max_depth = 10.0
                rend_depth = torch.clamp(render_pkg["surf_depth"], 0.0, max_depth)
                rendered_image = colormap((rend_depth / max_depth).cpu().numpy()[0], cmap='turbo', bar=False)
                output = tensor_to_viz(torch.tensor(rendered_image))
            elif render_type == "distortion":
                output = tensor_to_viz((render_pkg["rend_dist"] * 1000000).repeat(3,1,1))
            elif render_type == "base color":
                output = tensor_to_viz(render_pkg["rend_base_color"])
            elif render_type == "roughness":
                output = tensor_to_viz(render_pkg["rend_roughness"].repeat(3,1,1))
            elif render_type == "visibility":
                output = tensor_to_viz(render_pkg["rend_visibility"].repeat(3,1,1))
            elif render_type == "metallic":
                output = tensor_to_viz(render_pkg["rend_metallic"].repeat(3,1,1))
            elif render_type == "specular":
                output = tensor_to_viz(render_pkg["rend_specular"])
            elif render_type == "diffuse color":
                output = tensor_to_viz(render_pkg["rend_diffuse_color"])
            elif render_type == "specular color":
                output = tensor_to_viz(render_pkg["rend_specular_color"])
            elif render_type == "residual color":
                output = tensor_to_viz(render_pkg["rend_residual_color"])
            elif render_type == "direct light":
                output = tensor_to_viz(render_pkg["rend_direct_light"])
            elif render_type == "indirect light":
                output = tensor_to_viz(render_pkg["rend_indirect_light"])
            elif render_type == "feature map":
                output = tensor_to_viz(gaussians.mipmap.visualization())
            elif render_type == "envmap":
                output = tensor_to_viz(extract_env_map(gaussians.brdf_mlp, srgb=True))
            elif render_type == "envmap2":
                output = tensor_to_viz(extract_env_map(gaussians.brdf_mlp, rotate=True, srgb=True))
            elif render_type == "alpha map":
                output = tensor_to_viz(render_pkg["rend_alpha"].repeat(3,1,1))
            elif render_type == "bounding volume map":
                mask_map = torch.zeros_like(rendering) if render_pkg["mask_map"] is None else render_pkg["mask_map"].repeat(3,1,1)
                output = tensor_to_viz(mask_map)
            else:
                print(f"Unsupported render type: {render_type}")

            if "envmap" not in render_type:
                output = np.asarray(Image.fromarray(output).resize((400,400)))  # resize to speed up visualization
            client.scene.set_background_image(
                output,
                format="jpeg"
            )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if wandb.run:
        with open(os.path.join(args.model_path, "wandb_run_id.txt"), 'w') as wandb_id_f:
            wandb_id_f.write(wandb.run.id)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        envmap = extract_env_map(scene.gaussians.brdf_mlp, srgb=True).detach().cpu()
        envmap2 = extract_env_map(scene.gaussians.brdf_mlp, rotate=True, srgb=True).detach().cpu()
        envmap = envmap.clamp(0.0, 1.0)
        envmap2 = envmap2.clamp(0.0, 1.0)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in tqdm(enumerate(config['cameras']), total=len(config['cameras']), desc=f"Evaluating {config['name']} at iter {iteration}"):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    rend_image = render_pkg["render"].clamp(0.0, 1.0)
                    gt_image = viewpoint.original_image.to("cuda").clamp(0.0, 1.0)
                    error_map = np.linalg.norm((rend_image - gt_image).abs().permute(1,2,0).cpu().numpy(), axis=2)
                    error_cmap = colormap(error_map, cmap='turbo')

                    # log images in tensorboard and wandb
                    if idx == 0 and (tb_writer or wandb.run):
                        rend_alpha = render_pkg['rend_alpha']
                        rend_diffuse_color = render_pkg["rend_diffuse_color"]
                        rend_specular_color = render_pkg["rend_specular_color"]
                        rend_residual_color = render_pkg['rend_residual_color']
                        rend_direct_light = render_pkg["rend_direct_light"]
                        rend_indirect_light = render_pkg["rend_indirect_light"]
                        rend_base_color = render_pkg["rend_base_color"]
                        rend_roughness = render_pkg["rend_roughness"]
                        rend_metallic = render_pkg["rend_metallic"]
                        rend_specular = render_pkg["rend_specular"]
                        rend_visibility = render_pkg["rend_visibility"]
                        rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                        surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5

                        if tb_writer:
                            try:
                                prefix = f"{config['name']}_view_{viewpoint.image_name}"
                                tb_writer.add_images(f"{prefix}/gt", gt_image[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/rendering", rend_image[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/error", error_cmap[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/rend_alpha", rend_alpha[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/base_color", rend_base_color[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/diffuse", rend_diffuse_color[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/specular", rend_specular_color[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/residual_color", rend_residual_color[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/base_color", rend_base_color[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/roughness", rend_roughness[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/metallic", rend_metallic[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/specular", rend_specular[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/rend_normal", rend_normal[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/surf_normal", surf_normal[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/visibility", rend_visibility[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/direct_light", rend_direct_light[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/indirect_light", rend_indirect_light[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/envmap", envmap[None], global_step=iteration)
                                tb_writer.add_images(f"{prefix}/envmap2", envmap2[None], global_step=iteration)
                            except:
                                pass

                        # log images in wandb
                        if wandb.run:
                            wandb.log({f"{config['name']}/images": [
                                wandb.Image(gt_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="GT"),
                                wandb.Image(rend_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Rendering"),
                                wandb.Image(error_cmap.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Error map"),
                                wandb.Image(rend_alpha.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Alpha"),
                                wandb.Image(rend_diffuse_color.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Diffuse Color"),
                                wandb.Image(rend_specular_color.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Specular Color"),
                                wandb.Image(rend_residual_color.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Residual Color"),
                                wandb.Image(rend_base_color.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Base Color"),
                                wandb.Image(rend_roughness.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Roughness"),
                                wandb.Image(rend_metallic.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Metallic"),
                                wandb.Image(rend_specular.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Specular"),
                                wandb.Image(rend_normal.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Normal"),
                                wandb.Image(surf_normal.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Depth Normal"),
                                wandb.Image(rend_visibility.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Visibility"),
                                wandb.Image(rend_direct_light.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Direct Light"),
                                wandb.Image(rend_indirect_light.clamp(0, 1).permute(1, 2, 0).cpu().numpy(), caption="Indirect Light"),
                            ]}, step=iteration)

                    l1_test += l1_loss(rend_image, gt_image).mean().double()
                    psnr_test += psnr(rend_image.unsqueeze(0).contiguous(), gt_image.unsqueeze(0).contiguous()).item()

                l1_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if wandb.run:
                    wandb.log({f"{config['name']}/mse": l1_test, f"{config['name']}/psnr": psnr_test}, step=iteration)

        if wandb.run:
            wandb.log({"test/envmap": [
                wandb.Image(envmap.permute(1, 2, 0).cpu().numpy(), caption="envmap"),
                wandb.Image(envmap2.permute(1, 2, 0).cpu().numpy(), caption="envmap2"),
                ]}, step=iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.save_iterations.append(args.residual_from_iter)
    for iterations in range(args.test_iterations[-1] + 5000, args.iterations, 5000):
        args.test_iterations.append(iterations)
    if args.iterations not in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.gui:
        import viser
        server = network_gui.init()
    else:
        server = None

    if args.wandb:
        for id_name in ["wandb_run_id" , "wandb_id", "run_id"]:
            run_id_path = os.path.join(args.model_path, f"{id_name}.txt")
            if os.path.exists(run_id_path):
                with open(run_id_path, 'r') as f:
                    args.run_id = f.read().strip()
                print(f"File {id_name}.txt found. Logging metrics in session with id {args.run_id}.")
                break
        else:
            print(f"No {id_name}.txt file found. Logging metrics in session with id {args.run_id}.")

        exp_name, dataset_name, scene_name = args.model_path.split('/')[-3:]
        wandb.init(
            project="refgshader",  # set the wandb project where this run will be logged
            config=vars(args),  # track hyperparameters and run metadata
            group=dataset_name,
            name=f"{scene_name}.{exp_name}",
            job_type=exp_name.split('.')[0],
            id=args.run_id,
            resume="allow",
        )

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, server)

    # All done
    print("\nTraining complete.")